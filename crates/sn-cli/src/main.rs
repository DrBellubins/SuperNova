//! `sn` — the SuperNova command line.
//!
//! `Docs/design-plan.md` §6 puts one binary in front of every phase:
//!
//! ```text
//!   sn dump --plan     execution plan + per-token weight budget   (Phase 1)
//!   sn ref             CPU reference forward + logits             (Phase 2)
//!   sn bench           kernel timing vs roofline                  (Phase 3+)
//!   sn bench-hip       standalone kernel micro-bench              (Phase 4+)
//! ```

use std::fmt::Write as _;

use sn_core::budget;
use sn_core::gguf::Gguf;
use sn_core::plan::{Hyperparams, Plan};

const USAGE: &str = "\
sn — SuperNova inference engine

usage:
  sn dump --plan <model.gguf> [--json]     execution plan + per-token weight budget
  sn dump --tensors <model.gguf> [--json]  per-tensor table (name, dims, type, bytes)
  sn dump --meta <model.gguf>              GGUF metadata
  sn ref --model <model.gguf> (--tokens <id,id,...> | --tokens-file <f>)
         [--dump <dir>] [--dump-only p1,p2] [--iters N] [--limit-exec N] [--threads N] [--top N] [--json]
         [--logits-out <f>] [--f16-weights] [--f16-acts] [--kv-f16] [--attn-f16] [--rope-adjacent]
                                             reference forward + logits (Phase 2)
  sn hip [info]                            device, slabs, kernel image (Phase 3)
  sn bench graph [--ops N] [--reps N]      the Phase 3 graph-replay gate

Token ids come from the reference tokenizer for now (`Tools/tok --ids`); the GGUF BPE tokenizer
is Phase 10 work (Docs/design-plan.md §7).
";

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let out = run(&argv);
    match out {
        Ok(text) => {
            print!("{text}");
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("sn: error: {e}");
            std::process::exit(1);
        }
    }
}

fn run(argv: &[String]) -> sn_core::Result<String> {
    let flag = |n: &str| argv.iter().any(|a| a == n);
    let opt_val = |n: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == n)
            .and_then(|i| argv.get(i + 1))
            .cloned()
    };
    let positional: Vec<String> = argv.iter().filter(|s| !s.starts_with('-')).cloned().collect();

    match argv.first().map(|s| s.as_str()) {
        None | Some("-h") | Some("--help") | Some("help") => Ok(USAGE.to_string()),
        Some("info") => Ok(info()),
        Some("dump") => {
            let path = positional
                .get(1)
                .ok_or_else(|| sn_core::Error::msg("dump needs a model path"))?;
            let g = Gguf::open(path)?;
            if flag("--meta") {
                dump_meta(&g)
            } else if flag("--tensors") {
                dump_tensors(&g, flag("--json"))
            } else if flag("--plan") {
                dump_plan(&g, flag("--json"))
            } else {
                Ok("sn: dump needs one of --plan / --tensors / --meta\n".into())
            }
        }
        Some("ref") => cmd_ref(argv, &positional, opt_val, flag),
        Some("hip") => hip::cmd_hip(&positional, opt_val, flag),
        Some("bench") => hip::cmd_bench(&positional, opt_val),
        Some("deq") => cmd_deq(&positional, opt_val),
        Some(cmd) => Err(sn_core::Error::msg(format!(
            "unknown command '{cmd}' (try: sn --help)"
        ))),
    }
}

/// Phase 2 debug tool: write one dequantised weight row as text, for Tools/dequant_ref.py to
/// compare against a transliteration of GGML's own dequantiser.
fn cmd_deq(positional: &[String], opt_val: impl Fn(&str) -> Option<String>) -> sn_core::Result<String> {
    use std::io::Write as _;
    let path = opt_val("--model")
        .or_else(|| positional.get(1).cloned())
        .ok_or_else(|| sn_core::Error::msg("deq needs --model <gguf>"))?;
    let name = opt_val("--tensor").ok_or_else(|| sn_core::Error::msg("deq needs --tensor <name>"))?;
    let row = opt_val("--row")
        .map(|v| v.parse::<usize>())
        .transpose()
        .map_err(|_| sn_core::Error::msg("--row wants a non-negative integer"))?
        .unwrap_or(0);
    let out_path = opt_val("--out").ok_or_else(|| sn_core::Error::msg("deq needs --out <file>"))?;
    let g = Gguf::open(&path)?;
    let m = sn_core::dq::Mat::by_name(&g, &name)?;
    if row >= m.ne1 {
        return Err(sn_core::Error::msg(format!(
            "{name}: row {row} is outside the {} output rows",
            m.ne1
        )));
    }
    let mut v = vec![0.0f32; m.ne0];
    m.dequant_row(row, &mut v);
    let mut f = std::io::BufWriter::new(
        std::fs::File::create(&out_path).map_err(|e| sn_core::Error::msg(format!("{out_path}: {e}")))?,
    );
    writeln!(f, "# dequantised row {row} of {name} ({} elements)", m.ne0)
        .map_err(|e| sn_core::Error::msg(format!("{out_path}: {e}")))?;
    for x in &v {
        writeln!(f, "{x:.9e}").map_err(|e| sn_core::Error::msg(format!("{out_path}: {e}")))?;
    }
    Ok(format!("{name} row {row}: {} values -> {out_path}\n", m.ne0))
}

/// Phase 2 reference forward: token ids in, logits out, optional per-stage dumps.
fn cmd_ref(
    argv: &[String],
    positional: &[String],
    opt_val: impl Fn(&str) -> Option<String>,
    flag: impl Fn(&str) -> bool,
) -> sn_core::Result<String> {
    let _ = argv;
    let path = match opt_val("--model") {
        Some(p) => p,
        None => positional
            .get(1)
            .cloned()
            .ok_or_else(|| sn_core::Error::msg("ref needs a model path"))?,
    };
    let tokens = match (opt_val("--tokens"), opt_val("--tokens-file"), opt_val("--prompt-file")) {
        (Some(s), _, _) => parse_ids(&s)?,
        (_, Some(f), _) => parse_ids(&std::fs::read_to_string(&f)?)
            .map_err(|e| sn_core::Error::msg(format!("{f}: {e}")))?,
        // A "prompt file" is accepted when it holds ids; the GGUF BPE tokenizer is Phase 10.
        (_, _, Some(f)) => {
            let text = std::fs::read_to_string(&f)?;
            parse_ids(&text).map_err(|_| {
                sn_core::Error::msg(format!(
                    "{f}: no tokenizer in this build yet (Phase 10). Give token ids \
                     (--tokens / --tokens-file); Tools/tokenize.sh produces them from the reference tokenizer."
                ))
            })?
        }
        _ => return Err(sn_core::Error::msg("ref needs --tokens, --tokens-file or --prompt-file")),
    };
    if tokens.is_empty() {
        return Err(sn_core::Error::msg("no token ids given"));
    }
    let iters = match opt_val("--iters") {
        Some(v) => Some(v.parse::<usize>().map_err(|_| {
            sn_core::Error::msg(format!("--iters wants a positive integer, got {v}"))
        })?),
        None => None,
    };
    let threads = match opt_val("--threads") {
        Some(v) => Some(v.parse::<usize>().map_err(|_| {
            sn_core::Error::msg(format!("--threads wants a positive integer, got {v}"))
        })?),
        None => None,
    };
    let limit_exec = match opt_val("--limit-exec") {
        Some(v) => Some(v.parse::<usize>().map_err(|_| {
            sn_core::Error::msg(format!("--limit-exec wants a non-negative integer, got {v}"))
        })?),
        None => None,
    };
    let top = opt_val("--top")
        .map(|v| v.parse::<usize>())
        .transpose()
        .map_err(|_| sn_core::Error::msg("--top wants an integer"))?
        .unwrap_or(5);

    let t0 = std::time::Instant::now();
    let g = Gguf::open(&path)?;
    let plan = Plan::build(Hyperparams::from_gguf(&g)?)?;
    let only: Vec<String> = opt_val("--dump-only")
        .map(|v| v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
        .unwrap_or_default();
    let dump = sn_core::forward::Dumper::with_filter(
        opt_val("--dump").map(std::path::PathBuf::from), &only)?;
    let mut model = sn_core::forward::RefModel::new(&plan, &g)?;
    if let Some(t) = threads {
        model.set_threads(t);
    }
    model.set_limit_exec(limit_exec);
    model.set_f16_operands(flag("--f16-weights"), flag("--f16-acts"));
    model.set_kv_f16(flag("--kv-f16"));
    model.set_attn_f16(flag("--attn-f16"));
    model.set_rope_adjacent(flag("--rope-adjacent"));
    let logits = model.run(&tokens, iters, &dump)?;
    let elapsed = t0.elapsed();

    // `--logits-out` keeps the whole score vector for Tools/golden-check.py; the top-K table alone
    // is not enough to compute an NMSE against the reference distribution.
    if let Some(path) = opt_val("--logits-out") {
        use std::io::Write as _;
        let mut f = std::io::BufWriter::new(
            std::fs::File::create(&path)
                .map_err(|e| sn_core::Error::msg(format!("{path}: {e}")))?,
        );
        writeln!(f, "# logits for {} prompt token(s), mhc.iterations = {}", tokens.len(), iters.unwrap_or(plan.hp.mhc_iters))?;
        for v in &logits {
            writeln!(f, "{v:.9e}")
                .map_err(|e| sn_core::Error::msg(format!("{path}: {e}")))?;
        }
    }

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));

    if flag("--json") {
        let tps = tokens.len() as f64 / elapsed.as_secs_f64();
        let mut j = Json::new();
        j.begin(None, "{");
        j.field("model", json_str(&path));
        j.field("tokenizer", json_str("reference /tokenize"));
        j.field_num("n_prompt_tokens", tokens.len());
        j.field_num("iters", iters.unwrap_or(plan.hp.mhc_iters));
        j.field_num("top_token", order[0]);
        j.field_num("top_logit", logits[order[0]]);
        j.field_num("seconds", elapsed.as_secs_f64());
        j.field_num("tokens_per_second", tps);
        j.begin(Some("top_k"), "[");
        for &i in order.iter().take(top) {
            j.begin(None, "{");
            j.field_num("token", i);
            j.field_num("logit", logits[i]);
            j.end("}");
        }
        j.end("]");
        j.end("}");
        return Ok(j.finish());
    }

    let mut s = String::new();
    writeln!(
        s,
        "reference forward: {} prompt token(s), mhc.iterations = {}, {} ms ({:.1} tok/s)",
        tokens.len(),
        iters.unwrap_or(plan.hp.mhc_iters),
        elapsed.as_millis(),
        tokens.len() as f64 / elapsed.as_secs_f64()
    )?;
    writeln!(s, "top {top} continuation tokens:")?;
    for (rank, &i) in order.iter().take(top).enumerate() {
        writeln!(s, "  {:>3}  token {i:<7} logit {:>9.4}", rank + 1, logits[i])?;
    }
    if let Some(d) = opt_val("--dump") {
        writeln!(s, "stage dumps written to {d}")?;
    }
    Ok(s)
}

/// "1,2,3" or one id per line, or any mix of whitespace, commas and colons.
fn parse_ids(s: &str) -> sn_core::Result<Vec<u32>> {
    let mut out = Vec::new();
    for part in s.split(|c: char| c.is_whitespace() || c == ',' || c == ':') {
        if part.is_empty() {
            continue;
        }
        let v: u32 = part.parse().map_err(|_| {
            sn_core::Error::msg(format!(
                "expected a token id, got {part:?} (this build has no tokenizer yet)"
            ))
        })?;
        out.push(v);
    }
    Ok(out)
}

fn info() -> String {
    let mut s = String::new();
    writeln!(s, "SuperNova {}", env!("CARGO_PKG_VERSION")).unwrap();
    writeln!(s, "  target arch   : {}", std::env::consts::ARCH).unwrap();
    writeln!(
        s,
        "  implemented   : GGUF reader + execution plan + weight budget (Phases 0-1)"
    )
    .unwrap();
    writeln!(s, "  plan          : Docs/design-plan.md (Phase 0 baseline: Docs/baseline.md)").unwrap();
    s
}

/// Minimal JSON emitter: no dependency, and it keeps brace escaping out of format strings.
struct Json {
    s: String,
    indent: usize,
    first: bool,
}

impl Json {
    fn new() -> Self {
        Json { s: String::new(), indent: 0, first: true }
    }
    fn nl(&mut self) {
        self.s.push('\n');
        for _ in 0..self.indent {
            self.s.push_str("  ");
        }
    }
    fn sep(&mut self) {
        if !self.first {
            self.s.push(',');
        }
        self.first = false;
        self.nl();
    }
    fn begin(&mut self, key: Option<&str>, open: &str) {
        self.sep();
        if let Some(k) = key {
            write!(self.s, "{}: ", json_str(k)).unwrap();
        }
        self.s.push_str(open);
        self.indent += 1;
        self.first = true;
    }
    fn end(&mut self, close: &str) {
        self.indent -= 1;
        if !self.first {
            self.nl();
        }
        self.s.push_str(close);
        self.first = false;
    }
    /// `"key": <raw value text>`
    fn field(&mut self, key: &str, value: String) {
        self.sep();
        write!(self.s, "{}: {value}", json_str(key)).unwrap();
    }
    fn field_num(&mut self, key: &str, v: impl std::fmt::Display) {
        self.field(key, v.to_string());
    }
    fn finish(self) -> String {
        let mut s = self.s;
        s.push('\n');
        s
    }
}

fn dump_meta(g: &Gguf) -> sn_core::Result<String> {
    let mut s = String::new();
    writeln!(
        s,
        "file             : {} ({} bytes, GGUF v{}, align {})",
        g.path().display(),
        g.file_len(),
        g.version(),
        g.alignment
    )?;
    writeln!(s, "tensors          : {}", g.tensors().len())?;
    writeln!(s, "kv pairs         : {}", g.kv_len())?;
    writeln!(s, "data section     : {}", g.data_start)?;
    writeln!(s)?;
    for (k, v) in g.kv() {
        writeln!(s, "  {k:55} = {v}")?;
    }
    Ok(s)
}

fn dump_tensors(g: &Gguf, as_json: bool) -> sn_core::Result<String> {
    let mut s = String::new();
    if as_json {
        let mut j = Json::new();
        j.begin(None, "[");
        for t in g.tensors() {
            j.begin(None, "{");
            j.field("name", json_str(&t.name));
            j.field("dims", json_nums(&t.ne[..t.n_dims as usize]));
            j.field("type", json_str(&sn_core::qtypes::name(t.ty)));
            j.field_num("n_elts", t.n_elts());
            j.field_num("bytes", t.size);
            j.field_num("offset", t.offset);
            j.end("}");
        }
        j.end("]");
        j.end("}");
        return Ok(j.finish());
    }
    writeln!(
        s,
        "{:<48}{:>5}{:<20}{:<8}{:>14}{:>14}",
        "name", "dims", "shape", "type", "elements", "bytes"
    )?;
    for t in g.tensors() {
        let dims = &t.ne[..t.n_dims as usize];
        writeln!(
            s,
            "{:<48}{:>5}{:<20}{:<8}{:>14}{:>14}",
            t.name,
            dims.len(),
            format!("{dims:?}"),
            sn_core::qtypes::name(t.ty),
            t.n_elts(),
            t.size
        )?;
    }
    Ok(s)
}

fn dump_plan(g: &Gguf, as_json: bool) -> sn_core::Result<String> {
    let hp = Hyperparams::from_gguf(g)?;
    let plan = Plan::build(hp)?;
    let bud = budget::build(&plan, g)?;
    let hp = &plan.hp;

    if as_json {
        let mut j = Json::new();
        j.begin(None, "{");
        j.begin(Some("model"), "{");
        j.field("name", json_str(&hp.name));
        j.field("arch", json_str(&hp.arch));
        j.field("quant", json_str(&hp.quant_label));
        j.field("file", json_str(&g.path().display().to_string()));
        j.field_num("file_bytes", g.file_len());
        j.field_num("gguf_version", g.version());
        j.end("}");

        j.begin(Some("layers"), "{");
        j.field_num("n_layer_all", hp.n_layer_all);
        j.field_num("n_layer_nextn", hp.n_layer_nextn);
        j.field_num("n_layer", plan.n_layer);
        j.field_num("n_physical", plan.n_physical);
        j.field_num("n_physical_trunk", plan.n_physical - plan.hp.n_layer_nextn);
        j.field_num("extra_visits", plan.extra);
        j.field(
            "loop",
            format!("[{}, {}, {}]", hp.loop_start, hp.loop_end, hp.loop_count),
        );
        j.field(
            "mhc",
            format!(
                "{{\"start\": {}, \"end\": {}, \"iters\": {}, \"streams\": {}, \"pre_scale\": {}, \"fixed_identity\": {}}}",
                hp.mhc_start, hp.mhc_end, hp.mhc_iters, hp.mhc_streams, hp.mhc_pre_scale,
                hp.mhc_fixed_identity as u8
            ),
        );
        j.end("}");

        j.begin(Some("widths"), "{");
        j.field_num("n_embd", hp.n_embd);
        j.field_num("n_ffn", hp.n_ffn);
        j.field_num("n_head", hp.n_head);
        j.field_num("n_head_kv", hp.n_head_kv);
        j.field_num("head_dim", hp.head_dim);
        j.field_num("n_rot", hp.n_rot);
        j.field("rope_sections", json_nums(&hp.rope_sections.iter().map(|v| *v as u64).collect::<Vec<_>>()));
        j.field("rope", format!(
            "{{\"type\": {}, \"base\": {}, \"factor\": {}, \"original_ctx\": {}}}",
            json_str(&hp.rope_type), hp.rope_base, hp.yarn_factor, hp.ctx_orig
        ));
        j.field_num("d_conv", hp.d_conv);
        j.field_num("d_state", hp.d_state);
        j.field_num("n_group", hp.n_group);
        j.field_num("dt_rank", hp.dt_rank);
        j.field_num("d_inner", hp.d_inner);
        j.field_num("n_v_heads", hp.dims.n_v_heads);
        j.field_num("rms_eps", hp.rms_eps);
        j.field_num("vocab", hp.vocab);
        j.end("}");

        j.begin(Some("executions"), "{");
        j.field_num("total", plan.n_exec());
        j.field_num("gdn", plan.n_gdn_executions());
        j.field_num("full_attention", plan.n_attn_executions());
        j.field_num("mhc", plan.n_mhc_executions());
        j.field_num("mhc_routes", plan.n_mhc_executions() * 2);
        j.end("}");

        j.begin(Some("weights"), "{");
        j.field_num("tensors", bud.totals.n_tensors);
        j.field_num("stored_bytes", bud.totals.stored_bytes);
        j.field_num("bytes_per_token", bud.totals.bytes_per_token);
        j.field("re_read_factor", format!("{:.6}", bud.totals.re_read_factor));
        j.end("}");

        j.begin(Some("caches"), "{");
        j.field_num("n_kv_layers", budget::n_kv_layers(&plan));
        j.field_num("kv_bytes_per_token_f16", budget::kv_bytes_per_token(&plan, 2));
        j.field_num("kv_bytes_per_token_f32", budget::kv_bytes_per_token(&plan, 4));
        j.field_num("kv_bytes_per_token_q8", budget::kv_bytes_per_token(&plan, 1));
        j.field_num("kv_bytes_per_token_unique_f16_diagnostic", budget::kv_bytes_per_token_unique(&plan, 2));
        j.field_num("gdn_state_bytes_per_sequence", budget::gdn_state_bytes(&plan));
        j.field_num("gdn_state_bytes_per_layer", budget::gdn_state_per_layer(&plan));
        j.field_num("gdn_state_bytes_unique_diagnostic", budget::gdn_state_bytes_unique(&plan));
        j.end("}");

        j.begin(Some("tensors"), "[");
        for r in &bud.rows {
            j.begin(None, "{");
            j.field("name", json_str(&r.name));
            j.field("class", json_str(&r.class));
            j.field("dims", json_nums(&r.dims[..r.n_dims]));
            j.field("type", json_str(&r.type_name));
            j.field_num("n_elts", r.n_elts);
            j.field_num("bytes", r.stored_bytes);
            if r.gguf_block == usize::MAX {
                j.field("gguf_block", "null".into());
            } else {
                j.field_num("gguf_block", r.gguf_block);
            }
            j.field_num("visits", r.visits);
            j.field_num("bytes_per_token", r.bytes_per_token);
                j.end("}");
        }
        j.end("]");
        j.end("}");
        return Ok(j.finish());
    }

    let mut s = String::new();
    writeln!(
        s,
        "model            : {} [{}]  {}",
        hp.name,
        hp.quant_label,
        g.path().display()
    )?;
    writeln!(
        s,
        "layers           : {} declared = {} trunk + {} MTP   physical {} (incl MTP)   loop [{},{}) x{}   extra visits {}",
        hp.n_layer_all, plan.n_layer, hp.n_layer_nextn, plan.n_physical,
        hp.loop_start, hp.loop_end, hp.loop_count, plan.extra
    )?;
    writeln!(
        s,
        "mHC              : blocks [{},{})   streams {}   sweeps {}   pre_scale {}   fixed_identity {}",
        hp.mhc_start, hp.mhc_end, hp.mhc_streams, hp.mhc_iters, hp.mhc_pre_scale, hp.mhc_fixed_identity
    )?;
    writeln!(
        s,
        "widths           : embd {} ffn {}   heads {}/{}x{} rope {}/{}   GDN inner {} state {} heads {}/{} conv {} dt {}",
        hp.n_embd, hp.n_ffn, hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_rot, hp.head_dim,
        hp.d_inner, hp.d_state, hp.dims.n_v_heads, hp.dims.n_k_heads, hp.d_conv, hp.dt_rank
    )?;
    writeln!(
        s,
        "rope             : {} base {} factor {} orig_ctx {} sections {:?}",
        hp.rope_type, hp.rope_base, hp.yarn_factor, hp.ctx_orig, hp.rope_sections
    )?;
    writeln!(
        s,
        "executions/token : {} = {} GDN + {} full-attn (incl {} MTP)   mHC routers {} ({} route applications)",
        plan.n_exec(),
        plan.n_gdn_executions(),
        plan.n_attn_executions(),
        hp.n_layer_nextn,
        plan.n_mhc_executions(),
        plan.n_mhc_executions() * 2
    )?;
    writeln!(
        s,
        "weights          : {:.3} GB stored -> {:.3} GB streamed per token   (x{:.2} loop re-read)",
        bud.totals.stored_bytes as f64 / 1e9,
        bud.totals.bytes_per_token as f64 / 1e9,
        bud.totals.re_read_factor
    )?;
    writeln!(
        s,
        "caches           : {} KV layers (one per full-attn visit) = {}/token f16, {}/token q8   GDN state {:.1} MB/sequence ({} layers x {:.1} MB)",
        budget::n_kv_layers(&plan),
        budget::kv_bytes_per_token(&plan, 2),
        budget::kv_bytes_per_token(&plan, 1),
        budget::gdn_state_bytes(&plan) as f64 / 1e6,
        plan.n_gdn_executions(),
        budget::gdn_state_per_layer(&plan) as f64 / 1e6
    )?;

    writeln!(s)?;
    writeln!(s, "--- weight budget by class ---------------------------------------------")?;
    writeln!(
        s,
        "{:<40}{:>4}{:>13}{:>6}{:>12}",
        "class", "n", "stored MB", "exec", "MB/token"
    )?;
    for c in &bud.classes {
        writeln!(
            s,
            "{:<40}{:>4}{:>13.1}{:>6}{:>12.1}",
            c.class,
            c.n_tensors,
            c.stored_bytes as f64 / 1e6,
            c.visits,
            c.bytes_per_token as f64 / 1e6
        )?;
    }
    writeln!(s)?;
    writeln!(
        s,
        "{:<40}{:>4}{:>13.1}{:>6}{:>12.1}",
        "TOTAL",
        bud.totals.n_tensors,
        bud.totals.stored_bytes as f64 / 1e6,
        0,
        bud.totals.bytes_per_token as f64 / 1e6
    )?;

    writeln!(s)?;
    writeln!(s, "--- execution plan ---------------------------------------------------")?;
    writeln!(
        s,
        "{:>7}{:>9}{:>11}{:>5}{:>4}  {:<12}",
        "logical", "physical", "gguf_blk", "mhc", "mtp", "attention"
    )?;
    for e in &plan.execs {
        writeln!(
            s,
            "{:>7}{:>9}{:>11}{:>5}{:>4}  {}",
            e.logical,
            e.physical,
            e.tensor_block,
            e.mhc as usize,
            e.mtp,
            if e.recurrent { "gdn" } else { "full" }
        )?;
    }
    Ok(s)
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_nums<T: std::fmt::Display>(v: &[T]) -> String {
    let parts: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

mod hip;
