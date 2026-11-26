#!/usr/bin/env python3
"""Phase 2 gate: compare `sn ref` against the captured llama.cpp reference.

Inputs are the frozen corpus (Docs/golden/corpus.json), the token ids
(Docs/golden/tokens/<id>.ids) and the reference distributions
(Docs/golden/ref/<id>.json.gz, produced by Tools/golden-ref.py). For every prompt this runs

    sn ref --tokens-file <ids> --logits-out <file> [--iters N]

and scores the resulting score vector against the reference log-probs. Log-probs and logits differ
by a constant (the log-sum-exp), which cancels in a difference, so `lp_ref - lp_mine` is exactly
the logit difference.

Gate (Docs/design-plan.md §7, Phase 2):
  * top-1 token matches the reference for every prompt in the corpus;
  * NMSE of the centred-logit difference <= --nmse, or - when --floor-dir gives a second capture
    of the same prompts from another backend (CPU vs GPU) - <= --margin x that measured floor.
    The floor matters: this model's own GPU and CPU builds differ by logit NMSE ~2e-3..1e-2, so a
    different implementation cannot be expected to sit below it (see Docs/numerics.md).
  * with --iters 6 the logits must be visibly different from the 20-sweep run (reported, not gated
    by NMSE).

Reported but not gated: KL(ref || mine), max |d logp| over the reference's top-K, and the rank of
our top-1 inside the reference's ordering (a near-miss is worth seeing even when top-1 passes).

Usage:
  Tools/golden-check.py                          # all prompts, mhc.iterations as stored
  Tools/golden-check.py --prompts p1-capital     # one prompt
  Tools/golden-check.py --iters 6 --compare-iters 20
  Tools/golden-check.py --json out.json
Re-running with the same --iters reuses the cached logits under --work-dir; delete the cache (or
pass a fresh --work-dir) after any change to the forward.
"""
import argparse, gzip, json, math, os, subprocess, sys

DEF_MODEL = "/home/justin/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf"


def log_softmax(logits):
    # log p_i = x_i - max - log sum exp(x_j - max). Subtracting log(max) instead of max shifted
    # every value by ~max and made the NMSE meaningless (it reported 4.18e-01 for a distribution
    # whose true per-token differences were <= 0.5).
    m = max(logits)
    s = sum(math.exp(x - m) for x in logits)
    return [x - m - math.log(s) for x in logits]


def read_logits(path):
    vals = []
    for line in open(path):
        if line.startswith("#"):
            continue
        vals.append(float(line))
    return vals


def ref_logprobs(path, n_vocab):
    """-> list of log-probs indexed by token id, and the reference's descending order."""
    d = json.load(gzip.open(path))
    lp = [float("-inf")] * n_vocab
    for tok, p in d["logprobs"]:
        lp[int(tok)] = float(p)
    return d, lp


def centred_nmse(a, b):
    """NMSE of (a - b) against the centred magnitude of b, ignoring entries neither has."""
    fin = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if not fin:
        return float("inf")
    mean = sum(y for _, y in fin) / len(fin)
    num = sum((x - y) ** 2 for x, y in fin)
    den = sum((y - mean) ** 2 for _, y in fin)
    return num / max(1e-30, den)


def rank_of(order, token):
    try:
        return order.index(token)
    except ValueError:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="Docs/golden/corpus.json")
    ap.add_argument("--ref-dir", default="Docs/golden/ref")
    ap.add_argument("--tokens-dir", default="Docs/golden/tokens")
    ap.add_argument("--sn", default="target/release/sn")
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--work-dir", default="/tmp/sn-golden")
    ap.add_argument("--prompts", default="", help="comma-separated ids (default: all)")
    ap.add_argument("--iters", type=int, default=0, help="override mhc.iterations")
    ap.add_argument("--compare-iters", type=int, default=0,
                    help="with a non-default --iters, diff against the cached logits of this run")
    ap.add_argument("--nmse", type=float, default=1e-4)
    ap.add_argument("--floor-dir", help="second reference capture (another backend) to measure the "
                                       "reference's own implementation spread; the NMSE gate then "
                                       "uses max(--nmse, --margin * floor)")
    ap.add_argument("--margin", type=float, default=3.0, help="how far above that floor to allow")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--json", help="also write a machine-readable summary here")
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    corpus = json.load(open(args.corpus))
    want = {p.strip() for p in args.prompts.split(",") if p.strip()}
    prompts = [p for p in corpus["prompts"] if not want or p["id"] in want]
    if not prompts:
        sys.exit(f"no corpus prompts selected (have: {[p['id'] for p in corpus['prompts']]})")

    rows, failures = [], []
    for pr in prompts:
        pid = pr["id"]
        ids = [int(x) for x in open(f"{args.tokens_dir}/{pid}.ids").read().replace(",", " ").split()]
        ref_path = f"{args.ref_dir}/{pid}.json.gz"
        if not os.path.exists(ref_path):
            sys.exit(f"{ref_path} missing - run Tools/golden-ref.py first")
        rd, lp_ref = ref_logprobs(ref_path, corpus.get("n_vocab", 248320))
        order = [int(t) for t, _ in rd["logprobs"]]

        tag = f"{pid}.iters{args.iters or 'default'}"
        out = f"{args.work_dir}/{tag}.logits.txt"
        if not os.path.exists(out):
            cmd = [args.sn, "ref", "--model", args.model, "--tokens-file",
                   f"{args.tokens_dir}/{pid}.ids", "--threads", str(args.threads),
                   "--logits-out", out, "--top", "1"]
            if args.iters:
                cmd += ["--iters", str(args.iters)]
            print(f"$ {' '.join(cmd)}")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit(f"sn ref failed for {pid}: {r.stdout}{r.stderr}")
        mine = read_logits(out)
        if len(mine) != len(lp_ref):
            sys.exit(f"{pid}: {len(mine)} logits vs {len(lp_ref)} reference values")
        lp = log_softmax(mine)

        d = [a - b for a, b in zip(lp, lp_ref)]
        fin_ref = [b for b in lp_ref if math.isfinite(b)]
        nmse_full = sum(x * x for x in d) / max(1e-30, sum(b * b for b in fin_ref))
        # Logits and log-probs differ by the (unknown) log-sum-exp, which cancels in `d`. The
        # reference's absolute log-probs sit around -20, so dividing by sum(lp^2) makes that metric
        # insensitive; the plan's "logit NMSE" is meaningful only against centred logits, which is
        # what the gate uses.
        mean_ref = sum(fin_ref) / len(fin_ref)
        centred = [b - mean_ref for b in fin_ref]
        nmse_logit = sum(x * x for x in d) / max(1e-30, sum(x * x for x in centred))
        nmse_floor = None
        if args.floor_dir:
            # The reference's own spread between two of its implementations (GPU vs CPU) is the
            # honest noise floor: no other implementation can be closer to one capture than the
            # other capture is. centred_nmse() reuses the same metric on the two captures.
            other = os.path.join(args.floor_dir, f"{pid}.json.gz")
            if not os.path.exists(other):
                sys.exit(f"{other} missing")
            rd2, lp2 = ref_logprobs(other, corpus.get("n_vocab", 248320))
            nmse_floor = centred_nmse(lp2, lp_ref)
        rms_d = math.sqrt(sum(x * x for x in d) / len(d))
        top = order[: args.topk]
        dtop = [d[t] for t in top]
        nmse_top = sum(x * x for x in dtop) / max(1e-30, sum(lp_ref[t] ** 2 for t in top))
        max_top = max(abs(x) for x in dtop)
        kl = sum(math.exp(lp_ref[t]) * (lp_ref[t] - lp[t]) for t in top)
        mine_top = max(range(len(mine)), key=lambda i: mine[i])
        agree = mine_top == rd["top1"]

        row = {
            "id": pid, "tokens": len(ids), "iters": args.iters,
            "ref_top1": rd["top1"], "ref_top1_text": rd.get("top1_text"),
            "sn_top1": mine_top, "top1_match": agree,
            "nmse_full": nmse_full, "nmse_logit": nmse_logit, "rms_abs_dlogit": rms_d,
            "nmse_floor": nmse_floor,
            f"nmse_top{args.topk}": nmse_top,
            f"max_abs_top{args.topk}": max_top, "kl_top": kl,
            "rank_of_sn_top1_in_ref": rank_of(order, mine_top),
        }
        if args.compare_iters:
            # the default (checkpoint) run is cached as "itersdefault", so 20 must map back to it
            cmp_tag = "default" if args.compare_iters in (0, 20) else str(args.compare_iters)
            base = f"{args.work_dir}/{pid}.iters{cmp_tag}.logits.txt"
            if os.path.exists(base):
                b = read_logits(base)
                dd = [abs(x - y) for x, y in zip(mine, b)]
                row["diff_vs_iters"] = {
                    "iters": args.compare_iters,
                    "max_abs": max(dd),
                    "mean_abs": sum(dd) / len(dd),
                    "top1_changed": mine_top != max(range(len(b)), key=lambda i: b[i]),
                }
        rows.append(row)
        if not agree:
            failures.append(f"{pid}: top-1 {mine_top} != reference {rd['top1']}")
        if args.iters in (0, 20):
            budget = args.nmse
            if nmse_floor is not None:
                budget = max(budget, args.margin * nmse_floor)
            if nmse_logit > budget:
                note = f" (floor {nmse_floor:.2e} x {args.margin:g})" if nmse_floor is not None else ""
                failures.append(f"{pid}: centred logit NMSE {nmse_logit:.3e} > {budget:.3e}{note}")

    print()
    print(f"{'prompt':<18} {'tok':>4} {'top1 match':>11} {'NMSE(logit)':>12} {'floor':>10} "
          f"{'NMSE(lp)':>11} {'rms|dp|':>9} {'max|dp|top':>11} {'rank':>5}")
    for r in rows:
        key = f"nmse_top{args.topk}"
        floor = r["nmse_floor"]
        print(f"{r['id']:<18} {r['tokens']:>4} {str(r['top1_match']):>11} {r['nmse_logit']:>12.3e} "
              f"{(f'{floor:.2e}' if floor is not None else '-'):>10} "
              f"{r['nmse_full']:>11.3e} {r['rms_abs_dlogit']:>9.4f} "
              f"{r[f'max_abs_top{args.topk}']:>11.4f} {r['rank_of_sn_top1_in_ref']:>5}")
        print(f"{'':<18} reference top1 {r['ref_top1']} {r['ref_top1_text']!r}  "
              f"sn top1 {r['sn_top1']}")
        if "diff_vs_iters" in r:
            dv = r["diff_vs_iters"]
            print(f"{'':<18} vs iters={dv['iters']}: max|dlogit| {dv['max_abs']:.4f} "
                  f"mean|dlogit| {dv['mean_abs']:.6f} top1 changed: {dv['top1_changed']}")
    print()
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"nmse_gate": args.nmse, "margin": args.margin,
                   "floor_dir": args.floor_dir, "rows": rows}, f, indent=1)
        print(f"wrote {args.json}")
    if failures:
        print("FAIL")
        for f in failures:
            print("  " + f)
        return 1
    basis = f"logit NMSE {args.nmse:.0e}"
    if args.floor_dir:
        basis = f"logit NMSE within {args.margin:g}x of the reference's own backend spread"
    print(f"PASS: {len(rows)} prompt(s) match the reference top-1 within {basis}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
