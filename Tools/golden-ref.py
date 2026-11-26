#!/usr/bin/env python3
"""Capture the Phase 2 reference: token ids + full-vocabulary log-probs from llama.cpp.

The plan's Phase 2 gate compares `sn ref` against the fork, so this is the only place the reference
is allowed to be used as an oracle. One server start serves every prompt in the corpus:

  * Tools/tok (links libllama)          -> token ids; the server's /tokenize answers 501 in this
    fork's build, and the GGUF BPE tokenizer is Phase 10 work for us (Docs/design-plan.md §7)
  * POST /completion n_probs=<vocab>     -> the distribution after the last prompt token
    (tools/server/server-common.cpp:1814 get_token_probabilities softmaxes llama_get_logits_ith,
     so log p = logit - max - log sum: gaps between log p values are exactly logit gaps.)

Output, per prompt, under --out:
  tokens/<id>.ids        comma-separated token ids (what `sn ref --tokens-file` reads)
  ref/<id>.json.gz       {"tokens": [...], "top1": id, "n_vocab": N,
                          "logprobs": [[token, log p], ...] sorted by log p descending}

Usage: Tools/golden-ref.py [--corpus Docs/golden/corpus.json] [--out Docs/golden]
                           [--model M] [--limit 0] [--port 8123] [--ngl 99]
Keep the corpus frozen once captured: every later phase is measured against these files.
"""
import argparse, gzip, json, os, signal, subprocess, sys, time, urllib.request

DEF_MODEL = "/home/justin/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf"
HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(HERE, "tok")
DEF_SERVER = "/home/justin/Documents/Github/llama.cpp-opensensenova/build/bin/llama-server"
N_VOCAB = 248320  # tokenizer.ggml.n_vocab; overridden from the /props response when reachable


def wait_ready(port, deadline):
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def post(port, path, obj, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tok_ids(texts, out_dir, model):
    """Tokenise with the checkpoint's own tokenizer through Tools/tok (the server's /tokenize
    answers 501 in this fork's build). One call per batch: Tools/tok loads the model once."""
    files = []
    for i, t in enumerate(texts):
        path = f"{out_dir}/prompt-{i}.txt"
        with open(path, "w") as f:
            f.write(t)
        files.append(path)
    out = subprocess.run([TOK, "--ids", model] + files, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"Tools/tok failed: {out.stderr[-400:]}")
    lines = [l for l in out.stdout.splitlines() if l.strip() != ""]
    if len(lines) != len(texts):
        raise SystemExit(f"Tools/tok returned {len(lines)} lines for {len(texts)} prompts")
    return [[int(x) for x in l.split(",")] for l in lines]


def exp0(probs):
    import math
    return f"{math.exp(probs[0][1]):.4f}" if probs else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="Docs/golden/corpus.json")
    ap.add_argument("--out", default="Docs/golden")
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--server", default=DEF_SERVER)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="keep only the top N log-probs (0 = full vocab)")
    args = ap.parse_args()

    if subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).stdout:
        sys.exit("a llama-server is already running - stop it first (it would steal the port)")

    corpus = json.load(open(args.corpus))
    os.makedirs(f"{args.out}/tokens", exist_ok=True)
    os.makedirs(f"{args.out}/ref", exist_ok=True)

    log = open(f"{args.out}/server.log", "w")
    # -ctk/-ctv f32: the default f16 KV cache rounds the reference's own K/V, which adds ~1e-3
    # relative error per attention layer. The golden corpus is a numerics reference, so ask for the
    # cleanest reference the binary can produce (Phase 7 compares against this, not against f16 KV).
    cmd = [args.server, "-m", args.model, "-c", str(args.ctx), "-ngl", str(args.ngl),
           "-ctk", "f32", "-ctv", "f32",
           "--host", "127.0.0.1", "--port", str(args.port), "-t", str(os.cpu_count() or 8)]
    print("+ " + " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ok = True
    try:
        if not wait_ready(args.port, time.time() + 240):
            sys.exit(f"server never became healthy - see {args.out}/server.log")
        try:
            n_vocab = post(args.port, "/props", {}).get("total_n_tokens", N_VOCAB) or N_VOCAB
        except Exception:
            n_vocab = N_VOCAB
        # one tokenizer process for the whole corpus (it loads the model once)
        batches = tok_ids([pr["text"] for pr in corpus["prompts"]], args.out, args.model)
        all_ids = {pr["id"]: b for pr, b in zip(corpus["prompts"], batches)}
        for k, v in all_ids.items():
            print(f"  tokenised {k}: {len(v)} ids")
        for pr in corpus["prompts"]:
            ids = all_ids[pr["id"]]
            with open(f"{args.out}/tokens/{pr['id']}.ids", "w") as f:
                f.write(",".join(map(str, ids)) + "\n")
            # raw + add_special false == common_tokenize(vocab, text, false, true), which is what
            # /tokenize does, so the ids below are the ids the server processes. The prompt_n check
            # right after is what catches it if that ever stops being true.
            # The legacy /completion body takes n_probs directly (tools/server/server-task.cpp);
            # top_logprobs is only mapped to n_probs on the OpenAI-style paths
            # (tools/server/server-common.cpp:1690).
            comp = post(args.port, "/completion", {
                "prompt": pr["text"], "raw": True, "add_special": False,
                "n_predict": 1, "temperature": 0.0, "seed": 0,
                "cache_prompt": False, "n_probs": n_vocab if args.limit == 0 else args.limit,
            })
            # This fork answers the legacy shape: completion_probabilities[] entries carry
            # {id, token, logprob, top_logprobs:[{id, token, logprob}, ...]} for each generated
            # token. The first entry is the distribution at the last prompt position, which is what
            # `sn ref` returns. logprob is a log-probability, so differences between entries are
            # exactly differences of logits (the softmax constant cancels).
            cp = comp.get("completion_probabilities") or []
            if not cp:
                json.dump(comp, open(f"{args.out}/ref/{pr['id']}-raw.json", "w"), indent=1)
                print(f"  {pr['id']}: empty completion_probabilities - raw saved")
                ok = False
                continue
            first = cp[0]
            probs = [(e["id"], e["logprob"]) for e in first.get("top_logprobs", [])]
            pn = comp.get("tokens_evaluated")
            if pn is not None and pn != len(ids):
                print(f"  WARNING {pr['id']}: server tokens_evaluated={pn} != /tokenize {len(ids)} ids")
            payload = {
                "id": pr["id"], "tokens": ids, "n_prompt_tokens": len(ids),
                "top1": first["id"], "top1_text": first.get("token", ""),
                "timing": comp.get("timings", {}),
                "n_probs": len(probs),
                "logprobs": [[t, float(lp)] for t, lp in probs],
            }
            with gzip.open(f"{args.out}/ref/{pr['id']}.json.gz", "wt") as f:
                json.dump(payload, f)
            print(f"{pr['id']:>16}  {len(ids):>4} tokens  top1={first['id']} {first.get('token','')!r}  "
                  f"logprobs={len(probs)}  top_p={exp0(probs)}")
    finally:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        log.close()
    print(f"\nwrote {args.out}/tokens/*.ids and {args.out}/ref/*.json.gz (ok={ok})")


if __name__ == "__main__":
    main()
