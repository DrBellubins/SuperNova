#!/usr/bin/env python3
"""Capture the reference model's per-position hidden states.

`llama-server --embedding --pooling none --embd-normalize -1` returns the model's own hidden state
for every position of the prompt (that is the trunk output after `output_norm`, before the output
head). Those vectors are the stage-level oracle for `sn ref`: comparing them against our `h_final.<t>`
dump says whether the trunk or the output head is at fault, which the top-1 token alone cannot tell.

Writes Docs/golden/embd/<id>.json.gz:
    {"id", "tokens", "n_positions", "n_embd", "embd": [[...], ...]}

Usage: Tools/golden-embd.py [--prompts p1-capital,...] [--server <llama-server>] [--port 8124]
"""
import argparse, gzip, json, os, signal, subprocess, sys, time, urllib.error, urllib.request

DEF_MODEL = "/home/justin/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf"
DEF_SERVER = "/home/justin/Documents/Github/llama.cpp-opensensenova/build/bin/llama-server"


def post(url, body, timeout=600):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="Docs/golden/corpus.json")
    ap.add_argument("--out-dir", default="Docs/golden/embd")
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--server", default=DEF_SERVER)
    ap.add_argument("--port", type=int, default=8124)
    ap.add_argument("--ngl", type=int, default=99, help="0 runs the reference on the CPU")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--prompts", default="")
    ap.add_argument("--ctx", type=int, default=4096)
    args = ap.parse_args()

    try:
        # -x, not -f: `pgrep -f llama-server` also matches the shell that launched this
        # script whenever its command line happens to mention llama-server.
        subprocess.run(["pgrep", "-x", "llama-server"], check=True, capture_output=True)
        sys.exit("a llama-server is already running - stop it first")
    except subprocess.CalledProcessError:
        pass

    os.makedirs(args.out_dir, exist_ok=True)
    corpus = json.load(open(args.corpus))
    want = {p.strip() for p in args.prompts.split(",") if p.strip()}
    prompts = [p for p in corpus["prompts"] if not want or p["id"] in want]
    base = f"http://127.0.0.1:{args.port}"
    log = open(f"{args.out_dir}/server.log", "w")
    srv = subprocess.Popen(
        # f32 KV for the same reason as Tools/golden-ref.py: this is a numerics oracle.
        [args.server, "-m", args.model, "-c", str(args.ctx), "-ngl", str(args.ngl), "-ctk", "f32",
         "-ctv", "f32", "--embeddings",
         "--pooling", "none", "--embd-normalize", "-1", "--port", str(args.port)]
        + (["-t", str(args.threads)] if args.threads else []),
        stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                if srv.poll() is not None:
                    sys.exit(f"server exited with {srv.returncode} - see {args.out_dir}/server.log")
                time.sleep(1.0)
        else:
            sys.exit("server never became healthy - see " + f"{args.out_dir}/server.log")

        for pr in prompts:
            pid = pr["id"]
            out = f"{args.out_dir}/{pid}.json.gz"
            try:
                data = post(f"{base}/embedding", {"content": pr["text"], "raw": True,
                                                 "add_special": False})
            except urllib.error.HTTPError as e:
                print(f"{pid}: HTTP {e.code} {e.read()[:400]!r}")
                continue
            embs = None
            # this build answers /embedding with [{"index": 0, "embedding": [[...per position...]]}]
            if isinstance(data, list) and data and isinstance(data[0], dict) and "embedding" in data[0]:
                embs = data[0]["embedding"]
            elif isinstance(data, list):
                embs = data
            elif isinstance(data, dict):
                if isinstance(data.get("embedding"), list):
                    embs = data["embedding"]
                elif isinstance(data.get("data"), list):  # OpenAI-shaped
                    embs = [d.get("embedding") for d in data["data"]]
            if not embs:
                keys = list(data)[:12] if isinstance(data, dict) else f"list of {len(data)}"
                print(f"{pid}: no embeddings in response: {keys}")
                continue
            if embs and not isinstance(embs[0], list):
                embs = [embs]
            ids = [int(x) for x in open(f"Docs/golden/tokens/{pid}.ids").read().replace(",", " ").split()]
            rec = {"id": pid, "tokens": ids, "n_positions": len(embs),
                   "n_embd": len(embs[0]) if embs else 0, "embd": embs,
                   "pooling": "none", "embd_normalize": -1}
            with gzip.open(out, "w") as f:
                f.write(json.dumps(rec).encode())
            print(f"{pid}: {len(embs)} x {rec['n_embd']} -> {out} (prompt has {len(ids)} tokens)")
    finally:
        if srv is not None:
            srv.send_signal(signal.SIGINT)
            try:
                srv.wait(timeout=30)
            except subprocess.TimeoutExpired:
                srv.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
