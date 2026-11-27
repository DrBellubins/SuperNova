#!/usr/bin/env python3
"""Phase 1 gate: `sn dump --plan --json` must agree with Tools/roofline.py byte-for-byte.

Both tools independently compute, for every tensor in the checkpoint: exact stored bytes (from the
element type), how many times it is streamed per decoded token, and the resulting bytes/token. They
agree only if the execution plan (loop expansion, first-visit GGUF block numbering, MTP exclusion)
is understood identically. This script diffs them and fails on any disagreement.

Usage: plan-check.py <model.gguf> [--sn Engine/target/debug/sn]
"""
import json
import subprocess
import sys


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"command failed ({' '.join(cmd)}):\n{r.stderr}")
    return r.stdout


def main():
    model = sys.argv[1]
    sn = "Engine/target/debug/sn"
    if "--sn" in sys.argv:
        sn = sys.argv[sys.argv.index("--sn") + 1]

    ours = json.loads(run([sn, "dump", "--plan", "--json", model]))
    ref = json.loads(run(["python3", "Tools/roofline.py", model, "--json"]))

    mine = {t["name"]: t for t in ours["tensors"]}
    theirs = {t["name"]: t for t in ref["tensors"]}
    errs = []

    if set(mine) != set(theirs):
        errs.append(f"tensor sets differ: only-in-sn={sorted(set(mine)-set(theirs))[:5]} "
                    f"only-in-roofline={sorted(set(theirs)-set(mine))[:5]}")

    per_token = 0
    for name in sorted(set(mine) & set(theirs)):
        a, b = mine[name], theirs[name]
        for field in ("bytes", "visits", "bytes_per_token", "class", "type"):
            if a[field] != b[field]:
                errs.append(f"{name}: {field} {a[field]} != roofline {b[field]}")
        per_token += a["bytes_per_token"]

    for label, got, want in [
        ("tensor count", ours["weights"]["tensors"], len(theirs)),
        ("stored bytes", ours["weights"]["stored_bytes"], ref["stored_bytes"]),
        ("bytes per token", ours["weights"]["bytes_per_token"], ref["bytes_per_token"]),
        ("n_layer", ours["layers"]["n_layer"], ref["n_layer"]),
        ("n_physical (trunk)", ours["layers"]["n_physical_trunk"], ref["n_physical"]),
        ("extra visits", ours["layers"]["extra_visits"], ref["extra_visits"]),
        ("loop range", ours["layers"]["loop"], ref["loop"]),
        ("mHC executions", ours["executions"]["mhc"], ref["mhc_executions"]),
    ]:
        if got != want:
            errs.append(f"{label}: {got} != roofline {want}")

    print(f"tensors compared : {len(set(mine) & set(theirs))}")
    print(f"stored           : {ref['stored_bytes']:,} bytes")
    print(f"bytes per token  : {ref['bytes_per_token']:,} bytes  (sum over per-tensor: {per_token:,})")
    print(f"loop             : {ref['loop']} x  extra visits {ref['extra_visits']}  "
          f"mHC executions {ref['mhc_executions']}")
    if errs:
        print(f"\nMISMATCHES ({len(errs)}):")
        for e in errs[:40]:
            print("  " + e)
        return 1
    print("\nOK: sn and roofline.py agree byte-for-byte on every tensor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
