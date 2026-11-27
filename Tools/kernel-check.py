#!/usr/bin/env python3
"""Phase 4 gate: replay the mHC routing kernels against Phase 2 stage dumps on the GPU.

This is the wiring around `sn gpu route`, which is where the routing kernels meet real numbers:
`sn ref` decodes a frozen corpus prompt with the reference implementation and dumps every mHC stage
(streams, router logits, post-scales, mixes, the merged streams), then the kernels recompute those
stages from the *checkpoint weights and the dumped pre-merge streams* and the two are compared.

Gate (Docs/design-plan.md section 7, Phase 4):
  * NMSE of every stage <= 1e-5, and <= 1e-6 for the mix (the mix is raised to a power of the
    streams by the merge, so its error is the one that compounds downstream);
  * reported: the worst NMSE per class, and the count of failing stages.

The dump step is the slow one (it runs the reference decoder, and the dumps are text). It is
reused when the directory already has stages, so repeat runs are fast; --refetch forces it.

Usage:
  Tools/kernel-check.py                             # dump if needed, then check execs 4..11
  Tools/kernel-check.py --prompt p3-code --limit-exec 16
  Tools/kernel-check.py --execs 4-11 --dump /tmp/stage
  Tools/kernel-check.py --refetch                   # re-run `sn ref` even if the dump exists
"""

import argparse
import os
import subprocess
import sys
import time

DEF_MODEL = "/home/justin/Documents/AI/Models/SenseLoop/SenseNova-Loop-9B-Q4_K_M.gguf"
DUMP_ONLY = "mhc.,l.,l_out.,gdn.,ffn.,attn."


def run(cmd, quiet=False):
    print("+ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603 - fixed argv, no shell
    if p.stdout and not quiet:
        sys.stdout.write(p.stdout)
    if p.returncode != 0:
        sys.stderr.write(p.stderr or "")
        sys.exit(f"command failed ({p.returncode}): {' '.join(cmd)}")
    return p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--model", default=DEF_MODEL)
    ap.add_argument("--sn", default="target/release/sn")
    ap.add_argument("--prompt", default="p1-capital", help="id from Docs/golden/tokens/<id>.ids")
    ap.add_argument("--tokens", help="token id file (default Docs/golden/tokens/<prompt>.ids)")
    ap.add_argument("--dump", default="/tmp/sn-phase4-stages", help="stage dump directory")
    ap.add_argument("--limit-exec", type=int, default=12, help="executions to decode when dumping")
    ap.add_argument("--execs", default="", help="execution range to check (default: the mHC ones in the dump)")
    ap.add_argument("--refetch", action="store_true", help="re-run the reference decoder even if --dump exists")
    args = ap.parse_args()

    if not os.path.exists(args.sn):
        sys.exit(f"{args.sn} is not built: cargo build --release")

    # The kernel image must exist too; `sn gpu route` refuses otherwise, but saying it here points
    # at the fix instead of at a HIP error.
    images = [f for f in os.listdir("target/hip") if f.endswith(".fatbin")] if os.path.isdir("target/hip") else []
    if not images:
        sys.exit("no kernel images in target/hip: run ./Tools/build-kernels.sh (and keep its output visible)")

    tokens = args.tokens or f"Docs/golden/tokens/{args.prompt}.ids"
    if not os.path.exists(tokens):
        sys.exit(f"{tokens} is not in the frozen corpus")

    have_stages = os.path.isdir(args.dump) and any(f.startswith("mhc.") and f.endswith(".logits.txt") for f in os.listdir(args.dump))
    if args.refetch or not have_stages:
        os.makedirs(args.dump, exist_ok=True)
        t0 = time.time()
        run(
            [
                args.sn,
                "ref",
                "--model",
                args.model,
                "--tokens-file",
                tokens,
                "--dump",
                args.dump,
                "--dump-only",
                DUMP_ONLY,
                "--limit-exec",
                str(args.limit_exec),
            ],
            quiet=True,
        )
        print(f"stage dumps written to {args.dump} in {time.time() - t0:.0f} s")
    else:
        print(f"reusing stage dumps in {args.dump} (use --refetch to regenerate)")

    # Default: every mHC execution the dump actually covers. Executions before mhc_start have no
    # router, so listing them would make `sn gpu route` complain about missing stages.
    execs = args.execs
    if not execs:
        found = sorted(
            int(f.split(".")[1])
            for f in os.listdir(args.dump)
            if f.startswith("mhc.") and f.endswith(".logits.txt")
        )
        if not found:
            sys.exit(f"{args.dump} contains no mhc.<il>.logits.txt - regenerate with --refetch")
        execs = f"{found[0]}-{found[-1]}"

    out = run([args.sn, "gpu", "route", "--model", args.model, "--dump", args.dump, "--execs", execs], quiet=True)
    print(out, end="" if out.endswith("\n") else "\n")

    verdict = [l for l in out.splitlines() if l.startswith("GATE:")]
    worst = [l for l in out.splitlines() if l.startswith("worst NMSE")]
    if worst:
        print(worst[0])
    if not verdict:
        sys.exit("sn gpu route printed no verdict")
    print(verdict[0])
    return 0 if verdict[-1].strip().endswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
