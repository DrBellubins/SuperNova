#!/usr/bin/env bash
# Build SuperNova's device kernel image: .hip sources -> one fat binary loaded by hipModuleLoad.
#
# The Rust build deliberately invokes no HIP toolchain (design-plan §6.1), so this is the step that
# decides which ISA the engine can run. Verified recipe on this box (ROCm 10.1 / HIP 7.16):
#
#   hipcc --offload-arch=gfx1102 -c x.hip -o x.o
#   llvm-objcopy --dump-section .hip_fatbin=x.fatbin x.o
#   -> hipModuleLoad / hipModuleGetFunction / hipModuleLaunchKernel all accept the result.
#
# usage: Tools/build-kernels.sh [isa ...]        # default: every gfx the host reports
set -euo pipefail

ROCM=${ROCM:-/opt/rocm}
HIPCC=${HIPCC:-$ROCM/bin/hipcc}
# The cargo workspace is Engine/, so the images belong in Engine/target/hip whatever directory this
# is called from (the engine finds them there by search - sn-hip/src/bench.rs::image_dir).
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OBJCOPY=${OBJCOPY:-}
if [ -z "$OBJCOPY" ]; then
  for c in "$ROCM/lib/llvm/bin/llvm-objcopy" "$ROCM/bin/llvm-objcopy" "$(command -v llvm-objcopy || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then OBJCOPY="$c"; break; fi
  done
fi
SRC=${SRC:-$ROOT/Engine/crates/sn-hip/kernels}
OUT=${OUT:-$ROOT/Engine/target/hip}

[ -x "$HIPCC" ] || { echo "no hipcc at $HIPCC (set ROCM=/path/to/rocm)"; exit 1; }
[ -x "$OBJCOPY" ] || { echo "no llvm-objcopy found (set OBJCOPY=...)"; exit 1; }
mkdir -p "$OUT"

if [ $# -gt 0 ]; then
  ISAS=("$@")
else
  # Every GPU ISA the host reports. CPU agents show up as non-gfx names and are skipped; a second
  # iGPU is harmless - the fat binary carries code for each requested arch.
  mapfile -t ISAS < <( { "$ROCM/bin/rocminfo" 2>/dev/null || true; } \
      | sed -nE 's/^ *Name: *(gfx[0-9][0-9a-z]*).*/\1/p' | sort -u)
fi
[ ${#ISAS[@]} -gt 0 ] || { echo "could not determine a GPU ISA; pass one: $0 gfx1102"; exit 1; }

ARCH_FLAGS=()
for a in "${ISAS[@]}"; do ARCH_FLAGS+=(--offload-arch="$a"); done

for src in "$SRC"/*.hip; do
  base=$(basename "$src" .hip)
  echo "== $src -> ${ISAS[*]}"
  # The source must include hip_runtime.h: without it hipcc refuses a __global__ function with
  # "Can't find declaration for hipLaunchKernel".
  "$HIPCC" "${ARCH_FLAGS[@]}" -O2 -c "$src" -o "$OUT/$base.o"
  "$OBJCOPY" --dump-section ".hip_fatbin=$OUT/$base.fatbin" "$OUT/$base.o"
  # Record the workgroup constants the image was compiled with. A stale image is invisible at run
  # time in the bad way: Rust picks the workgroup size and the compiled kernel indexes its shared
  # scratch with its own `#define`, so a mismatch silently reads outside the scratch and produces
  # wrong router logits (it did - Docs/bench.md). The loader refuses the pair instead.
  {
    echo "arch=${ISAS[*]}"
    echo "sn_block=$(sed -n 's/^#define SN_BLOCK  *\([0-9]*\).*/\1/p' "$src" | head -1)"
    echo "sn_route_threads=$(sed -n 's/^#define SN_ROUTE_THREADS  *\([0-9]*\).*/\1/p' "$src" | head -1)"
    echo "sn_gemv_block=$(sed -n 's/^#define SN_GEMV_BLOCK  *\([0-9]*\).*/\1/p' "$src" | head -1)"
    echo "source=$src"
    # The parameter count of every __global__ in the image. `ProgramSpec::build` compares it against
    # the argument list each launch carries: HIP does not check that, and an extra or missing
    # argument is a wrong *number*, not a fault (an extra `n_v_heads` before `d_v` made the GDN
    # gated norm read `d_v = 32` and produced a plausible-looking wrong forward - Docs/bench.md §4).
    # Absent sidecars are allowed: the guard degrades to no check.
    if [ -n "${PYTHON:-python3}" ] && command -v "${PYTHON:-python3}" >/dev/null 2>&1; then
      "${PYTHON:-python3}" - "$src" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
# Join a signature that wraps, then count top-level commas between the outer parentheses.
for m in re.finditer(r'__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(', src):
    name, i, depth, j = m.group(1), m.end() - 1, 0, m.end() - 1
    while True:
        c = src[j]
        depth += (c == '(') - (c == ')')
        if depth == 0:
            break
        j += 1
    args = src[i + 1:j]
    d, n = 0, 0
    for c in args:
        d += (c in '([') - (c in ')]')
        n += (c == ',' and d == 0)
    print(f"argc_{name}={0 if not args.strip() or args.strip() == 'void' else n + 1}")
PY
    fi
  } > "$OUT/$base.meta"
  ls -l "$OUT/$base.fatbin"
done
cat > "$OUT/manifest.txt" <<EOF
built $(date -Is)
hipcc: $($HIPCC --version | head -1)
isas: ${ISAS[*]}
sources: $(ls "$SRC"/*.hip | tr '\n' ' ')
EOF
echo "kernel images in $OUT (SN_KERNELS=a:f overrides; default: every Engine/target/hip/*.fatbin)"
