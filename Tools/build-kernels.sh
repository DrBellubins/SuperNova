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
OBJCOPY=${OBJCOPY:-}
if [ -z "$OBJCOPY" ]; then
  for c in "$ROCM/lib/llvm/bin/llvm-objcopy" "$ROCM/bin/llvm-objcopy" "$(command -v llvm-objcopy || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then OBJCOPY="$c"; break; fi
  done
fi
SRC=crates/sn-hip/kernels
OUT=${OUT:-target/hip}

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
  ls -l "$OUT/$base.fatbin"
done
cat > "$OUT/manifest.txt" <<EOF
built $(date -Is)
hipcc: $($HIPCC --version | head -1)
isas: ${ISAS[*]}
sources: $(ls "$SRC"/*.hip | tr '\n' ' ')
EOF
echo "kernel images in $OUT (SN_KERNELS=<path> overrides; default target/hip/phase3.fatbin)"
