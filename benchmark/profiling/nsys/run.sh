#!/usr/bin/env bash
set -euo pipefail

OPERATION="matmul"
KERNEL="naive_matmul"
M=128
K=1024
N=1024
DTYPE="float32"
WARMUP_ITERS=2
ITERS=5
DRY_RUN=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/benchmarks/reports/nsys"
TARGET_SCRIPT="${SCRIPT_DIR}/target.py"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --operation) OPERATION="$2"; shift 2 ;;
        --kernel) KERNEL="$2"; shift 2 ;;
        --M) M="$2"; shift 2 ;;
        --K) K="$2"; shift 2 ;;
        --N) N="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --warmup-iters) WARMUP_ITERS="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

NSYS_BIN=""
for candidate in nsys /usr/local/cuda/bin/nsys; do
    if command -v "$candidate" &>/dev/null; then NSYS_BIN="$candidate"; break; fi
done
if [[ -z "$NSYS_BIN" ]]; then
    echo "ERROR: nsys not found. Add /usr/local/cuda/bin to PATH."
    exit 1
fi

mkdir -p "$REPORT_DIR"
STEM="${OPERATION}_${KERNEL}_M${M}_K${K}_N${N}_${DTYPE}_${TIMESTAMP}"
OUT_BASE="${REPORT_DIR}/${STEM}"

CMD=(
    "$NSYS_BIN"
    profile
    --trace=cuda,nvtx,osrt
    --sample=none
    --force-overwrite=true
    --output="${OUT_BASE}"
    python3 "${TARGET_SCRIPT}"
        --operation "${OPERATION}"
        --kernel "${KERNEL}"
        --M "${M}" --K "${K}" --N "${N}"
        --dtype "${DTYPE}"
        --warmup-iters "${WARMUP_ITERS}"
        --iters "${ITERS}"
)

echo "CUDA GEMM Benchmark - nsys Trace"
echo "kernel: ${KERNEL}"
echo "shape : M=${M} K=${K} N=${N}"
echo "dtype : ${DTYPE}"

if $DRY_RUN; then
    printf '  %s \\\n' "${CMD[@]}"
    exit 0
fi

"${CMD[@]}"
