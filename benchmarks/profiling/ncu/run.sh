#!/usr/bin/env bash
set -euo pipefail

OPERATION="matmul"
KERNEL="naive_matmul"
M=512
K=2048
N=2048
DTYPE="float32"
DRY_RUN=false
KERNEL_REGEX=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPORT_DIR="${PROJECT_ROOT}/benchmarks/reports/ncu"
TARGET_SCRIPT="${SCRIPT_DIR}/target.py"
PARSER_SCRIPT="${SCRIPT_DIR}/parse_csv.py"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --operation) OPERATION="$2"; shift 2 ;;
        --kernel) KERNEL="$2"; shift 2 ;;
        --M) M="$2"; shift 2 ;;
        --K) K="$2"; shift 2 ;;
        --N) N="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --kernel-regex) KERNEL_REGEX="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${KERNEL_REGEX}" ]]; then
    KERNEL_REGEX="$(
        PROJECT_ROOT="${PROJECT_ROOT}" python3 - "$OPERATION" "$KERNEL" <<'PY'
import os
import sys
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"])
src_root = project_root / "src"
for path in (project_root, src_root):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from benchmarks.profiling.common import profiler_kernel_regex
from benchmarks.harness.baselines import resolve_kernel_name

operation, kernel = sys.argv[1], sys.argv[2]
try:
    canonical = resolve_kernel_name(operation, kernel)
except Exception:
    canonical = kernel
regex = profiler_kernel_regex(operation, canonical)
print(regex or "")
PY
    )"
fi

if [[ -z "${KERNEL_REGEX}" ]]; then
    echo "ERROR: No default regex is known for operation='${OPERATION}' kernel='${KERNEL}'." >&2
    exit 1
fi

NCU_BIN=""
for candidate in ncu /usr/local/cuda/bin/ncu; do
    if command -v "$candidate" &>/dev/null; then NCU_BIN="$candidate"; break; fi
done
if [[ -z "$NCU_BIN" ]]; then
    echo "ERROR: ncu not found. Add /usr/local/cuda/bin to PATH."
    exit 1
fi

mkdir -p "$REPORT_DIR"
STEM="${OPERATION}_${KERNEL}_M${M}_K${K}_N${N}_${DTYPE}"
REP_FILE="${REPORT_DIR}/${STEM}.ncu-rep"
CSV_FILE="${REPORT_DIR}/${STEM}.csv"
SUMMARY_FILE="${REPORT_DIR}/summary_${STEM}_${TIMESTAMP}.txt"
EXTRACTED_CSV_FILE="${REPORT_DIR}/parsed_${STEM}_${TIMESTAMP}.csv"

NCU_CMD=(
    "$NCU_BIN"
    --target-processes all
    --kernel-name "regex:${KERNEL_REGEX}"
    --launch-count 1
    --replay-mode kernel
    --set full
    --export "${REP_FILE}"
    --force-overwrite
    python3 "${TARGET_SCRIPT}"
        --operation "${OPERATION}"
        --kernel "${KERNEL}"
        --M "${M}" --K "${K}" --N "${N}"
        --dtype "${DTYPE}"
)

NCU_CSV_CMD=(
    "$NCU_BIN"
    --import "${REP_FILE}"
    --csv
    --page raw
)

echo "CUDA GEMM Benchmark - ncu Profile"
echo "kernel: ${KERNEL} (regex: ${KERNEL_REGEX})"
echo "shape : M=${M} K=${K} N=${N}"
echo "dtype : ${DTYPE}"

if $DRY_RUN; then
    printf '  %s \\\n' "${NCU_CMD[@]}"
    exit 0
fi

"${NCU_CMD[@]}"
"${NCU_CSV_CMD[@]}" > "${CSV_FILE}"
python3 "${PARSER_SCRIPT}" "${CSV_FILE}" > "${EXTRACTED_CSV_FILE}"
cp "${EXTRACTED_CSV_FILE}" "${SUMMARY_FILE}"
