#!/bin/sh
# Run LOL against the Emmy Hackathon open problem set on the hosted
# runner. Usage: ./check.sh [problem ...]   (default: all fifteen)
set -u
TASK_DIR="${EMMY_TASK_DIR:-$HOME/workspace/emmy-hackathon-2026}"
PY="$TASK_DIR/.venv/bin/python"
[ -x "$PY" ] || PY=python3
if [ $# -eq 0 ]; then
    set -- p00_scalar_add p01_vector_add p02_scale_shift p03_relu_mul \
           r01_array_sum r02_big_sum r03_mean r04_row_sum \
           m01_matmul_512 m02_matmul_rect \
           f01_rmsnorm f02_row_softmax f03_l2norm \
           a01_attention a02_attention_128
fi
cd "$TASK_DIR" || exit 1
pass=0; fail=0
for p in "$@"; do
    out=$("$PY" submit.py "$p" --compiler "$HOME/workspace/lol-emmy/compile.py" 2>&1)
    if printf '%s' "$out" | grep -q " PASS "; then
        printf "%-22s PASS\n" "$p"; pass=$((pass + 1))
    else
        printf "%-22s FAIL\n" "$p"; fail=$((fail + 1))
        printf '%s\n' "$out" | tail -3
    fi
done
echo "---"
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
