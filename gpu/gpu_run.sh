#!/usr/bin/env bash
# Upload the solver to a CUDA box, build it, check it, and time a few configurations.
#   gpu/gpu_run.sh "ssh -p PORT root@HOST"      (same string as the ssh command)
set -euo pipefail
SSH=${1:?usage: $0 "ssh -p PORT root@HOST"}
PORT=$(echo "$SSH" | sed -n 's/.*-p *\([0-9]*\).*/\1/p')
HOST=$(echo "$SSH" | awk '{print $NF}')
DIR=/workspace/dd
cd "$(dirname "$0")/.."

ssh -p "$PORT" "$HOST" "mkdir -p $DIR"
scp -q -P "$PORT" src/dd.h src/dd_bounds.h src/dd_wave.h src/npy.h gpu/gpu_main.cu data/deals_20k.npy "$HOST:$DIR/"
ssh -p "$PORT" "$HOST" bash -s <<EOF
set -e
cd $DIR
export PATH=\$PATH:/usr/local/cuda/bin
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ARCH=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
nvcc -O3 -std=c++17 -arch=sm_\$ARCH -Xptxas -v -o dd_gpu gpu_main.cu 2>&1 | grep -E "_kernel|stack frame" | head -4

echo "== endings (checked against the CPU on the same box)"
WAVE=1 STACK_KB=8 ./dd_gpu 8 20000 10 300
echo "== full deals, 120 s (checked against the Pgx tables)"
WAVE=1 STACK_KB=8 TIME_LIMIT=120 ./dd_gpu deals_20k.npy 0 20000 | grep -v "jobs done"
EOF
