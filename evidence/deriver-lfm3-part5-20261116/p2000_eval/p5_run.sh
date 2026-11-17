#!/bin/bash
# eval runner: assumes d1 already unloaded by operator
D=$HOME/p5-eval-20261116; R=/opt/stacks/llama-cpp/releases/p2000-a4d4f2f-sm61
BASE=/opt/stacks/llama-cpp/ssd-specialists/derivbench-models/lfm3-1.7b-base-Q4_K_M.gguf
L=""; for a in p4-pct20 p5-final p5-pct20 p5-pct40 p5-pct60 p5-pct80 p5-pct10 p5-pct30 p5-pct50 p5-pct70 p5-pct90; do
 case $a in p4-pct20) f=p4-pct20-f32.gguf;; p5-final) f=p5-final-f32.gguf;; *) f=${a}-f32.gguf;; esac; L="$L,$D/$f"; done
L=${L#,}
export CUDA_VISIBLE_DEVICES=0 GGML_CUDA_ENABLE_UNIFIED_MEMORY=0 LD_LIBRARY_PATH=$R/lib
$R/bin/llama-server -m $BASE --lora "$L" --lora-init-without-apply --host 127.0.0.1 --port 18999 -np 1 -c 8192 -ngl 99 --fit off -fa on --threads 4 --no-webui > $D/server.log 2>&1 &
echo $! > $D/server.pid
for i in $(seq 1 90); do curl -sf http://127.0.0.1:18999/health >/dev/null && break; sleep 2; done
python3 $D/p5_client.py > $D/client.log 2>&1
kill $(cat $D/server.pid)
