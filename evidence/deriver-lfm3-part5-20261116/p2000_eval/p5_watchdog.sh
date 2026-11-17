#!/bin/bash
# independent restoration supervisor: after EVAL_DONE or deadline, stop OUR server, restore d1, verify
D=$HOME/p5-eval-20261116; DEADLINE=$(( $(date +%s) + ${1:-14400} ))
while [ ! -f $D/EVAL_DONE ] && [ $(date +%s) -lt $DEADLINE ] && ! [ -f $D/WATCHDOG_STOP ]; do sleep 20; done
kill $(cat $D/server.pid) 2>/dev/null; sleep 5; kill -9 $(cat $D/server.pid) 2>/dev/null
for t in 1 2 3 4 5; do
 curl -s -m 120 http://192.168.1.23:18000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"decision-d1-2b","messages":[{"role":"user","content":"ok"}],"max_tokens":1}' >/dev/null
 curl -s -m 10 http://192.168.1.23:18000/running | grep -q '"decision-d1-2b".*"state":"ready"\|"model":"decision-d1-2b","state":"ready"' && { echo "RESTORED $(date -Is)" > $D/RESTORE_STATUS; nvidia-smi --query-gpu=memory.used --format=csv >> $D/RESTORE_STATUS; exit 0; }
 sleep 10
done
echo "RESTORE_FAILED $(date -Is)" > $D/RESTORE_STATUS
