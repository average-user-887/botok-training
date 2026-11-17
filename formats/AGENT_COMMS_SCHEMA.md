# Inter-agent communication schema (hardline mailbox, synthetic)

Harnesses/agents: `claude`, `codex` (alias `astra`), `antigravity`, `hermes`. Lanes: `<agent>:<session_id>`; `inbox('<agent>')` reads bare + own lane, only the lane owner consumes.

## Message envelope
`{"message_id","created_at","from_agent","to_agent","body","deliver":bool,"status":"pending|delivered|acked"}`.
Limits: body 400-4000 chars (envelope extra); inbox default 25, max 200; response cap 60k chars.
Timeouts (s): hermes 180, claude 900, codex 14400.

## Tool surface
`send(from_agent,to_agent,message,deliver)`, `inbox`, `peek`, `ack`, `history`, `list_agents`, `role_send(recipient,...)` / `role_route` / `role_status`, `kanban_submit` (always lands in triage, manual release), `job_status/job_result/job_cancel/list_jobs`, `ask_hermes` (direct route is explicit-only, 1 concurrent, fail-closed).

## Body convention for task messages
First line `[<TAG> v1]` (e.g. `[Hardline fresh-codex v1]`), then `TASK:`, `CONTEXT:` (paths only, no secrets), `CONSTRAINTS:`, `REPLY:` (format, <=N lines, where to write). Replies: `RESULT:` / `BLOCKED:` / `NEEDS INPUT:` first line, then evidence paths.

## Rules
Prefer native subagents/steering over hardline to your own harness; no secrets in bodies; ack after handling; one message = one ask.
See `examples/agent_comms_synthetic.jsonl`.
