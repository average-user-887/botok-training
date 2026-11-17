# Honcho formats (synthetic)

Workspace-scoped objects. Peers are lowercase ids (`operator`, `hermes`, `mira_k`, `dev-bot`).

## Peer card
A JSON array of strings, each `CATEGORY: text`. Categories seen: `IDENTITY`, `RELATIONSHIP`, `PREFERENCE`, `PROJECT`, `CONSTRAINT`. Compact, 3-15 lines, no secrets.
```json
["IDENTITY: Name: operator","IDENTITY: Kind: Human, owns the lab","RELATIONSHIP: Collaborator: hermes","PREFERENCE: Wants short, dense replies","CONSTRAINT: Private data must not be published"]
```
Agent peer example: `["IDENTITY: Name: hermes","IDENTITY: Kind: Autonomous agent harness","RELATIONSHIP: Collaborator: operator"]`.

## Session + messages
`add_messages_to_session` body: `{"messages":[{"peer_id":"operator","content":"..."},{"peer_id":"hermes","content":"..."}]}`. One `session_id` per coherent context (thread/project). Each message later has `id`, `created_at` (ISO), `peer_id`, `content`.
Deriver input renders them as lines: `YYYY-MM-DD HH:MM:SS peer: text`.

## Conclusion (derived fact)
Fields: `id`, `content` (one atomic self-contained sentence, subject = exact peer id, absolute date), `observer_id`, `observed_id` (self-conclusion when equal), `session_id`, `created_at`.
```json
{"id":"c_0001","observer_id":"hermes","observed_id":"mira_k","session_id":"s_demo_01","created_at":"2030-03-02T09:08:55Z","content":"mira_k bought a Sigma 56mm f/1.4 lens on February 27, 2030 for 389 euro."}
```
Rules: no pronoun-only subjects, no relative dates, no duplicates/paraphrases, hedges kept as hedged, plan != completion, negation preserved.

## Deriver SFT row
Exactly as in `../examples/deriver/SCHEMA.md`:
`{"id","messages":[{"role":"user","content":<deriver prompt with <messages>block</messages> and Target peer: X>},{"role":"assistant","content":"{\"explicit\": [{\"content\": \"...\"}]}"}]}`; empty window -> `{"explicit": []}`.
Raw form: `{"id","peer","messages":[{"ts","peer","content"}],"gold":["fact", ...]}`; ts strictly increasing.

## Judge (observation matching) output
`{"matched":[gold idx],"missed":[gold idx],"hallucinated":[cand idx],"attribution_invalid":[cand idx],"pass":bool}`; see `../examples/judge/`.
