# PROPOSED scribe candidate v1

This is a candidate training contract designed for discussion, NOT a verified deployed SI-lab interface and NOT drop-in compatible with botok-training. Public repository inspection found no scribe runtime contract. Exactly 12 synthetic examples: 10 training and 2 development, no hidden evaluation. Prior deriver heldout exposure is disclosed; these are not blind test cases.

## Role and design rationale
Produce a concise continuity handoff from a bounded multi-speaker project chat as of its last supplied message. Select useful current state, decisions, pending work and material uncertainty. Unlike the deriver role, do not extract only one peer's personal facts or attempt an exhaustive fact inventory. Reported receipts remain reported evidence, not independent tool execution. Retain enough history to explain a current blocker; avoid stale intentions as completed actions.

Items are categorized rather than forced into a fixed count. Each item cites source message IDs. Owner and due date are nullable: do not infer that the speaker owns a task or invent deadlines. A handoff can combine closely related clauses for concision; citations must support the entire item. Preserve prohibition/approval scope, unresolved causes, and distinctions between mockups, fixtures, live systems and publication.

The input is an `as_of` string plus chronological `messages` with stable `id`, `ts`, `speaker`, and `text`. Output has `title`, matching `as_of`, and `items`. Each item has `kind` (`state`, `decision`, `pending`, `uncertainty`), `text`, nullable `owner`/`due`, and an array of `evidence` message IDs. `items: []` is allowed when no useful supported handoff content exists. No abstention samples are included in this bounded positive pilot; robustness to adversarial source instructions and empty conversations remains unevaluated.

## Limits and split policy
All conversations are authored synthetic local engineering/teaching scenarios. No customer records, credentials, medical material, GPU execution or V11 pilot modification. Sources were written and snapshotted before labels. `source/*.jsonl` and authoring scripts are provenance only; train only from data/train.jsonl. Dev uses distinct stories but the same author/schema and is for development, not blind generalization claims. Do not merge this role's examples or schema into deriver/pruner/judge without explicit integration design.

The validator checks structural consistency, exact source identity and evidence references, not entailment or model quality. Independent review is required before describing annotation QA as accepted. Runtime parser integration, downstream consumer expectations and deployment remain unverified. This small pilot is not a 12-hour training budget.

## Annotation-review limitations
This bounded pilot has only positive outputs, all due fields null and short 6–7-message conversations. Several share a correction/status/permission cadence despite different project narratives. This does not demonstrate coverage of long conversations, missing evidence, adversarial instructions, competing owners or explicit deadline extraction. Dates relevant to urgency may appear in item text; the March 5 demonstration is not invented as a deadline for every pending task. A larger role-specific benchmark must address those gaps before readiness claims.
