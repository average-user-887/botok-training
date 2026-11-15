# Deriver task (Honcho "minimal deriver")

Input: a window of chat messages (humans and/or AI agents) and a target peer id. Output: strict JSON listing EXPLICIT atomic facts about the target peer, derived only from what the target said (others are context).

## Exact SFT format (one JSON per line, see examples.jsonl)
{"id": "...", "messages": [{"role":"user","content": <minimal_deriver_prompt(peer_id, messages)>}, {"role":"assistant","content":"{\"explicit\": [{\"content\": \"<peer> ...\"}, ...]}"}]}
- Prompt = code/deriver_prompt.py `minimal_deriver_prompt` verbatim (static ~450 tokens). Messages block = lines `YYYY-MM-DD HH:MM:SS peer: text`, joined by newline, inside <messages></messages>. No system message.
- Assistant content = `json.dumps({"explicit":[{"content": str}, ...]}, ensure_ascii=False)`; empty window -> `{"explicit": []}`.
- Dot may instead emit the raw window form (examples_raw.jsonl: id, peer, messages[{ts,peer,content}], gold[str]); code/build_sft_v2.py converts raw -> SFT. Raw is easier to validate.

## Rules for gold observations
- Subject = exact target peer id (lowercase id like `oren`, `mira_k`, `dev-bot`, `claude-agent`); if the target talks about someone else, make the subject clear.
- Atomic, self-contained, each fact once (no paraphrase duplicates). Absolute dates (convert "tomorrow" using message ts). Hedges ("I think") are extracted as hedged or skipped.
- Directly supported by the target's own messages; no invented facts. ~30% of windows target an AI/bot peer (its stated actions, findings, commitments, tool results it reports).
- Realistic traffic: one-liners to multi-paragraph, pasted logs/code/config/tables, typos, occasional Polish phrases, corrections, plans vs completed.

## Distributions of the real training set (v2_train, 1115 rows)
- Total tokens per row (prompt+output): p10 600, median 1618, p90 5809, max 7520; mean 2415. Output JSON tokens: median 330, p90 1259, max 2683.
- Observations per row: median 12, p90 48, max 103. Empty-gold rows: 14.8% (165).
- Raw synthetic windows (910): short 350 / medium 270 / long 210 / empty 80. Message tokens (chars/4): short 245-996 (median ~700), medium 807-2736 (median ~1800), long 2192-4796 (median ~3600).
- Gold density: roughly one observation per 60-100 tokens of the target's fact-bearing text; long windows 23-103 observations.

## Validator (examples/deriver/validate.py, run `python3 -I validate.py file.jsonl`, prints OK/FIX; raw format)
Window id must be `<tag>-<class>-<n>` (class short|medium|long|empty). Checks: JSON keys id/peer/messages/gold; ts format and strictly increasing; target must speak if gold non-empty; class limits on message tokens (chars/4) short 500-1000, medium 1500-3000, long 3400-5500 (max x1.25), messages short 3-20 / medium 8-36 / long 30-70 (+6), gold short 3-40 / medium 8-55 / long 30-110 (>=0.6x lower bound, <=1.2x upper); every gold names the peer; no duplicate gold; at most max(2, 15%) gold weakly grounded (word overlap < 35% with messages).
Note: ~7% of the older synthetic windows used for v1/v2 do not pass this (stricter) validator; the shipped examples all pass.
Target lengths for new data: short ~3.2k chars, medium ~9k chars, long ~17k chars of message text.
