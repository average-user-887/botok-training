# Judge role (illustrative, planned; no model trained yet)
Scores a candidate extraction against a gold list for the same window, strict JSON only.
- Input: target peer, GOLD list, CANDIDATE list (the output of a deriver model).
- Output: {"matched":[gold idx],"missed":[gold idx],"hallucinated":[candidate idx],"verdict":"pass"|"fail"}; optional short "note".
- Rules: a candidate matches if it states the same fact with the peer id as subject (paraphrase ok); duplicates of a matched fact are not extra matches; candidate facts unsupported by gold/source are hallucinated; pass = nothing missed and nothing hallucinated.
- examples.jsonl: 6 INVENTED ILLUSTRATIVE rows, not from any real data and not from a trained system. The goal for generated data: varied gold/candidate pairs (perfect, partial, extra facts, paraphrases, empty gold, duplicates, wrong subject).
