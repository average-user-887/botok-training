# Pruner task
Runs before a large model reads a tool result: keep only the lines the large model needs for the current goal.
- Input (single prompt string): `GOAL: <what the agent is doing, <=300 tokens>\n\nTOOL OUTPUT:\n1| line\n2| line ...\n\nKEEP:` (chunks <= 4000 tokens, lines numbered from 1).
- Output, nothing else: ` KEEP 12-18,40,77-81` or ` KEEP none` or ` KEEP all` (regex `^KEEP (none|all|\d+(-\d+)?(,\d+(-\d+)?)*)$`). The harness copies kept lines verbatim, so the model never writes content.
- Rules of thumb used for labels: error lines and their neighbours are always kept; boilerplate/progress noise dropped; `all` only for short outputs (<=40 lines) where >=80% is relevant; ~17% `none`, ~10% `all`, rest partial; ~70% of tokens removed on average.
- Real training data is private and NOT included. examples.jsonl holds 8 INVENTED illustrative rows (fields prompt, completion). Dot should generate similar rows (pytest, git, docker, df, grep, make, logs, lab-ops commands), 10-4000 tokens each, with varied goals, ~17% none, ~10% all.
- Train format: prompt + completion (completion starts with a space), loss on completion only.
