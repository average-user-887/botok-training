# RELIABILITY

Honest account of what failed, with raw logs. All runs single T4.

## Run history (account name redacted)
| Run | Outcome | Evidence |
|---|---|---|
| v1 `p2000-deriver-lfm3-lora` (330-row dataset) | COMPLETE: load 5394 s, 152 steps, total 6652 s | `../../logs/kaggle/p2000-deriver-lfm3-lora/` |
| v2a run 1 (1115-row dataset, older script) | CUDA OOM at ~1660 s after load 1468 s; steps 1/2/5 at 28/45/112 s; loss 2.536/2.123/1.349; 2.58 GiB alloc with 13.23 GiB in use | `raw/v2a_run1_oom.log` |
| v2a run 2 (same) | CUDA OOM at 761 s after load 576 s; steps 1/2/5 at 29/44/104 s; 2.58 GiB alloc with 12.78 GiB in use. Queued ~2 h; started 15:18 CET, OOM 15:33 CET | `raw/v2a_run2_oom.txt` |
| v2a v3 (fixes: `logits_to_keep`, `expandable_segments`, SDPA, OOM-skip), pushed 15:33 CET | never ran: status CANCEL_ACKNOWLEDGED at 15:48 while queued. Cancelled by someone other than the agent that wrote this packet; the cancelling party is not known here | no run log exists |
| probe (12-step, this packet) | kernel COMPLETE, but resume check failed with OOM | `raw/probe_kernel_log.txt` |
The v2a kernel's last recorded state is CANCEL_ACKNOWLEDGED. Earlier-version scripts for run 1/2 are not recoverable beyond the repo copy noted in IDENTITY. The full v3 recipe has therefore NEVER been run to completion on the 1115-row dataset; only the probe loop was.

## Probe findings
1. OOM-skips: 10 of 96 sampled rows (10.4%) OOM-ed in forward/backward and were skipped (rows 482, 555, 1077, 64, 323, 362, 938, 704, 366, 415). All have total length 6500-6946 tokens. Peak torch allocation 14307 MiB of 14.56 GiB. Rows of roughly >=6500 tokens effectively do not train on one T4 with this recipe. 48 of the 1061 kept train rows (4.5%) are >=6500 tokens (the exact failure threshold was not located; some rows near 6500 may succeed).
2. GradScaler: grad_norm was NaN/inf for steps 1-4, so the scaler skipped those 4 optimizer updates (fp16 scale calibration). Only 8 of 12 steps updated the adapter. The "lr_scheduler.step() before optimizer.step()" warning has the same cause. Real grad norms from step 5: 8.65, 6.97, 3.40, 1.00, 1.34, 1.76, 0.83, 0.72.
3. Second GPU idle (HARDWARE.md).
4. Model load varies 576-5394 s across runs of the same weights (TWELVE_HOUR_ESTIMATE).
5. NaN in the raw log is written as `null` in the committed `step_log.jsonl` and `trainer_state_equivalent.json` (original was not strict JSON).

## Resume / checkpoint: NOT VERIFIED
- Verified: adapter saved (0.275 s), file sha256 recorded, adapter reloaded with the expected 32,937,472 trainable params.
- Not verified: continuing training from a checkpoint. After `model.unload()`, `PeftModel.from_pretrained(is_trainable=True)` and one extra step, the extra step hit `torch.OutOfMemoryError` (2.82 GiB allocation, 13.06 GiB held by PyTorch) inside `scaler.scale(l).backward()`. The extra step had no OOM-skip guard, and memory from the old model, optimizer and activations probably had not been released, so the failure is likely a probe-harness artifact, but that was not tested. `resume_check` was never written to the summary; the pre-save vs reloaded loss comparison was computed but not saved, so there is no loss-continuity evidence.
- Design limit that is known from the script: only the adapter and `progress.json` ({"ep":0,"step":12,"elapsed":1664.4}) are checkpointed per epoch. AdamW moments, LR scheduler state and GradScaler state are not saved, so a restart is a warm start with fresh optimizer state, not an exact resume.
- The script's internal budget is 10.4 h (`TIME_BUDGET_S`). Kaggle's session cap is 12 h; behaviour at the cap beyond that was not tested.
