# HARDWARE

Kaggle GPU kernel, machine shape `NvidiaTeslaT4`: 2x Tesla T4 (compute capability 7.5), 14.56 GiB usable each (15360 MiB nvidia-smi total), driver 580.178.04. CPU model and host RAM were not recorded by the probe. GPU UUIDs redacted. Raw: `raw/gpu_samples.csv` (nvidia-smi every 10 s, 168 samples per GPU, 16:11:17-16:39:19Z), `raw/step_log.jsonl`.

## Did GPU1 train? NO.
Both GPUs were visible (`device_count` 2), but the recipe is a single process using cuda:0 only: no DDP, no device_map sharding, `CUDA_VISIBLE_DEVICES` and `WORLD_SIZE` unset. GPU1 showed 0% utilisation in all 168 samples, 3 MiB memory, max 16.5 W (idle). `torch.cuda.max_memory_allocated` per device: {0: 14307.3 MiB, 1: 0.0}. All numbers in this packet are single-T4 numbers.

## Per-GPU peaks
| | GPU0 | GPU1 |
|---|---|---|
| nvidia-smi util, max / mean over whole probe | 100% / 39.5% | 0% / 0% |
| util mean during training window (16:33:35-16:38:50, 32 samples) | 96.7% | 0% |
| nvidia-smi memory used, max | 14891 MiB | 3 MiB |
| memory used, mean in training window | 12150 MiB | 3 MiB |
| torch max allocated / max reserved | 14307.3 / 14762.0 MiB | 0 / 0 |
| power max (sampled) | 93.3 W | 16.5 W |

Memory is the binding constraint on one T4: peak allocation is within about 250 MiB of the 14.56 GiB device capacity. In the probe, 10 of 96 rows OOM-ed and were skipped (see TIMING/RELIABILITY); all were 6500-6946 tokens long. Per-step `cuda0_max_mem_alloc_mib` is in `raw/step_log.jsonl` (13546-14307 MiB).

Loading the 3.46 GB weights took 767 s in this run (other runs: see TWELVE_HOUR_ESTIMATE). 3423 MiB was allocated on cuda:0 after the PEFT wrap.
