# Deriver LoRA (LFM3-1.7B-Base) - 2xT4 DDP Trial Run 1 Evidence, 2026-11-15

This directory contains benchmark logs, telemetry, and failure analysis for Trial Run 1 (`<kaggle-user>/p2000-deriver-v15-sprint-a1`) executed on 2x Tesla T4 GPUs using PyTorch `torchrun` with DDP.

## Executive Summary

- **Kernel Reference**: `<kaggle-user>/p2000-deriver-v15-sprint-a1`
- **Execution Target**: 2x Tesla T4 GPUs (`cuda:0`, `cuda:1`, `world_size=2`)
- **Kernel Status**: `KernelWorkerStatus.ERROR`
- **DDP Verification**: `false` (no training steps completed due to startup exception)
- **Steps Completed**: 0
- **Peak VRAM (Telemetry)**: GPU 0: 3,579 MiB, GPU 1: 3,579 MiB (during initial model weight transfer)
- **Trial 2 Launch Status**: BLOCKED / NOT LAUNCHED (Safety precondition failed: Trial 1 exited with error)

## Root Cause Analysis

### 1. Failure Symptom
During torchrun distributed launch, both rank 0 and rank 1 processes crashed immediately upon instantiating `CurriculumCompletionTrainer`:
```
[rank0]: Traceback (most recent call last):
[rank0]:   File "/kaggle/input/datasets/<kaggle-user>/p2000-deriver-v11-sprint-payload/train_deriver.py", line 1408, in <module>
[rank0]:     main()
[rank0]:   File "/kaggle/input/datasets/<kaggle-user>/p2000-deriver-v11-sprint-payload/train_deriver.py", line 1325, in main
[rank0]:     trainer = CurriculumCompletionTrainer(...)
[rank0]:   File "/usr/local/lib/python3.13/dist-packages/transformers/trainer.py", line 504, in __init__
[rank0]:     self._move_model_to_device(model, args.device)
[rank0]:   File "/usr/local/lib/python3.13/dist-packages/transformers/trainer.py", line 4723, in _move_model_to_device
[rank0]:     model = model.to(device)
[rank0]:   File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 1384, in to
[rank0]:     return self._apply(convert)
[rank0]:   File "/usr/local/lib/python3.13/dist-packages/torch/nn/modules/module.py", line 934, in _apply
[rank0]:     module._apply(fn)
...
[rank0]:   [Previous line repeated 988 more times]
[rank0]: RecursionError: maximum recursion depth exceeded
```

### 2. Underlying Bug Mechanism
In `train_deriver.py`, `patch_lm_head_for_sliced_training(model)` binds a back-reference to the parent model directly onto `lm_head`:
```python
lm_head._model_ref = model
```
Because both `model` and `lm_head` inherit from `torch.nn.Module`, assigning an `nn.Module` to an attribute of another `nn.Module` automatically registers `model` as a child submodule of `lm_head` via `nn.Module.__setattr__` (`self._modules['_model_ref'] = value`).
Since `lm_head` is already a submodule of `model`, this creates a cyclic dependency in the module graph:
`model` -> `lm_head` -> `_model_ref (model)` -> `lm_head` -> ...

When `Trainer._move_model_to_device(model, args.device)` calls `model.to(device)`, PyTorch iterates over `self.children()` recursively via `_apply()`, falling into an infinite loop until `RecursionError` is raised.

### 3. Required Remediation for Future Sprints
To avoid PyTorch submodule registration, back-references between `nn.Module` instances must be assigned using `object.__setattr__(lm_head, '_model_ref', model)` or a `weakref.ref(model)`.

## Telemetry Summary
- Both T4 GPUs were identified with distinct hardware UUIDs (`<gpu-0-uuid>` and `<gpu-1-uuid>`).
- VRAM occupancy climbed symmetrically to 3,579 MiB on both devices as weights were loaded before the crash.
- Run lock was recorded successfully in `run-lock.json`.
