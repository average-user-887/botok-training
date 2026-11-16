"""Patched 2xT4 DDP multi-role training recipe for LFM3-1.7B-Base. CPU-static-checked; does not run on import."""
import argparse
import dataclasses
import datetime
import glob
import hashlib
import importlib
import json
import math
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc
import random
import re
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import Trainer, TrainerCallback
except ImportError:
    Trainer = object
    TrainerCallback = object

MODEL = 'LiquidAI/LFM3-1.7B-Base'
REVISION = '7bca5038823c0d0f14efe06439109354a10e014c'
DEFAULT_LOCAL_MODEL_DIR = '<models-root>/hf/LFM3-1.7B-Base'
ROLES = ('deriver', 'pruner', 'judge', 'verifier', 'scribe')
TARGET_MODULE_PRESETS = {
    'attn-mlp': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'w1', 'w2', 'w3'],
    'all-linear': 'all-linear',
}
KEEP_RE = re.compile(r'^KEEP (none|all|\d+(-\d+)?(,\d+(-\d+)?)*)$')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _adapter_weights_sha256(adapter_dir):
    if not adapter_dir:
        return None
    p = Path(adapter_dir)
    for fname in ('adapter_model.safetensors', 'adapter_model.bin'):
        w = p / fname
        if w.is_file():
            return digest(w)
    return None


def validate_non_test_path(path):
    p = Path(path)
    name = p.name.lower()
    if re.search(r'(^|[._-])(test|sealed|holdout)([._-]|$)', name) or any(
        part.lower() == 'sealed' for part in p.parts[:-1]
    ):
        raise ValueError(f'Refusing held-out/test/sealed file path: {path}')


def resolve_model_dir(model_dir=None, search_roots=('/kaggle/input',), fallback_dir=DEFAULT_LOCAL_MODEL_DIR):
    """Resolve an offline LFM3 model directory from CLI arg, env, /kaggle/input, or local NVMe fallback."""
    if model_dir:
        p = Path(model_dir)
        assert (p / 'config.json').is_file(), f'Missing config.json in --model-dir: {model_dir}'
        return p
    env_dir = os.environ.get('MODEL_DIR')
    if env_dir and (Path(env_dir) / 'config.json').is_file():
        return Path(env_dir)
    hits = []
    for root in search_roots:
        for cfg_path in glob.glob(str(Path(root) / '**' / 'config.json'), recursive=True):
            parent = Path(cfg_path).parent
            try:
                cfg = json.loads(Path(cfg_path).read_text())
            except Exception:
                continue
            archs = [str(x).lower() for x in cfg.get('architectures', [])]
            is_lfm3 = cfg.get('model_type') == 'lfm3' or any('lfm3' in a for a in archs)
            if is_lfm3 and (parent / 'tokenizer.json').is_file() and list(parent.glob('*.safetensors')):
                hits.append(parent)
    if hits:
        return sorted(hits, key=lambda p: len(str(p)))[0]
    if fallback_dir:
        fallback = Path(fallback_dir)
        if (fallback / 'config.json').is_file():
            return fallback
    raise FileNotFoundError(
        'Offline LFM3 model directory not found; pass --model-dir or attach lfm3-1p7b-base under /kaggle/input'
    )


def disable_peft_torchao():
    """Disable peft's torchao integration before LoRA wrap to avoid peft 0.21.2 + torchao 0.10.0 import/runtime errors."""
    for mod_name in ('peft.import_utils', 'peft.tuners.lora.torchao'):
        try:
            importlib.import_module(mod_name)
        except Exception:
            pass
    for _m in list(sys.modules.values()):
        if getattr(_m, '__name__', '').startswith('peft') and hasattr(_m, 'is_torchao_available'):
            setattr(_m, 'is_torchao_available', lambda: False)


def apply_peft_lora(model, lora_rank=16, lora_alpha=32, target_modules='all-linear', resume_adapter=None):
    """Apply PEFT LoRA (with torchao disabled) and enable input require grads for gradient checkpointing."""
    from peft import LoraConfig, PeftModel, get_peft_model

    disable_peft_torchao()
    if resume_adapter:
        model = PeftModel.from_pretrained(model, str(resume_adapter), is_trainable=True)
    else:
        resolved_targets = TARGET_MODULE_PRESETS.get(target_modules, target_modules)
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias='none',
                task_type='CAUSAL_LM',
                target_modules=resolved_targets,
            ),
        )
    model.enable_input_require_grads()
    # PEFT normally promotes adapter weights, explicitly enforce FP32 trainable weights.
    for x in model.parameters():
        if x.requires_grad:
            x.data = x.data.float()
    enable_lfm3_gradient_checkpointing(model)
    patch_lm_head_for_sliced_training(model)
    return model


def validate_resume_checkpoint(checkpoint_path, out_dir=None, world=1):
    """Validate that a resume checkpoint directory contains all required state files."""
    checkpoint = Path(checkpoint_path).resolve()
    if out_dir is not None:
        assert checkpoint.parent == Path(out_dir).resolve(), 'Resume checkpoint must belong to exact output directory'
    assert checkpoint.name.startswith('checkpoint-') and (
        checkpoint.name[11:].isdigit() or checkpoint.name == 'checkpoint-latest'
    ), 'Checkpoint directory name must match checkpoint-<step> or checkpoint-latest'
    for name in ['trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'adapter_config.json']:
        assert (checkpoint / name).is_file(), f'Incomplete checkpoint: missing {name}'
    rng_files = ['rng_state.pth'] if world == 1 else [f'rng_state_{r}.pth' for r in range(world)]
    for name in rng_files:
        assert (checkpoint / name).is_file(), f'Missing RNG state: {name}'
    assert (checkpoint / 'adapter_model.safetensors').is_file() or (checkpoint / 'adapter_model.bin').is_file(), 'Missing adapter model weights'
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    if checkpoint.name != 'checkpoint-latest':
        assert state['global_step'] == int(checkpoint.name[11:]), 'Checkpoint step mismatch'
    return True


def find_lfm3_layers(model):
    """Traverse model wrapper hierarchy to locate Lfm3DecoderLayer module list."""
    curr = model
    for _ in range(6):
        if curr is None:
            break
        layers = getattr(curr, 'layers', None)
        if layers is not None and hasattr(layers, '__iter__'):
            return layers
        if hasattr(curr, 'model'):
            curr = curr.model
        elif hasattr(curr, 'base_model'):
            curr = curr.base_model
        elif hasattr(curr, 'module'):
            curr = curr.module
        else:
            break
    return None


def disable_lfm3_gradient_checkpointing(model):
    """Temporarily disable gradient checkpointing on Lfm3DecoderLayer modules (e.g. during .generate)."""
    layers = find_lfm3_layers(model)
    if layers:
        for layer in layers:
            layer._gradient_checkpointing_enabled = False


def enable_lfm3_gradient_checkpointing(model):
    """Enable torch.utils.checkpoint on Lfm3DecoderLayer modules because Lfm3Model lacks native checkpointing."""
    import torch.utils.checkpoint

    layers = find_lfm3_layers(model)
    if not layers:
        return
    for layer in layers:
        layer._gradient_checkpointing_enabled = True
        if getattr(layer, '_checkpointing_wrapped', False):
            continue
        orig_forward = layer.forward

        def _make_cp(fn, lyr):
            def _cp_forward(hidden_states, *args, **kwargs):
                if getattr(lyr, 'training', False) and getattr(lyr, '_gradient_checkpointing_enabled', True):
                    def _custom_forward(h, *a):
                        return fn(h, *a, **kwargs)
                    return torch.utils.checkpoint.checkpoint(_custom_forward, hidden_states, *args, use_reentrant=False)
                return fn(hidden_states, *args, **kwargs)
            return _cp_forward

        layer.forward = _make_cp(orig_forward, layer)
        layer._checkpointing_wrapped = True


def find_lm_head(model):
    """Traverse model wrapper hierarchy to locate lm_head Linear module."""
    curr = model
    for _ in range(6):
        if curr is None:
            break
        if hasattr(curr, 'lm_head') and (isinstance(curr.lm_head, torch.nn.Linear) or hasattr(curr.lm_head, 'weight')):
            return curr.lm_head
        if hasattr(curr, 'base_model'):
            curr = curr.base_model
        elif hasattr(curr, 'model'):
            curr = curr.model
        elif hasattr(curr, 'module'):
            curr = curr.module
        else:
            break
    return getattr(model, 'lm_head', None)


def patch_lm_head_for_sliced_training(model):
    """Wrap/monkey-patch lm_head on underlying causal LM to slice hidden_states before F.linear."""
    import types

    lm_head = find_lm_head(model)
    if lm_head is None or getattr(lm_head, '_sliced_wrapped', False):
        return lm_head

    slice_state = {"keep": None}
    orig_forward = lm_head.forward

    def sliced_forward(self, x, *args, **kwargs):
        keep = slice_state.get('keep', None)
        if keep is None and hasattr(self, '_lm_head_slice_state') and isinstance(self._lm_head_slice_state, dict):
            keep = self._lm_head_slice_state.get('keep', None)
        if keep is None:
            keep = getattr(self, '_keep_tokens', None)
        if keep is not None and keep > 0 and hasattr(x, 'ndim') and x.ndim == 3 and x.shape[1] > keep:
            x = x[:, -(keep + 1):-1, :].contiguous()
        return orig_forward(x, *args, **kwargs)

    lm_head.forward = types.MethodType(sliced_forward, lm_head)
    lm_head._sliced_wrapped = True
    object.__setattr__(lm_head, "_lm_head_slice_state", slice_state)
    lm_head._keep_tokens = None

    curr = model
    for _ in range(6):
        if curr is None:
            break
        object.__setattr__(curr, "_lm_head_slice_state", slice_state)
        if hasattr(curr, 'module'):
            curr = curr.module
        elif hasattr(curr, 'base_model'):
            curr = curr.base_model
        elif hasattr(curr, 'model'):
            curr = curr.model
        else:
            break
    return lm_head


def set_model_keep_tokens(model, keep):
    curr = model
    for _ in range(6):
        if curr is None:
            break
        try:
            curr._keep_tokens = keep
        except Exception:
            pass
        if hasattr(curr, '_lm_head_slice_state') and isinstance(curr._lm_head_slice_state, dict):
            try:
                curr._lm_head_slice_state['keep'] = keep
            except Exception:
                pass
        if hasattr(curr, 'lm_head'):
            try:
                curr.lm_head._keep_tokens = keep
            except Exception:
                pass
            if hasattr(curr.lm_head, '_lm_head_slice_state') and isinstance(curr.lm_head._lm_head_slice_state, dict):
                try:
                    curr.lm_head._lm_head_slice_state['keep'] = keep
                except Exception:
                    pass
        if hasattr(curr, 'module'):
            curr = curr.module
        elif hasattr(curr, 'base_model'):
            curr = curr.base_model
        elif hasattr(curr, 'model'):
            curr = curr.model
        else:
            break


def _validate_deriver_row(r):
    m = r['messages']
    assert [x['role'] for x in m] == ['user', 'assistant'], 'Use exact native minimal-deriver prompt; no system role'
    obj = json.loads(m[-1]['content'])
    assert set(obj) == {'explicit'} and isinstance(obj['explicit'], list)
    for f in obj['explicit']:
        assert isinstance(f, dict) and set(f) == {'content'}, 'Native minimal deriver emits content-only facts'
        assert isinstance(f['content'], str) and f['content'].strip()


def read_rows(path, role='deriver'):
    assert role in ROLES, f'Unsupported role {role}; expected one of {ROLES}'
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    for r in rows:
        if role == 'deriver':
            _validate_deriver_row(r)
        elif role == 'pruner':
            assert 'prompt' in r and 'completion' in r, 'Pruner rows require prompt and completion fields'
            assert isinstance(r['prompt'], str) and r['prompt'].strip()
            assert isinstance(r['completion'], str) and KEEP_RE.match(r['completion'].strip()), (
                f'Invalid pruner completion: {r.get("completion")!r}'
            )
        else:
            if 'messages' in r:
                m = r['messages']
                assert [x['role'] for x in m] == ['user', 'assistant'], f'{role} messages must be [user, assistant]'
                completion_text = m[-1]['content']
            else:
                assert 'prompt' in r and 'completion' in r, f'{role} row must use messages or prompt/completion'
                assert isinstance(r['prompt'], str) and r['prompt'].strip()
                completion_text = r['completion']
            assert isinstance(completion_text, str) and completion_text.strip()
            if role in ('judge', 'verifier'):
                obj = json.loads(completion_text)
                assert isinstance(obj, dict), f'{role} completion must be a JSON object'
    assert rows
    return rows


def encode_row(r, tokenizer, max_seq_len=5600, role='deriver', over_limit_policy='reject', return_admission=False):
    assert tokenizer.eos_token_id is not None
    if 'messages' in r:
        messages = r['messages']
        prefix = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
        )
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=False
        )
        assert rendered[: len(prefix)] == prefix, 'Chat prefix mismatch; stop rather than mis-mask'
        source_text = messages[-2]['content']
        gold_text = messages[-1]['content']
        completion = tokenizer(gold_text, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    else:
        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        source_text = r['prompt']
        gold_text = r['completion']
        prefix = bos + tokenizer(source_text, add_special_tokens=False).input_ids
        completion = tokenizer(gold_text, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    orig_prompt_len = len(prefix)
    comp_len = len(completion)
    full_len = orig_prompt_len + comp_len
    truncation_side = 'none'
    truncated_tokens = 0
    if max_seq_len is not None and full_len > max_seq_len:
        if over_limit_policy == 'reject':
            raise ValueError(
                f"Row {r.get('id', '?')}: {full_len} tokens exceeds max_seq_len={max_seq_len} (over_limit_policy='reject')"
            )
        if over_limit_policy != 'middle-truncate-prompt':
            raise ValueError(f'Unsupported over_limit_policy: {over_limit_policy}')
        keep_prefix = max_seq_len - comp_len
        assert keep_prefix > 0, f'Completion length {comp_len} exceeds max_seq_len {max_seq_len}'
        head_len = keep_prefix // 2
        tail_len = keep_prefix - head_len
        prefix = prefix[:head_len] + prefix[len(prefix) - tail_len:]
        truncation_side = 'prompt_middle'
        truncated_tokens = full_len - (len(prefix) + comp_len)
    full = prefix + completion
    assert len(full) > len(prefix) > 0, 'Both prompt prefix and completion must be non-empty'
    is_empty = _is_empty_gold(r, role)
    item = {'input_ids': full, 'labels': [-100] * len(prefix) + completion, 'is_empty': is_empty}
    if not return_admission:
        return item
    admission = {
        'id': r.get('id'),
        'original_prompt_tokens': orig_prompt_len,
        'completion_tokens': comp_len,
        'original_total_tokens': full_len,
        'retained_prompt_tokens': len(prefix),
        'retained_total_tokens': len(full),
        'truncation_side': truncation_side,
        'truncated_tokens': truncated_tokens,
        'admitted': True,
        'source_sha256': hashlib.sha256(source_text.encode('utf-8')).hexdigest(),
        'gold_sha256': hashlib.sha256(gold_text.encode('utf-8')).hexdigest(),
    }
    return item, admission


def build_admission_receipt(rows, tokenizer, max_seq_len=5600, role='deriver', over_limit_policy='reject'):
    """Encode rows with full per-row admission metadata; records rejections cleanly before raising."""
    encoded = []
    receipt_rows = []
    rejected_ids = []
    for idx, r in enumerate(rows):
        rid = r.get('id', f'row-{idx}')
        try:
            ex, adm = encode_row(
                r,
                tokenizer,
                max_seq_len=max_seq_len,
                role=role,
                over_limit_policy=over_limit_policy,
                return_admission=True,
            )
            encoded.append(ex)
            receipt_rows.append(adm)
        except ValueError:
            _, adm = encode_row(
                r,
                tokenizer,
                max_seq_len=None,
                role=role,
                over_limit_policy=over_limit_policy,
                return_admission=True,
            )
            adm['admitted'] = False
            adm['truncation_side'] = 'rejected_over_limit'
            adm['retained_prompt_tokens'] = 0
            adm['retained_total_tokens'] = 0
            receipt_rows.append(adm)
            rejected_ids.append(rid)
    receipt = {
        'policy': over_limit_policy,
        'max_length': max_seq_len,
        'role': role,
        'total_rows': len(rows),
        'admitted_rows': sum(1 for x in receipt_rows if x['admitted']),
        'rejected_rows': len(rejected_ids),
        'rejected_ids': rejected_ids,
        'truncated_rows': sum(1 for x in receipt_rows if x['truncated_tokens'] > 0),
        'excess_tokens_total': sum(x['truncated_tokens'] for x in receipt_rows),
        'original_total_tokens': sum(x['original_total_tokens'] for x in receipt_rows),
        'retained_total_tokens': sum(x['retained_total_tokens'] for x in receipt_rows),
        'rows': receipt_rows,
    }
    return encoded, receipt


def encode_prompt_and_gold(r, tokenizer):
    if 'messages' in r:
        messages = r['messages']
        prefix = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
        )
        gold_text = messages[-1]['content']
    else:
        bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        prefix = bos + tokenizer(r['prompt'], add_special_tokens=False).input_ids
        gold_text = r['completion']
    comp_len = len(tokenizer(gold_text, add_special_tokens=False).input_ids) + 1
    return prefix, gold_text, len(prefix) + comp_len


def apply_curriculum_order(encoded, curriculum_order='none'):
    if curriculum_order == 'none':
        return list(encoded)
    if curriculum_order == 'medium-to-long':
        return sorted(encoded, key=lambda x: len(x['input_ids']))
    raise ValueError(f'Unknown curriculum_order: {curriculum_order}')


def order_non_empty_warmup(encoded, warmup_ratio=0.25):
    """Ensure non-empty rows are scheduled during warmup steps (first 25% of steps)."""
    n = len(encoded)
    warmup_n = math.ceil(n * warmup_ratio)
    non_empty_indices = [i for i, x in enumerate(encoded) if not x.get('is_empty', False)]
    if len(non_empty_indices) >= warmup_n:
        warmup_set = set(non_empty_indices[:warmup_n])
        first_part = [encoded[i] for i in non_empty_indices[:warmup_n]]
        second_part = [x for i, x in enumerate(encoded) if i not in warmup_set]
        return first_part + second_part
    else:
        empty_indices = [i for i, x in enumerate(encoded) if x.get('is_empty', False)]
        return [encoded[i] for i in non_empty_indices] + [encoded[i] for i in empty_indices]


def order_stress_longest_first(encoded, stress_n=8, curriculum_order='none', warmup_ratio=0.25):
    """Place the stress_n longest sequences in the first optimizer step to prove peak VRAM on step 1,
    followed by the remaining rows ordered by curriculum and non-empty warmup."""
    if len(encoded) <= stress_n:
        return sorted(encoded, key=lambda x: len(x['input_ids']), reverse=True)

    sorted_indices = sorted(range(len(encoded)), key=lambda i: len(encoded[i]['input_ids']), reverse=True)
    stress_indices = set(sorted_indices[:stress_n])
    stress_rows = [encoded[i] for i in sorted_indices[:stress_n]]
    remaining_rows = [x for i, x in enumerate(encoded) if i not in stress_indices]

    remaining_rows = apply_curriculum_order(remaining_rows, curriculum_order)
    remaining_rows = order_non_empty_warmup(remaining_rows, warmup_ratio=warmup_ratio)

    return stress_rows + remaining_rows


def _norm_tokens(s):
    return set(re.findall(r'[a-z0-9]+', s.lower()))


def _jaccard_match(a, b):
    x, y = _norm_tokens(a), _norm_tokens(b)
    return len(x & y) / max(1, len(x | y)) >= 0.5


def _parse_keep_set(spec, n):
    if spec == 'none':
        return set()
    if spec == 'all':
        return set(range(1, n + 1))
    out = set()
    for part in spec.split(','):
        a, _, b = part.partition('-')
        lo, hi = int(a), int(b or a)
        if hi < lo or hi - lo > 100000:
            return None
        out.update(range(lo, hi + 1))
    return out


def _numbered_lines(prompt):
    return {int(m.group(1)): m.group(2) for m in re.finditer(r'(?m)^(\d+)\| ?(.*)$', prompt)}


def _is_empty_gold(r, role='deriver'):
    raw = None
    if 'messages' in r:
        raw = r['messages'][-1]['content']
    elif 'completion' in r:
        comp = r['completion'].strip()
        if comp == 'KEEP none' or comp == '{"explicit": []}':
            return True
        raw = comp
    if raw is not None:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                if obj.get('explicit') == []:
                    return True
                if role == 'scribe' and obj.get('items') == []:
                    return True
                if role == 'verifier' and {'timeline', 'findings', 'unknowns', 'next_checks'}.issubset(obj.keys()):
                    return all(obj.get(k) == [] for k in ('timeline', 'findings', 'unknowns', 'next_checks'))
                list_vals = [v for v in obj.values() if isinstance(v, list)]
                if list_vals and all(len(v) == 0 for v in list_vals):
                    return True
                if 'verdict' in obj and not str(obj.get('verdict') or '').strip() and not any(
                    isinstance(v, list) and len(v) > 0 for v in obj.values()
                ):
                    return True
        except Exception:
            pass
    return False


def oversample_empty_rows(rows, role='deriver', target_ratio=0.0, seed=42):
    """Deterministically oversample empty-gold training rows up to target_ratio."""
    if not (math.isfinite(target_ratio) and 0.0 <= target_ratio < 1.0):
        raise ValueError(f'empty_target_ratio must be finite and in [0.0, 1.0), got {target_ratio}')
    if not rows or target_ratio == 0.0:
        return list(rows)
    empty_rows = [r for r in rows if _is_empty_gold(r, role=role)]
    non_empty_count = len(rows) - len(empty_rows)
    if not empty_rows or non_empty_count == 0:
        return list(rows)
    if len(empty_rows) / len(rows) >= target_ratio:
        return list(rows)
    needed = math.ceil((target_ratio * non_empty_count) / (1.0 - target_ratio) - len(empty_rows))
    if needed <= 0:
        return list(rows)
    rng = random.Random(seed)
    full_cycles, rem = divmod(needed, len(empty_rows))
    extra = []
    for _ in range(full_cycles):
        extra.extend(empty_rows)
    if rem > 0:
        indices = list(range(len(empty_rows)))
        rng.shuffle(indices)
        extra.extend(empty_rows[i] for i in indices[:rem])
    return list(rows) + extra


def select_stratified_probe(dev_rows, tokenizer, role='deriver', max_rows=20):
    if not dev_rows:
        return []
    if len(dev_rows) <= max_rows:
        return list(dev_rows)
    buckets = {'empty': [], 'short': [], 'medium': [], 'long': []}
    for r in dev_rows:
        _, _, tok_len = encode_prompt_and_gold(r, tokenizer)
        if _is_empty_gold(r, role):
            buckets['empty'].append(r)
        elif tok_len < 1500:
            buckets['short'].append(r)
        elif tok_len < 3500:
            buckets['medium'].append(r)
        else:
            buckets['long'].append(r)
    selected = []
    seen_idx = set()
    cursors = {k: 0 for k in buckets}
    keys = [k for k in ('short', 'medium', 'long', 'empty') if buckets[k]]
    while len(selected) < max_rows and keys:
        progressed = False
        for k in keys:
            if len(selected) >= max_rows:
                break
            idx = cursors[k]
            if idx < len(buckets[k]):
                row = buckets[k][idx]
                cursors[k] += 1
                obj_id = id(row)
                if obj_id not in seen_idx:
                    selected.append(row)
                    seen_idx.add(obj_id)
                progressed = True
        if not progressed:
            break
    return selected


def score_prediction(role, row, pred_text, finished=True):
    """Score one prediction for a given role. Returns (schema_valid, tp, pred_n, gold_n, gold_tp, is_empty_gold, empty_fp)."""
    if role == 'deriver':
        gold = [o['content'] for o in json.loads(row['messages'][-1]['content'])['explicit']]
        is_empty_gold = len(gold) == 0
        try:
            obj = json.loads(pred_text)
            valid_shape = (
                isinstance(obj, dict)
                and set(obj) == {'explicit'}
                and isinstance(obj['explicit'], list)
                and all(
                    isinstance(f, dict) and set(f) == {'content'} and isinstance(f['content'], str) and f['content'].strip()
                    for f in obj['explicit']
                )
            )
            pred = [f['content'] for f in obj['explicit']] if valid_shape else []
            ok = bool(finished and valid_shape)
        except Exception:
            pred, ok = [], False
        tp = sum(any(_jaccard_match(p, g) for g in gold) for p in pred)
        gtp = sum(any(_jaccard_match(p, g) for p in pred) for g in gold)
        empty_fp = is_empty_gold and len(pred) > 0
        return ok, tp, len(pred), len(gold), gtp, is_empty_gold, empty_fp

    if role == 'pruner':
        prompt = row['prompt']
        lines = _numbered_lines(prompt)
        n_lines = max(lines) if lines else int(row.get('meta', {}).get('n_lines', 0))
        gold_spec = row['completion'].strip().removeprefix('KEEP ').strip()
        is_empty_gold = gold_spec == 'none'
        gold = _parse_keep_set(gold_spec, n_lines) or set()
        stripped = pred_text.strip()
        ok = bool(finished and KEEP_RE.match(stripped))
        pred = _parse_keep_set(stripped[5:], n_lines) if ok else None
        if pred is None:
            return False, 0, 0, len(gold), 0, is_empty_gold, False
        inter = len(pred & gold)
        empty_fp = is_empty_gold and len(pred) > 0
        return ok, inter, len(pred), len(gold), inter, is_empty_gold, empty_fp

    gold_raw = row['messages'][-1]['content'] if 'messages' in row else row['completion']
    try:
        gold_obj = json.loads(gold_raw)
    except Exception:
        gold_obj = None
    try:
        pred_obj = json.loads(pred_text)
    except Exception:
        pred_obj = None
    is_empty_gold = False
    empty_fp = False
    if role == 'judge':
        ok = bool(
            finished
            and isinstance(pred_obj, dict)
            and {'matched', 'missed', 'hallucinated', 'verdict'}.issubset(pred_obj.keys())
        )
    elif role == 'verifier':
        if isinstance(gold_obj, dict) and {'timeline', 'findings', 'unknowns', 'next_checks'}.issubset(gold_obj.keys()):
            section_specs = {
                'timeline': {'text', 'evidence'},
                'findings': {'kind', 'text', 'evidence'},
                'unknowns': {'question', 'evidence'},
                'next_checks': {'action', 'purpose', 'evidence'},
            }
            valid_findings_kinds = {'observed_error', 'observed_recovery', 'hypothesis', 'scope_limit'}
            is_empty_gold = all(
                isinstance(gold_obj.get(sec), list) and len(gold_obj[sec]) == 0
                for sec in section_specs
            )
            valid_log_analyser = (
                isinstance(pred_obj, dict)
                and set(pred_obj) == set(section_specs)
                and all(isinstance(pred_obj.get(sec), list) for sec in section_specs)
                and all(
                    isinstance(it, dict)
                    and set(it) == keys
                    and isinstance(it.get('evidence'), list)
                    and len(it['evidence']) >= 1
                    and all(isinstance(ev, str) and bool(ev.strip()) for ev in it['evidence'])
                    and len(set(it['evidence'])) == len(it['evidence'])
                    and all(isinstance(it.get(k), str) and bool(it[k].strip()) for k in keys if k != 'evidence')
                    and (sec != 'findings' or it.get('kind') in valid_findings_kinds)
                    for sec, keys in section_specs.items()
                    for it in pred_obj[sec]
                )
            )
            ok = bool(finished and valid_log_analyser)
            empty_fp = (
                is_empty_gold
                and isinstance(pred_obj, dict)
                and any(isinstance(pred_obj.get(sec), list) and len(pred_obj[sec]) > 0 for sec in section_specs)
            )
        else:
            ok = bool(
                finished
                and isinstance(pred_obj, dict)
                and {'verified', 'blocking_issues', 'verdict'}.issubset(pred_obj.keys())
            )
    elif role == 'scribe' and isinstance(gold_obj, dict) and 'items' in gold_obj:
        is_empty_gold = isinstance(gold_obj.get('items'), list) and len(gold_obj['items']) == 0
        valid_items = (
            isinstance(pred_obj, dict)
            and set(pred_obj) == {'title', 'as_of', 'items'}
            and isinstance(pred_obj.get('title'), str)
            and bool(pred_obj['title'].strip())
            and isinstance(pred_obj.get('as_of'), str)
            and isinstance(pred_obj.get('items'), list)
            and all(
                isinstance(it, dict)
                and set(it) == {'kind', 'text', 'owner', 'due', 'evidence'}
                and it.get('kind') in ('state', 'decision', 'pending', 'uncertainty')
                and isinstance(it.get('text'), str)
                and bool(it['text'].strip())
                and (it.get('owner') is None or isinstance(it.get('owner'), str))
                and (it.get('due') is None or isinstance(it.get('due'), str))
                and isinstance(it.get('evidence'), list)
                and len(it['evidence']) >= 1
                and all(isinstance(ev, str) for ev in it['evidence'])
                and len(set(it['evidence'])) == len(it['evidence'])
                for it in pred_obj['items']
            )
        )
        ok = bool(finished and valid_items)
        empty_fp = is_empty_gold and isinstance(pred_obj, dict) and isinstance(pred_obj.get('items'), list) and len(pred_obj['items']) > 0
    else:
        ok = bool(finished and bool(pred_text.strip()) and (gold_obj is None or isinstance(pred_obj, dict)))
    g_tok = _norm_tokens(gold_raw)
    p_tok = _norm_tokens(pred_text)
    inter = len(g_tok & p_tok)
    return ok, inter, len(p_tok), len(g_tok), inter, is_empty_gold, empty_fp


def evaluate_dev_rows(model, tokenizer, dev_rows, role='deriver', max_new_tokens=1800, device='cuda:0'):
    import torch

    if not dev_rows:
        return {
            'dev_n': 0,
            'dev_schema_valid': None,
            'dev_precision': None,
            'dev_recall': None,
            'dev_f1': None,
            'dev_token_f1': None,
            'empty_fp_rate': None,
            'long_window_recall': None,
            'short_window_recall': None,
            'non_empty_pred_rate': None,
            'empty_collapse_detected': None,
        }, []

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    unwrapped = model.module if hasattr(model, 'module') else model
    was_training = unwrapped.training
    prev_cache = getattr(unwrapped.config, 'use_cache', False)
    base_cfg = getattr(getattr(getattr(unwrapped, 'base_model', None), 'model', None), 'config', None)
    prev_base_cache = getattr(base_cfg, 'use_cache', prev_cache) if base_cfg is not None else None
    unwrapped.eval()
    unwrapped.config.use_cache = True
    if base_cfg is not None:
        base_cfg.use_cache = True

    disable_lfm3_gradient_checkpointing(unwrapped)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    use_cuda = str(device).startswith('cuda') and torch.cuda.is_available()

    valid = tp = pn = gn = gtp = 0
    empty_total = empty_fp = 0
    long_gn = long_gtp = short_gn = short_gtp = 0
    outputs = []
    row_tok_f1s = []

    try:
        for idx, r in enumerate(dev_rows):
            t_row0 = time.time()
            prefix, gold_text, total_len = encode_prompt_and_gold(r, tokenizer)
            ids = torch.tensor([prefix], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            amp_ctx = torch.amp.autocast('cuda', dtype=torch.float16) if use_cuda else nullcontext()
            with torch.inference_mode(), amp_ctx:
                out = unwrapped.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    use_cache=True,
                )
            gen = out[0, len(prefix) :].tolist()
            finished = eos_id in gen
            stop_type = 'eos' if finished else 'limit'
            text = tokenizer.decode([g for g in gen if g != eos_id], skip_special_tokens=True).strip()
            ok, r_tp, r_pn, r_gn, r_gtp, is_empty_gold, is_empty_fp = score_prediction(role, r, text, finished=finished)
            valid += int(ok)
            tp += r_tp
            pn += r_pn
            gn += r_gn
            gtp += r_gtp
            if is_empty_gold:
                empty_total += 1
                empty_fp += int(is_empty_fp)
            if total_len >= 3500:
                long_gn += r_gn
                long_gtp += r_gtp
            else:
                short_gn += r_gn
                short_gtp += r_gtp

            g_tok = _norm_tokens(gold_text)
            p_tok = _norm_tokens(text)
            inter_tok = len(g_tok & p_tok)
            p_prec = inter_tok / len(p_tok) if p_tok else (1.0 if not g_tok else 0.0)
            p_rec = inter_tok / len(g_tok) if g_tok else (1.0 if not p_tok else 0.0)
            r_tok_f1 = (2.0 * p_prec * p_rec / (p_prec + p_rec)) if (p_prec + p_rec) > 0 else 0.0
            row_tok_f1s.append(r_tok_f1)

            outputs.append(
                {
                    'id': r.get('id', f'row-{idx}'),
                    'ok': ok,
                    'finished': finished,
                    'stop_type': stop_type,
                    'prompt_tokens': len(prefix),
                    'tokens_predicted': len(gen),
                    'tokens': total_len,
                    'tp': r_tp,
                    'gtp': r_gtp,
                    'fp': max(0, r_pn - r_tp),
                    'fn': max(0, r_gn - r_gtp),
                    'n_pred': r_pn,
                    'n_gold': r_gn,
                    'elapsed_s': round(time.time() - t_row0, 4),
                    'text': text,
                }
            )
    finally:
        unwrapped.config.use_cache = prev_cache
        if base_cfg is not None:
            base_cfg.use_cache = prev_base_cache
        enable_lfm3_gradient_checkpointing(unwrapped)
        if was_training:
            unwrapped.train()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n = len(dev_rows)
    prec = tp / max(1, pn) if (pn > 0 or gn > 0) else 1.0
    rec = gtp / max(1, gn) if gn > 0 else (1.0 if pn == 0 else 0.0)
    pred_micro_rec = tp / max(1, gn) if gn > 0 else (1.0 if pn == 0 else 0.0)
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    dev_token_f1 = sum(row_tok_f1s) / len(row_tok_f1s) if row_tok_f1s else 0.0
    non_empty_preds = sum(1 for o in outputs if o['n_pred'] > 0)
    non_empty_pred_rate = (non_empty_preds / n) if n > 0 else 0.0
    empty_collapse_detected = bool(n > 0 and gn > 0 and non_empty_preds == 0)
    capped_ids = [o['id'] for o in outputs if not o['finished']]
    metrics = {
        'dev_n': n,
        'dev_schema_valid': valid / n if n > 0 else 0.0,
        'dev_precision': prec,
        'dev_recall': rec,
        'pred_micro_recall': pred_micro_rec,
        'dev_f1': f1,
        'dev_token_f1': dev_token_f1,
        'empty_fp_rate': (empty_fp / empty_total) if empty_total > 0 else 0.0,
        'long_window_recall': (long_gtp / long_gn) if long_gn > 0 else None,
        'short_window_recall': (short_gtp / short_gn) if short_gn > 0 else None,
        'tp': tp,
        'gtp': gtp,
        'fp': max(0, pn - tp),
        'fn': max(0, gn - gtp),
        'pred_items': pn,
        'gold_items': gn,
        'max_new_tokens': max_new_tokens,
        'capped_count': len(capped_ids),
        'capped_ids': capped_ids,
        'non_empty_pred_rate': non_empty_pred_rate,
        'empty_collapse_detected': empty_collapse_detected,
    }
    return metrics, outputs


def parse_milestones(spec):
    if not spec:
        return []
    if isinstance(spec, (list, tuple)):
        vals = [int(x) for x in spec]
    else:
        vals = [int(x.strip()) for x in str(spec).split(',') if x.strip()]
    return sorted({v for v in vals if 0 < v < 100})


class CompletionOnlyTrainer(Trainer):
    # Intentional equal-microbatch mean (equal-row mean for default microbatch=1), not global token normalization.
    loss_is_scaled_for_ga = False

    def __init__(self, *args, no_downweight_empty=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_downweight_empty = bool(no_downweight_empty)
        self.model_accepts_loss_kwargs = False
        self.rank_rows_processed = 0
        self.rank_non_pad_tokens = 0
        self.rank_supervised_tokens = 0
        self.rank_nan_skips = 0
        self.rank_oom_skips = 0
        self.rank_effective_steps = 0
        self._configure_grad_scaler()

    def _configure_grad_scaler(self):
        if hasattr(self, 'accelerator') and getattr(self.accelerator, 'scaler', None) is not None:
            scaler = getattr(self.accelerator.scaler, '_scaler', self.accelerator.scaler)
            scaler._init_scale = 1024.0
            if hasattr(scaler, '_scale') and scaler._scale is not None:
                if hasattr(scaler._scale, 'fill_'):
                    scaler._scale.fill_(1024.0)

    def create_accelerator_and_postprocess(self):
        super().create_accelerator_and_postprocess()
        self._configure_grad_scaler()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Restrict expensive vocabulary logits, not attention/context. All prompt tokens still condition the model.
        labels = inputs.pop('labels')
        start = inputs.pop('completion_start')
        is_empty = inputs.pop('is_empty', None)
        is_empty_target = inputs.pop('is_empty_target', None)
        assert type(start) == int and 0 < start < labels.shape[1]
        keep = labels.shape[1] - start

        input_ids = inputs.get('input_ids')
        if input_ids is not None:
            self.rank_rows_processed += input_ids.shape[0]
            tok = getattr(self, 'processing_class', None)
            pad_id = getattr(tok, 'pad_token_id', None)
            if pad_id is not None:
                self.rank_non_pad_tokens += int((input_ids != pad_id).sum().item())
            else:
                self.rank_non_pad_tokens += input_ids.numel()

        if torch is not None and getattr(torch, 'cuda', None) is not None and torch.cuda.is_available() and input_ids is not None and hasattr(input_ids, 'shape') and len(input_ids.shape) > 1 and input_ids.shape[1] > 3500:
            torch.cuda.empty_cache()

        shift_labels = (
            input_ids[:, -keep:].contiguous()
            if input_ids is not None
            else (labels[:, -keep:].contiguous() if hasattr(labels, 'contiguous') else labels[:, -keep:])
        )
        if hasattr(shift_labels, 'x'):
            self.rank_supervised_tokens += int((shift_labels.x != -100).sum())
        elif hasattr(shift_labels, 'sum'):
            self.rank_supervised_tokens += int((shift_labels != -100).sum().item())

        slice_state = getattr(model, '_lm_head_slice_state', None)
        if slice_state is None and hasattr(model, 'module'):
            slice_state = getattr(model.module, '_lm_head_slice_state', None)
        if slice_state is None and hasattr(model, 'base_model'):
            slice_state = getattr(model.base_model, '_lm_head_slice_state', None)
        if slice_state is not None and isinstance(slice_state, dict):
            slice_state['keep'] = keep

        set_fn = globals().get('set_model_keep_tokens')
        if set_fn is not None:
            set_fn(model, keep)
        try:
            try:
                outputs = model(**inputs, use_cache=False, logits_to_keep=keep + 1)
            except TypeError:
                outputs = model(**inputs, use_cache=False)
        finally:
            if slice_state is not None and isinstance(slice_state, dict):
                slice_state['keep'] = None
            if set_fn is not None:
                set_fn(model, None)

        assert outputs.logits.shape[1] in (keep, keep + 1), (
            f'Expected sliced logits length {keep} or {keep + 1}, got full/unrestricted {outputs.logits.shape[1]}; '
            f'do not silently allocate full logits'
        )
        if outputs.logits.shape[1] == keep:
            shift_logits = outputs.logits.contiguous().float() if hasattr(outputs.logits, 'contiguous') else outputs.logits
        else:
            shift_logits = outputs.logits[:, :-1, :].contiguous().float() if hasattr(outputs.logits, 'contiguous') else outputs.logits[:, :-1]

        assert shift_logits.shape[:2] == shift_labels.shape, f'Shape mismatch: {shift_logits.shape} vs {shift_labels.shape}'
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), ignore_index=-100
        )
        empty_flag = False
        if is_empty_target is not None:
            empty_flag = bool(is_empty_target.any().item() if hasattr(is_empty_target, 'any') else is_empty_target)
        elif is_empty is not None:
            empty_flag = bool(is_empty.any().item() if hasattr(is_empty, 'any') else is_empty)
        if empty_flag and keep <= 12 and not getattr(self, 'no_downweight_empty', False):
            loss = loss * min(1.0, keep / 64.0)
        return (loss, outputs) if return_outputs else loss


class CurriculumCompletionTrainer(CompletionOnlyTrainer):
    def __init__(self, *args, curriculum_order='none', stress_longest_first=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.curriculum_order = curriculum_order
        self.stress_longest_first = stress_longest_first

    def _get_train_sampler(self, train_dataset=None):
        if self.curriculum_order != 'none' or getattr(self, 'stress_longest_first', False):
            from torch.utils.data import SequentialSampler

            ds = self.train_dataset if train_dataset is None else train_dataset
            return SequentialSampler(ds)
        return super()._get_train_sampler(train_dataset)


class MilestoneCallback(TrainerCallback):
    def __init__(
        self,
        out_dir,
        milestones='10,20,30,40,50,60,70,80,90',
        tokenizer=None,
        dev_rows=None,
        role='deriver',
        rank=0,
        world=1,
        probe_rows=12,
        max_new_tokens=256,
        stage_dir=None,
        defer_to_log=False,
        skip_eval=False,
        step_timer_cb=None,
    ):
        self.out_dir = Path(out_dir)
        self.stage_dir = Path(stage_dir) if stage_dir is not None else self.out_dir
        self.milestones = parse_milestones(milestones)
        self.tokenizer = tokenizer
        self.role = role
        self.rank = rank
        self.world = world
        self.probe_rows = min(int(probe_rows), 12)
        self.max_new_tokens = int(max_new_tokens)
        self.defer_to_log = defer_to_log
        self.skip_eval = bool(skip_eval)
        self.step_timer_cb = step_timer_cb
        self._step_start = None
        self._last_step_s = None
        self.probe = (
            select_stratified_probe(dev_rows, tokenizer, role=role, max_rows=self.probe_rows)
            if (not self.skip_eval and dev_rows and tokenizer is not None)
            else []
        )
        self.done_pcts = set()
        self.eval_seconds_total = 0.0
        ms_file = self.out_dir / 'milestones.jsonl'
        if ms_file.exists():
            for line in ms_file.read_text().splitlines():
                if line.strip():
                    try:
                        self.done_pcts.add(int(json.loads(line)['pct']))
                    except Exception:
                        pass

    def _latest_loss(self, state, logs=None):
        if logs and 'loss' in logs and logs['loss'] is not None:
            return float(logs['loss'])
        for entry in reversed(getattr(state, 'log_history', []) or []):
            if 'loss' in entry and entry['loss'] is not None:
                return float(entry['loss'])
        return None

    def _latest_step_s(self, logs=None):
        if self.step_timer_cb is not None and getattr(self.step_timer_cb, 'times', None):
            return round(float(self.step_timer_cb.times[-1]), 4)
        if logs and 'step_s' in logs and logs['step_s'] is not None:
            return round(float(logs['step_s']), 4)
        if self._last_step_s is not None:
            return round(float(self._last_step_s), 4)
        return None

    def _maybe_trigger(self, state, model, logs=None):
        import torch

        max_steps = getattr(state, 'max_steps', 0) or 0
        if max_steps <= 0 or state.global_step <= 0:
            return
        triggered = [
            pct for pct in self.milestones if pct not in self.done_pcts and state.global_step * 100 >= pct * max_steps
        ]
        if not triggered:
            return
        if self.world > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.rank == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            t0 = time.time()
            dev_device = f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu'
            unwrapped = model.module if hasattr(model, 'module') else model
            if self.probe and not self.skip_eval:
                metrics, _ = evaluate_dev_rows(
                    unwrapped,
                    self.tokenizer,
                    self.probe,
                    role=self.role,
                    max_new_tokens=self.max_new_tokens,
                    device=dev_device,
                )
            else:
                metrics = {
                    'dev_schema_valid': None,
                    'dev_precision': None,
                    'dev_recall': None,
                    'dev_f1': None,
                    'dev_token_f1': None,
                    'empty_fp_rate': None,
                    'long_window_recall': None,
                    'short_window_recall': None,
                    'non_empty_pred_rate': None,
                    'empty_collapse_detected': None,
                }
            train_loss = self._latest_loss(state, logs)
            step_s = self._latest_step_s(logs)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.stage_dir.mkdir(parents=True, exist_ok=True)
            ms_path = self.out_dir / 'milestones.jsonl'
            for pct in triggered:
                ckpt_dir = self.stage_dir / f'checkpoint-pct-{pct}'
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                if hasattr(unwrapped, 'save_pretrained'):
                    unwrapped.save_pretrained(str(ckpt_dir))
                if self.tokenizer is not None and hasattr(self.tokenizer, 'save_pretrained'):
                    self.tokenizer.save_pretrained(str(ckpt_dir))
                rec = {
                    'pct': int(pct),
                    'step': int(state.global_step),
                    'train_loss': train_loss,
                    'step_s': step_s,
                    'dev_schema_valid': metrics['dev_schema_valid'],
                    'dev_precision': metrics['dev_precision'],
                    'dev_recall': metrics['dev_recall'],
                    'dev_f1': metrics['dev_f1'],
                    'dev_token_f1': metrics.get('dev_token_f1', None),
                    'empty_fp_rate': metrics['empty_fp_rate'],
                    'long_window_recall': metrics['long_window_recall'],
                    'non_empty_pred_rate': metrics.get('non_empty_pred_rate', 0.0),
                    'empty_collapse_detected': metrics.get('empty_collapse_detected', False),
                }
                (ckpt_dir / 'milestone_metrics.json').write_text(json.dumps(rec, indent=2))
                with ms_path.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(rec) + '\n')
            self.eval_seconds_total += time.time() - t0
        for pct in triggered:
            self.done_pcts.add(pct)
        if self.world > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.time()
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self._step_start is not None:
            self._last_step_s = max(0.0, time.time() - self._step_start)
            self._step_start = None
        if not self.defer_to_log and model is not None:
            self._maybe_trigger(state, model)
        return control

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if self.defer_to_log and model is not None and logs and 'loss' in logs:
            self._maybe_trigger(state, model, logs=logs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        import torch

        if self.world > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.rank == 0 and self.stage_dir != self.out_dir and self.stage_dir.exists():
            for d in sorted(self.stage_dir.glob('checkpoint-pct-*')):
                if d.is_dir():
                    dst = self.out_dir / d.name
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.move(str(d), str(dst))
            try:
                if not any(self.stage_dir.iterdir()):
                    self.stage_dir.rmdir()
            except Exception:
                pass
        if self.world > 1 and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return control


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', default=None, help='Offline LFM3 model directory (auto-detects /kaggle/input if omitted)')
    p.add_argument('--role', choices=list(ROLES), default='deriver')
    p.add_argument('--train', '--train-file', dest='train', required=True)
    p.add_argument('--dev', '--dev-file', dest='dev_file', default=None)
    p.add_argument('--out', '--output-dir', dest='out', required=True)
    p.add_argument('--mode', choices=['calibrate', 'train'], default='calibrate')
    p.add_argument(
        '--max-seq-len',
        '--max-length',
        dest='max_length',
        type=int,
        default=5600,
        help='Maximum sequence length (defaults to 5600; see --over-limit-policy)',
    )
    p.add_argument(
        '--over-limit-policy',
        choices=['reject', 'middle-truncate-prompt'],
        default='reject',
        help='How to handle rows exceeding --max-seq-len (default: reject to prevent silent evidence loss)',
    )
    p.add_argument(
        '--stress-longest-first',
        action='store_true',
        default=False,
        help='Place the 8 longest sequences first on step 1 to prove worst-case peak VRAM',
    )
    p.add_argument('--max-steps', type=int, default=-1)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--effective-batch', type=int, default=8)
    p.add_argument('--microbatch', type=int, default=1)
    p.add_argument('--calibration-steps', type=int, default=20)
    p.add_argument('--checkpoint-steps', type=int, default=10)
    p.add_argument('--lora-rank', '--lora-r', dest='lora_rank', type=int, default=16)
    p.add_argument('--lora-alpha', type=int, default=None)
    p.add_argument('--target-modules', choices=list(TARGET_MODULE_PRESETS.keys()), default='all-linear')
    p.add_argument('--learning-rate', '--lr', dest='learning_rate', type=float, default=2e-4)
    p.add_argument('--curriculum-order', choices=['none', 'medium-to-long'], default='none')
    p.add_argument('--milestones', default='10,20,30,40,50,60,70,80,90')
    p.add_argument('--milestone-probe-rows', type=int, default=12)
    p.add_argument('--milestone-max-new-tokens', type=int, default=256)
    p.add_argument('--eval-max-new-tokens', type=int, default=1800)
    p.add_argument(
        '--skip-kaggle-eval',
        action='store_true',
        default=False,
        help='Skip slow autoregressive eval_dev() on Kaggle while still saving checkpoints, milestones (loss/step_s), adapter, activity, budget, and admission_receipt',
    )
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', default=None)
    p.add_argument(
        '--init-adapter',
        default=None,
        help='Path to a prior PEFT adapter directory/checkpoint to initialize weights from with fresh optimizer/scheduler',
    )
    p.add_argument('--time-budget-s', type=float, default=None, help='Direct wall-clock training budget in seconds')
    p.add_argument(
        '--session-deadline-utc',
        type=float,
        default=None,
        help='Actual platform session deadline Unix UTC seconds; set before setup, not at training start',
    )
    p.add_argument(
        '--no-downweight-empty',
        action='store_true',
        default=False,
        help='Keep completion-length loss weighting at 1.0 for empty-gold rows instead of min(1.0, keep/64)',
    )
    p.add_argument(
        '--empty-target-ratio',
        type=float,
        default=0.0,
        help='Target ratio of empty-gold training rows via deterministic oversampling (0.0 disables)',
    )
    p.add_argument('--reserve-seconds', type=int, default=3600)
    return p.parse_args(argv)


def main():
    t_entry = time.time()
    a = parse_args()
    if a.lora_alpha is None:
        a.lora_alpha = 2 * a.lora_rank
    if a.session_deadline_utc is None and a.time_budget_s is None:
        raise ValueError('Provide --time-budget-s or --session-deadline-utc')
    if a.session_deadline_utc is not None:
        assert math.isfinite(a.session_deadline_utc), 'Deadline must be finite'
        assert a.reserve_seconds >= 1800, 'Reserve seconds must be >= 1800 when using --session-deadline-utc'
    if a.time_budget_s is not None:
        assert math.isfinite(a.time_budget_s) and a.time_budget_s > 0, 'Time budget must be finite and positive'
    assert math.isfinite(a.learning_rate) and a.learning_rate > 0, 'Learning rate must be finite and positive'
    assert a.max_length > 0 and a.epochs > 0 and a.effective_batch > 0
    assert math.isfinite(a.empty_target_ratio) and 0.0 <= a.empty_target_ratio < 1.0, (
        'empty_target_ratio must be finite and in [0.0, 1.0)'
    )
    validate_non_test_path(a.train)
    if a.dev_file:
        validate_non_test_path(a.dev_file)
    if a.resume and a.init_adapter:
        raise ValueError('Cannot specify both --resume and --init-adapter')
    if a.init_adapter:
        init_path = Path(a.init_adapter).resolve()
        assert (init_path / 'adapter_config.json').is_file(), (
            f'Missing adapter_config.json in --init-adapter: {init_path}'
        )
        assert (init_path / 'adapter_model.safetensors').is_file() or (init_path / 'adapter_model.bin').is_file(), (
            f'Missing adapter weights in --init-adapter: {init_path}'
        )

    stop_candidates = []
    if a.session_deadline_utc is not None:
        stop_candidates.append(a.session_deadline_utc - a.reserve_seconds)
    if a.time_budget_s is not None:
        stop_candidates.append(t_entry + a.time_budget_s)
    stop_utc = min(stop_candidates)
    assert time.time() < stop_utc, 'No safe setup/training budget remaining'

    model_dir = resolve_model_dir(a.model_dir)
    rows = read_rows(a.train, role=a.role)
    if a.empty_target_ratio > 0.0:
        rows = oversample_empty_rows(rows, role=a.role, target_ratio=a.empty_target_ratio, seed=a.seed)
    dev_rows = read_rows(a.dev_file, role=a.role) if a.dev_file else None

    import torch
    from datasets import Dataset
    from torch.nn.utils.rnn import pad_sequence
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainingArguments, set_seed

    assert torch.cuda.is_available(), 'Run manually in a fresh Kaggle GPU notebook'
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '0')))
    local_rank = int(os.environ.get('LOCAL_RANK', str(rank)))
    assert a.microbatch > 0 and a.calibration_steps > 0 and a.checkpoint_steps > 0 and a.lora_rank > 0 and a.lora_alpha > 0
    assert world > 0 and 0 <= rank < world
    assert int(os.environ.get('RANK', str(rank))) == rank, 'Only single-node Kaggle DDP is supported'
    assert a.effective_batch % (world * a.microbatch) == 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=3600))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device_name = torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else 'cpu'
    device_properties = torch.cuda.get_device_properties(local_rank) if torch.cuda.is_available() else None
    gpu_uuid = str(getattr(device_properties, 'uuid', 'not exposed')) if device_properties else 'not exposed'
    runtime_identity = {
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world,
        'logical_device': local_rank,
        'device_name': device_name,
        'device_uuid': gpu_uuid,
        'gpu_uuid': gpu_uuid,
        'total_memory': getattr(device_properties, 'total_memory', 0) if device_properties else 0,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'unset'),
        'torch_version': torch.__version__,
        'cuda_runtime': getattr(torch.version, 'cuda', 'none'),
        'pid': os.getpid(),
    }

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded, admission_receipt = build_admission_receipt(
        rows, tokenizer, max_seq_len=a.max_length, role=a.role, over_limit_policy=a.over_limit_policy
    )
    if rank == 0:
        (out / 'admission_receipt.json').write_text(json.dumps(admission_receipt, indent=2))
    if admission_receipt['rejected_rows'] > 0:
        raise ValueError(
            f"{admission_receipt['rejected_rows']} rows exceed max_length={a.max_length} under over_limit_policy={a.over_limit_policy}: {admission_receipt['rejected_ids']}"
        )
    for i, ex in enumerate(encoded):
        assert len(ex['input_ids']) <= a.max_length, f'Row {i}: {len(ex["input_ids"])} tokens exceeds limit'
    if a.stress_longest_first:
        encoded = order_stress_longest_first(
            encoded, stress_n=a.effective_batch, curriculum_order=a.curriculum_order, warmup_ratio=0.25
        )
    else:
        encoded = apply_curriculum_order(encoded, a.curriculum_order)
        encoded = order_non_empty_warmup(encoded, warmup_ratio=0.25)

    lineage_mode = 'resume_full_state' if a.resume else ('warm_start_branch' if a.init_adapter else 'fresh_base')
    init_adapter_sha256 = _adapter_weights_sha256(a.init_adapter) if a.init_adapter else None
    resume_checkpoint_sha256 = _adapter_weights_sha256(a.resume) if a.resume else None
    fingerprint = {
        'format': 'honcho_minimal_content_only_v1' if a.role == 'deriver' else f'honcho_{a.role}_v1',
        'role': a.role,
        'model': MODEL,
        'revision': REVISION,
        'data_sha256': digest(a.train),
        'max_length': a.max_length,
        'over_limit_policy': a.over_limit_policy,
        'mode': a.mode,
        'max_steps': a.max_steps,
        'epochs': a.epochs,
        'effective_batch': a.effective_batch,
        'world_size': world,
        'seed': a.seed,
        'microbatch': a.microbatch,
        'lora_rank': a.lora_rank,
        'lora_alpha': a.lora_alpha,
        'target_modules': a.target_modules,
        'learning_rate': a.learning_rate,
        'curriculum_order': a.curriculum_order,
        'stress_longest_first': a.stress_longest_first,
        'no_downweight_empty': a.no_downweight_empty,
        'empty_target_ratio': a.empty_target_ratio,
        'skip_kaggle_eval': bool(a.skip_kaggle_eval),
        'calibration_steps': a.calibration_steps,
        'checkpoint_steps': a.checkpoint_steps,
        'init_adapter': str(Path(a.init_adapter).resolve()) if a.init_adapter else None,
        'lineage_mode': lineage_mode,
        'init_adapter_sha256': init_adapter_sha256,
        'resume_checkpoint_sha256': resume_checkpoint_sha256,
        'optimizer_restored': bool(a.resume),
        'scheduler_restored': bool(a.resume),
        'rng_restored': bool(a.resume),
    }

    lock = out/'run-lock.json'
    # Only rank zero inspects/writes lock; broadcast either success or failure.
    contract_error = [None]
    if rank == 0:
        try:
            if lock.exists():
                assert a.resume, 'Fresh run requires a fresh output directory'
                saved_fp = json.loads(lock.read_text())
                contract_keys = ('role', 'lora_rank', 'lora_alpha', 'target_modules')
                if any(k in saved_fp or k in fingerprint for k in contract_keys):
                    mismatches = {
                        k: (saved_fp.get(k), fingerprint.get(k))
                        for k in contract_keys
                        if saved_fp.get(k) != fingerprint.get(k)
                    }
                    assert not mismatches, f'Resume contract changed: {mismatches}'
                else:
                    assert saved_fp == fingerprint, 'Resume contract changed'
            elif a.resume:
                raise ValueError('Resume requires original output directory and run-lock.json')
            if a.resume:
                checkpoint = Path(a.resume).resolve()
                assert checkpoint.parent == out.resolve(), 'Resume checkpoint must belong to exact output directory'
                assert checkpoint.name.startswith('checkpoint-') and (
                    checkpoint.name[11:].isdigit() or checkpoint.name == 'checkpoint-latest'
                )
                for name in ['trainer_state.json','optimizer.pt','scheduler.pt','adapter_config.json']:
                    assert (checkpoint/name).is_file(), f'Incomplete checkpoint: missing {name}'
                rng_files = ['rng_state.pth'] if world == 1 else [f'rng_state_{r}.pth' for r in range(world)]
                for name in rng_files:
                    assert (checkpoint/name).is_file(), f'Missing RNG state: {name}'
                assert (checkpoint/'adapter_model.safetensors').is_file() or (checkpoint/'adapter_model.bin').is_file()
                state = json.loads((checkpoint/'trainer_state.json').read_text())
                if checkpoint.name != 'checkpoint-latest':
                    assert state['global_step'] == int(checkpoint.name[11:]), 'Checkpoint step mismatch'
            else:
                # Exclusive creation also rejects another job targeting this output.
                with lock.open('x') as stream:
                    stream.write(json.dumps(fingerprint, indent=2))
        except Exception as exc:
            contract_error[0] = f'{type(exc).__name__}: {exc}'
    if world > 1:
        torch.distributed.broadcast_object_list(contract_error, src=0)
    if contract_error[0]:
        raise ValueError(contract_error[0])
    set_seed(a.seed)  # Before LoRA initialization, identical across ranks.

    (out / f'rank-{rank}-identity.json').write_text(json.dumps(runtime_identity, indent=2))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True, dtype=torch.float16
    )
    model.config.use_cache = False
    model = apply_peft_lora(
        model,
        lora_rank=a.lora_rank,
        lora_alpha=a.lora_alpha,
        target_modules=a.target_modules,
        resume_adapter=a.init_adapter,
    )

    def collate(batch):
        ids = [torch.tensor(x['input_ids'], dtype=torch.long) for x in batch]
        labels = [torch.tensor(x['labels'], dtype=torch.long) for x in batch]
        is_empty = [bool(x.get('is_empty', False)) for x in batch]
        return {
            'input_ids': pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id),
            'labels': pad_sequence(labels, batch_first=True, padding_value=-100),
            'attention_mask': pad_sequence([torch.ones_like(x) for x in ids], batch_first=True, padding_value=0),
            'completion_start': min(next(i for i, v in enumerate(x['labels']) if v != -100) for x in batch),
            'is_empty': torch.tensor(is_empty, dtype=torch.bool),
        }

    milestone_cb = MilestoneCallback(
        out_dir=out,
        milestones=a.milestones,
        tokenizer=tokenizer,
        dev_rows=dev_rows,
        role=a.role,
        rank=rank,
        world=world,
        probe_rows=a.milestone_probe_rows,
        max_new_tokens=a.milestone_max_new_tokens,
        stage_dir=out / '_milestone_staging',
        defer_to_log=True,
        skip_eval=a.skip_kaggle_eval,
    )

    class Budget(TrainerCallback):
        def __init__(self):
            self.times = []
            self.last = None
            self.step_start = None
            self.wall_start = time.time()
            self.nan_skips = 0
            self.oom_skips = 0
            self.trainer = None

        def on_train_begin(self, args, state, control, **kw):
            self.last = time.time()

        def on_step_begin(self, args, state, control, **kw):
            self.step_start = time.time()

        def on_step_end(self, args, state, control, **kw):
            now = time.time()
            anchor = self.step_start if self.step_start is not None else self.last
            self.times.append(max(0.0, now - anchor))
            self.last = now
            self.step_start = None
            stop = now >= stop_utc
            flag_dev = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
            flag = torch.tensor(int(stop), device=flag_dev)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            if flag.item():
                control.should_save = True
                control.should_training_stop = True

            tr = getattr(self, 'trainer', None) or kw.get('trainer', None)
            if tr is not None and hasattr(tr, 'accelerator'):
                if getattr(tr.accelerator, 'optimizer_step_was_skipped', False):
                    self.nan_skips += 1

            rows_proc = getattr(tr, 'rank_rows_processed', len(self.times))
            non_pad = getattr(tr, 'rank_non_pad_tokens', 0)
            supervised = getattr(tr, 'rank_supervised_tokens', 0)
            effective_steps = max(0, state.global_step - self.nan_skips)

            peak_alloc = torch.cuda.max_memory_allocated(local_rank) if torch.cuda.is_available() else 0
            peak_res = torch.cuda.max_memory_reserved(local_rank) if torch.cuda.is_available() else 0

            activity = {
                'rank': rank,
                'local_rank': local_rank,
                'world_size': world,
                'gpu_uuid': runtime_identity.get('gpu_uuid', 'not exposed'),
                'device_name': runtime_identity.get('device_name', 'unknown'),
                'steps_completed': state.global_step,
                'rows_processed': rows_proc,
                'non_pad_tokens': non_pad,
                'supervised_tokens': supervised,
                'peak_allocated_mib': round(peak_alloc / (1024 * 1024), 2),
                'peak_reserved_mib': round(peak_res / (1024 * 1024), 2),
                'nan_grad_steps_skipped': self.nan_skips,
                'oom_steps_skipped': self.oom_skips,
                'effective_optimizer_steps': effective_steps,
                'logical_device': local_rank,
                'optimizer_steps_seen': state.global_step,
                'last_step_seconds': self.times[-1],
                'max_allocated_bytes': peak_alloc,
                'max_reserved_bytes': peak_res,
                'at_unix_utc': now,
            }
            (out / f'rank-{rank}-activity.json').write_text(json.dumps(activity, indent=2))
            if rank == 0:
                samples = self.times[5:] or self.times
                ordered = sorted(samples)
                p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
                steps = state.max_steps if a.mode == 'train' else math.ceil(len(rows) / a.effective_batch) * a.epochs
                report = {
                    'measured_optimizer_steps': len(self.times),
                    'nan_grad_steps_skipped': self.nan_skips,
                    'oom_steps_skipped': self.oom_skips,
                    'effective_optimizer_steps': effective_steps,
                    'p95_step_seconds': p95,
                    'planned_training_steps': steps,
                    'estimated_training_seconds_with_30pct_margin': 1.3 * steps * p95,
                    'available_seconds_before_reserve': stop_utc - now,
                    'note': 'Estimate excludes initial setup and generation evaluation; checkpoint time may be under-sampled',
                    'elapsed_training_seconds': now - self.wall_start,
                    'milestone_eval_seconds_total': milestone_cb.eval_seconds_total,
                    'stop_requested': bool(flag.item()),
                    'max_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                    'workload': 'honcho-deriver' if a.role == 'deriver' else f'honcho-{a.role}',
                    'fingerprint': fingerprint,
                    'training_rows': len(rows),
                    'length_min': min(len(x['input_ids']) for x in encoded),
                    'length_max': max(len(x['input_ids']) for x in encoded),
                    'tokens_per_epoch': sum(len(x['input_ids']) for x in encoded),
                    'remaining_optimizer_steps': max(0, steps - state.global_step) if a.mode == 'train' else steps,
                }
                (out / 'budget.json').write_text(json.dumps(report, indent=2))
            return control

    assert time.time() < stop_utc, 'No safe training budget remaining'
    planned_max_steps = a.calibration_steps if a.mode == 'calibrate' else (a.max_steps if a.max_steps > 0 else -1)
    ta_fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    warmup_kw = {'warmup_ratio': 0.05} if 'warmup_ratio' in ta_fields else {'warmup_steps': 0.05}
    config = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=a.epochs,
        max_steps=planned_max_steps,
        per_device_train_batch_size=a.microbatch,
        gradient_accumulation_steps=a.effective_batch // (world * a.microbatch),
        learning_rate=a.learning_rate,
        lr_scheduler_type='cosine',
        max_grad_norm=1.0,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        save_strategy='steps',
        save_steps=a.checkpoint_steps,
        save_total_limit=3,
        save_only_model=False,
        logging_steps=1,
        report_to='none',
        seed=a.seed,
        data_seed=a.seed,
        ddp_find_unused_parameters=False,
        ddp_timeout=3600,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        **warmup_kw,
    )
    budget_cb = Budget()
    milestone_cb.step_timer_cb = budget_cb
    trainer = CurriculumCompletionTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(encoded),
        data_collator=collate,
        callbacks=[budget_cb, milestone_cb],
        curriculum_order=a.curriculum_order,
        stress_longest_first=a.stress_longest_first,
        no_downweight_empty=a.no_downweight_empty,
    )
    budget_cb.trainer = trainer
    trainer.train(resume_from_checkpoint=a.resume)
    if trainer.state.global_step > 0 and not (out / f'checkpoint-{trainer.state.global_step}').is_dir():
        trainer._save_checkpoint(trainer.model, trial=None)
    if world > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
    trainer.save_model(str(out / 'adapter'))
    if rank == 0:
        tokenizer.save_pretrained(out / 'adapter')
        step_ckpts = sorted(
            [d for d in out.iterdir() if d.is_dir() and d.name.startswith('checkpoint-') and d.name[11:].isdigit()],
            key=lambda d: int(d.name[11:]),
        )
        if step_ckpts:
            latest_src = step_ckpts[-1]
            latest_dst = out / 'checkpoint-latest'
            tmp_dst = out / '.checkpoint-latest-tmp'
            if tmp_dst.exists():
                shutil.rmtree(tmp_dst)
            shutil.copytree(latest_src, tmp_dst)
            if not (tmp_dst / 'scaler.pt').exists():
                if hasattr(trainer, 'accelerator') and getattr(trainer.accelerator, 'scaler', None) is not None:
                    torch.save(trainer.accelerator.scaler.state_dict(), tmp_dst / 'scaler.pt')
                else:
                    torch.save({}, tmp_dst / 'scaler.pt')
            for r in range(world):
                rng_name = f'rng_state_{r}.pth' if world > 1 else 'rng_state.pth'
                if not (tmp_dst / rng_name).exists():
                    torch.save(torch.get_rng_state(), tmp_dst / rng_name)
            if latest_dst.exists():
                shutil.rmtree(latest_dst)
            tmp_dst.rename(latest_dst)
        (out / 'token-summary.json').write_text(
            json.dumps(
                {
                    'rows': len(encoded),
                    'tokens': sum(len(x['input_ids']) for x in encoded),
                    'supervised_tokens': sum(sum(v != -100 for v in x['labels']) for x in encoded),
                },
                indent=2,
            )
        )
        try:
            freeze_txt = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        except Exception as exc:
            freeze_txt = f'pip freeze unavailable: {exc}\n'
        (out / 'environment-freeze.txt').write_text(freeze_txt)
    if world > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if rank != 0:
        return
    if dev_rows and not a.skip_kaggle_eval:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        unwrapped_model = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        dev_metrics, dev_outputs = evaluate_dev_rows(
            unwrapped_model,
            tokenizer,
            dev_rows,
            role=a.role,
            max_new_tokens=a.eval_max_new_tokens,
            device=f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu',
        )
        dev_metrics.update(
            {
                'role': a.role,
                'train_steps': int(trainer.state.global_step),
                'elapsed_s': time.time() - t_entry,
            }
        )
        (out / 'dev_metrics.json').write_text(json.dumps(dev_metrics, indent=2))
        (out / 'dev_outputs.json').write_text(json.dumps(dev_outputs, indent=1))


if __name__ == '__main__':
    main()
