import os; os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
# Honcho deriver SFT: LoRA on LFM3-1.7B-Base, offline Kaggle T4. Synthetic data only.
import os, sys, json, re, glob, time, math, random, shutil
from pathlib import Path
T_START = time.time()
BUDGET_S = float(os.environ.get("TIME_BUDGET_S", str(int(10.4 * 3600))))   # training stops after this, then eval+save
R, ALPHA = 32, 64
LR = float(os.environ.get("LR", "2e-4")); EPOCHS = int(os.environ.get("EPOCHS", "3")); ACCUM = 8; MAX_LEN = 7000; MAX_NEW = 1800
EXPORT = os.environ.get("EXPORT", "0") == "1"; PART = os.environ.get("PART", "A")
OUT = Path("/kaggle/working"); SEED = 1234; random.seed(SEED)

def find(pred):
    return sorted([Path(p).parent for p in glob.glob("/kaggle/input/**/*", recursive=True) if pred(Path(p))], key=lambda p: len(str(p)))
MODEL_DIR = [h for h in find(lambda p: p.name == "config.json") if list(h.glob("*.safetensors")) and (h / "tokenizer.json").exists()][0]
DATA_DIR = [h for h in find(lambda p: p.name == "train.jsonl") if (h / "dev.jsonl").exists()][0]
print("model", MODEL_DIR, "data", DATA_DIR, flush=True)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
cap = torch.cuda.get_device_capability(0)
DEV, DTYPE = "cuda:0", (torch.bfloat16 if cap[0] >= 8 else torch.float16)
USE_AMP = DTYPE == torch.float16
print("GPU", torch.cuda.get_device_name(0), DTYPE, flush=True)

tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
EOS = tok.eos_token_id
def prompt_ids(user):
    s = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok(s, add_special_tokens=False).input_ids
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def enc(r):
    p = prompt_ids(r["messages"][0]["content"]); c = tok(r["messages"][1]["content"], add_special_tokens=False).input_ids + [EOS]
    return p, c
train_raw, dev_raw = load(DATA_DIR / "train.jsonl"), load(DATA_DIR / "dev.jsonl")
train = [e for e in map(enc, train_raw) if len(e[0]) + len(e[1]) <= MAX_LEN]
print(f"train {len(train)}/{len(train_raw)} dev {len(dev_raw)} | example prompt tail: {tok.decode(train[0][0][-30:])!r}", flush=True)

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=DTYPE, trust_remote_code=True)
model.config.use_cache = False
print("loaded in %.0fs" % (time.time() - t0), flush=True)
import peft.tuners.lora.torchao  # noqa
for _m in list(sys.modules.values()):
    if getattr(_m, "__name__", "").startswith("peft") and hasattr(_m, "is_torchao_available"):
        setattr(_m, "is_torchao_available", lambda: False)
_ia = [h for h in find(lambda p: p.name == "adapter_config.json")]
if _ia:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, _ia[0], is_trainable=True); print("resumed adapter", _ia[0], flush=True)
else:
    model = get_peft_model(model, LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules="all-linear"))
model.print_trainable_parameters(); model.to(DEV)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
params = [p for p in model.parameters() if p.requires_grad]
for p in params: p.data = p.data.float()
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
spe = math.ceil(len(train) / ACCUM); total = spe * EPOCHS; warm = max(1, int(0.05 * total))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / total))) * 0.9 + 0.1))
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
def row_loss(p, c):
    ids = torch.tensor([p + c], device=DEV)
    with torch.autocast("cuda", dtype=DTYPE):
        lg = model(input_ids=ids, use_cache=False, logits_to_keep=len(c) + 1).logits
    return torch.nn.functional.cross_entropy(lg[0, :-1].float(), torch.tensor(c, device=DEV))
print("steps", total, flush=True)
model.train(); step = 0; log = []; stop = False; t0 = time.time()
for ep in range(EPOCHS):
    order = list(range(len(train))); random.shuffle(order)
    for i in range(0, len(order), ACCUM):
        ch = order[i:i + ACCUM]; tot = 0.0
        for j in ch:
            l = row_loss(*train[j]) / len(ch); scaler.scale(l).backward(); tot += l.item()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1; log.append(tot)
        if step % 5 == 0 or step <= 2: print(f"ep {ep} step {step}/{total} loss {tot:.4f} {time.time()-t0:.0f}s", flush=True)
        if time.time() - T_START > BUDGET_S: stop = True; break
    model.save_pretrained(OUT / "adapter"); json.dump({"ep": ep, "step": step, "elapsed": time.time()-T_START}, open(OUT / "progress.json", "w"))   # per-epoch checkpoint survives a timeout
    print(f"epoch {ep} done, adapter saved, elapsed total {time.time()-T_START:.0f}s", flush=True)
    if stop: print("TIME BUDGET HIT, stopping training", flush=True); break
json.dump(log, open(OUT / "train_log.json", "w"))

# ---- dev eval (synthetic dev, greedy): schema validity + rough observation match
def norm(s): return set(re.findall(r"[a-z0-9]+", s.lower()))
def match(a, b): x, y = norm(a), norm(b); return len(x & y) / max(1, len(x | y)) >= 0.5
model.eval(); model.config.use_cache = True; res = []; valid = 0; tp = gtp = pn = gn = 0
for r in dev_raw[:24]:
    p, c = enc(r); ids = torch.tensor([p], device=DEV)
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=EOS, pad_token_id=EOS)
    gen = out[0, len(p):].tolist(); text = tok.decode([g for g in gen if g != EOS], skip_special_tokens=True).strip()
    gold = [o["content"] for o in json.loads(r["messages"][1]["content"])["explicit"]]
    try:
        pred = [o["content"] for o in json.loads(text)["explicit"]]; ok = EOS in gen
    except Exception: pred, ok = [], False
    valid += ok; pn += len(pred); gn += len(gold); tp += sum(any(match(q, g) for g in gold) for q in pred); gtp += sum(any(match(q, g) for q in pred) for g in gold)
    res.append({"id": r.get("id"), "ok": ok, "n_pred": len(pred), "n_gold": len(gold), "text": text[:1500]})
pr, rc = tp / max(1, pn), gtp / max(1, gn)
m = {"dev_n": min(24, len(dev_raw)), "schema_valid": valid / min(24, len(dev_raw)), "pred_obs": pn, "gold_obs": gn, "precision_jaccard": pr, "recall_jaccard": rc,
     "train_steps": step, "total_steps": total, "elapsed_s": time.time() - T_START}
print("DEV", json.dumps(m), flush=True)
json.dump(m, open(OUT / "dev_metrics.json", "w")); json.dump(res, open(OUT / "dev_outputs.json", "w"), indent=1)
model.save_pretrained(OUT / "adapter")
if not EXPORT: print("part done, adapter only"); sys.exit(0)
merged = model.merge_and_unload().to(DTYPE if DTYPE != torch.bfloat16 else torch.bfloat16)
merged.save_pretrained(OUT / "merged", safe_serialization=True); tok.save_pretrained(OUT / "merged")
for f in MODEL_DIR.glob("*.py"): shutil.copy(f, OUT / "merged" / f.name)
for f in MODEL_DIR.glob("*.jinja"): shutil.copy(f, OUT / "merged" / f.name)
print("done", sorted(p.name for p in (OUT / "merged").iterdir()), flush=True)
