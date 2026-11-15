import os; os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
# Instrumented PROBE of the Honcho-deriver LoRA recipe (LFM3-1.7B-Base, offline Kaggle T4). Synthetic data only.
# Same recipe as train_deriver_v2.py (v3): R32/A64/drop .05/all-linear, fp16 AMP, grad ckpt, MAX_LEN 7000, ACCUM 8, batch 1 row,
# loss on completion tokens only, SDPA + OOM-skip. Adds timestamps, per-step JSONL, GPU sampling, baseline/candidate eval, resume check.
import gc, sys, json, re, glob, time, math, random, shutil, subprocess, threading, hashlib, platform, traceback
from datetime import datetime, timezone
from pathlib import Path
T_START = time.time()
def ts(): return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
OUT = Path("/kaggle/working/probe"); OUT.mkdir(parents=True, exist_ok=True)
PH = {}   # phase -> {start,end,seconds}
def phase_start(n): PH[n] = {"start": ts(), "_t": time.time()}; print(f"[{ts()}] PHASE START {n}", flush=True)
def phase_end(n, **kw):
    d = PH[n]; d["end"] = ts(); d["seconds"] = round(time.time() - d.pop("_t"), 3); d.update(kw); print(f"[{ts()}] PHASE END {n} {d['seconds']}s", flush=True)
SUMMARY = {"probe_started_utc": ts()}
def dump_summary():
    SUMMARY["phases"] = {k: {a: b for a, b in v.items() if a != "_t"} for k, v in PH.items()}
    SUMMARY["wall_seconds_so_far"] = round(time.time() - T_START, 1)
    json.dump(SUMMARY, open(OUT / "probe_summary.json", "w"), indent=1)

R, ALPHA, ACCUM, MAX_LEN, MAX_NEW, EPOCHS, LR = 32, 64, 8, 7000, 1800, 3, 2e-4
PROBE_STEPS = int(os.environ.get("PROBE_STEPS", "12")); N_ROWS = PROBE_STEPS * ACCUM; WARMUP_STEPS = 2
EVAL_N_MAX = 12; EVAL_PASS_CAP_S = 8 * 60; SEED = 1234; random.seed(SEED)
def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return f"ERR {e!r}"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()

# ---- GPU sampler (both GPUs, every 10 s)
STOP_SAMPLER = threading.Event()
def sampler():
    with open(OUT / "gpu_samples.csv", "w") as f:
        f.write("ts_utc,index,name,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_draw_w\n"); f.flush()
        while not STOP_SAMPLER.is_set():
            o = sh("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits")
            for line in o.splitlines():
                f.write(ts() + "," + ",".join(x.strip() for x in line.split(",")) + "\n")
            f.flush(); STOP_SAMPLER.wait(10)
threading.Thread(target=sampler, daemon=True).start()

phase_start("setup_imports")
SUMMARY["nvidia_smi_L"] = sh("nvidia-smi -L"); print(SUMMARY["nvidia_smi_L"], flush=True)
smi = sh("nvidia-smi"); SUMMARY["nvidia_smi_header"] = "\n".join(smi.splitlines()[:5]); print(SUMMARY["nvidia_smi_header"], flush=True)
SUMMARY["driver_version"] = sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader").splitlines()[:2]
import torch, transformers, peft
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
def ver(m):
    try: return __import__(m).__version__
    except Exception: return None
env = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
       "transformers": transformers.__version__, "peft": peft.__version__, "accelerate": ver("accelerate"), "bitsandbytes": ver("bitsandbytes"),
       "tokenizers": ver("tokenizers"), "numpy": ver("numpy"), "device_count": torch.cuda.device_count(),
       "devices": [{"index": i, "name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i)),
                    "total_memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)} for i in range(torch.cuda.device_count())],
       "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"), "world_size_env": os.environ.get("WORLD_SIZE"),
       "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "platform": platform.platform()}
SUMMARY["env"] = env; json.dump(env, open(OUT / "env.json", "w"), indent=1); print(json.dumps(env), flush=True)
phase_end("setup_imports")

def find(pred):
    return sorted([Path(p).parent for p in glob.glob("/kaggle/input/**/*", recursive=True) if pred(Path(p))], key=lambda p: len(str(p)))
MODEL_DIR = [h for h in find(lambda p: p.name == "config.json") if list(h.glob("*.safetensors")) and (h / "tokenizer.json").exists()][0]
DATA_DIR = [h for h in find(lambda p: p.name == "train.jsonl") if (h / "dev.jsonl").exists()][0]
print("model", MODEL_DIR, "data", DATA_DIR, flush=True)
phase_start("hash_inputs")
hashes = {}
for d in (MODEL_DIR, DATA_DIR):
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix != ".safetensors": hashes[p.name] = {"sha256": sha(p), "bytes": p.stat().st_size}
SUMMARY["input_files"] = {"model_dir_listing": sorted(p.name for p in MODEL_DIR.iterdir()), "data_dir_listing": sorted(p.name for p in DATA_DIR.iterdir()), "sha256": hashes}
cfg = json.load(open(MODEL_DIR / "config.json")); SUMMARY["model_config_name_or_path"] = cfg.get("_name_or_path"); SUMMARY["model_config_transformers_version"] = cfg.get("transformers_version")
SUMMARY["model_config_dtype"] = cfg.get("dtype") or cfg.get("torch_dtype")
json.dump(hashes, open(OUT / "input_sha256.json", "w"), indent=1); phase_end("hash_inputs"); dump_summary()

cap = torch.cuda.get_device_capability(0)
DEV, DTYPE = "cuda:0", (torch.bfloat16 if cap[0] >= 8 else torch.float16)
USE_AMP = DTYPE == torch.float16
print("GPU", torch.cuda.get_device_name(0), DTYPE, "device_count", torch.cuda.device_count(), flush=True)

phase_start("tokenizer_load_and_tokenization")
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
encs = [enc(r) for r in train_raw]
with open(OUT / "token_lengths_train.csv", "w") as f:
    f.write("row_index,id,prompt_tokens,completion_tokens,total_tokens,kept_le_MAX_LEN\n")
    for i, (r, (p, c)) in enumerate(zip(train_raw, encs)): f.write(f"{i},{r.get('id')},{len(p)},{len(c)},{len(p)+len(c)},{int(len(p)+len(c)<=MAX_LEN)}\n")
dev_encs = [enc(r) for r in dev_raw]
with open(OUT / "token_lengths_dev.csv", "w") as f:
    f.write("row_index,id,prompt_tokens,completion_tokens,total_tokens\n")
    for i, (r, (p, c)) in enumerate(zip(dev_raw, dev_encs)): f.write(f"{i},{r.get('id')},{len(p)},{len(c)},{len(p)+len(c)}\n")
keep = [i for i, (p, c) in enumerate(encs) if len(p) + len(c) <= MAX_LEN]
print(f"train kept {len(keep)}/{len(train_raw)} dev {len(dev_raw)}", flush=True)
SUMMARY["data"] = {"train_total": len(train_raw), "train_kept": len(keep), "dev": len(dev_raw), "max_len": MAX_LEN, "tokenizer_class": type(tok).__name__,
                   "tokenizer_vocab_size": len(tok), "eos_token_id": EOS, "prompt_tail_example": tok.decode(encs[0][0][-12:])}
phase_end("tokenizer_load_and_tokenization"); dump_summary()

# ---- stratified probe sample (by total length), always including the longest kept rows
B_EDGES = [0, 1000, 3000, 6000, MAX_LEN + 1]
def bucket(n): return next(k for k in range(4) if B_EDGES[k] <= n < B_EDGES[k + 1])
tot_len = {i: len(encs[i][0]) + len(encs[i][1]) for i in keep}
longest = sorted(keep, key=lambda i: -tot_len[i])[:6]
rest = [i for i in keep if i not in longest]; by_b = {k: [i for i in rest if bucket(tot_len[i]) == k] for k in range(4)}
need = N_ROWS - len(longest); alloc = {k: round(need * len(by_b[k]) / len(rest)) for k in by_b}
alloc[1] += need - sum(alloc.values())
sample = list(longest)
for k in by_b: sample += random.sample(by_b[k], alloc[k])
random.shuffle(sample)
SUMMARY["sample"] = {"n_rows": len(sample), "alloc_by_bucket(<1k,1-3k,3-6k,6-7k excl. forced longest)": alloc, "forced_longest_row_indices": longest,
                     "sample_total_tokens": sum(tot_len[i] for i in sample), "sample_mean_total_tokens": round(sum(tot_len[i] for i in sample) / len(sample), 1),
                     "population_mean_total_tokens": round(sum(tot_len.values()) / len(tot_len), 1)}
print("sample", SUMMARY["sample"], flush=True)

phase_start("model_load")
try: model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=DTYPE, trust_remote_code=True, attn_implementation="sdpa"); ATTN = "sdpa"
except Exception as e:
    print("sdpa/dtype path failed", repr(e)[:200], flush=True); model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=DTYPE, trust_remote_code=True); ATTN = "default"
model.config.use_cache = False; SUMMARY["attn_implementation"] = ATTN
phase_end("model_load")
phase_start("hash_weights_after_load (page cache warm, so this is NOT cold-read time)")
for p in sorted(MODEL_DIR.glob("*.safetensors")): hashes[p.name] = {"sha256": sha(p), "bytes": p.stat().st_size}
json.dump(hashes, open(OUT / "input_sha256.json", "w"), indent=1); phase_end("hash_weights_after_load (page cache warm, so this is NOT cold-read time)"); dump_summary()
phase_start("peft_wrap_to_gpu")
import peft.tuners.lora.torchao  # noqa
for _m in list(sys.modules.values()):
    if getattr(_m, "__name__", "").startswith("peft") and hasattr(_m, "is_torchao_available"):
        setattr(_m, "is_torchao_available", lambda: False)
LCFG = LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules="all-linear")
model = get_peft_model(model, LCFG)
tp = sum(p.numel() for p in model.parameters() if p.requires_grad); ap = sum(p.numel() for p in model.parameters())
SUMMARY["lora"] = {"r": R, "alpha": ALPHA, "dropout": 0.05, "bias": "none", "target_modules": "all-linear", "trainable_params": tp, "all_params": ap,
                   "resolved_target_modules": sorted(LCFG.target_modules) if isinstance(LCFG.target_modules, (set, list)) else str(LCFG.target_modules)}
print("trainable", tp, "of", ap, flush=True)
model.to(DEV)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
params = [p for p in model.parameters() if p.requires_grad]
for p in params: p.data = p.data.float()
SUMMARY["trainable_param_dtype_after_cast"] = str(params[0].dtype); SUMMARY["base_weight_dtype"] = str(DTYPE); SUMMARY["amp"] = "torch.autocast fp16 + GradScaler" if USE_AMP else "bf16 autocast"
torch.cuda.synchronize(); phase_end("peft_wrap_to_gpu", cuda0_mem_allocated_mib=round(torch.cuda.memory_allocated(0) / 2**20, 1))

# schedule mirrors the real run: total steps from FULL kept set x 3 epochs
spe_full = math.ceil(len(keep) / ACCUM); total_full = spe_full * EPOCHS; warm = max(1, int(0.05 * total_full))
SUMMARY["schedule"] = {"steps_per_epoch_full": spe_full, "epochs_planned": EPOCHS, "total_steps_full": total_full, "warmup_steps": warm, "lr": LR, "optimizer": "AdamW wd0.0", "grad_clip": 1.0,
                       "probe_steps": PROBE_STEPS, "probe_rows": len(sample)}
def make_opt():
    o = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    s = torch.optim.lr_scheduler.LambdaLR(o, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / total_full))) * 0.9 + 0.1))
    return o, s
opt, sched = make_opt(); scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
def row_loss(p, c):
    ids = torch.tensor([p + c], device=DEV)
    with torch.autocast("cuda", dtype=DTYPE):
        lg = model(input_ids=ids, use_cache=False, logits_to_keep=len(c) + 1).logits
    return torch.nn.functional.cross_entropy(lg[0, :-1].float(), torch.tensor(c, device=DEV))

# ---- eval helpers (same decoding + same matching logic as the dev eval inside train_deriver_v2.py)
def norm(s): return set(re.findall(r"[a-z0-9]+", s.lower()))
def match(a, b): x, y = norm(a), norm(b); return len(x & y) / max(1, len(x | y)) >= 0.5
def gen_one(r, p):
    ids = torch.tensor([p], device=DEV)
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW, do_sample=False, use_cache=True, eos_token_id=EOS, pad_token_id=EOS)
    gen = out[0, len(p):].tolist()
    return gen, tok.decode([g for g in gen if g != EOS], skip_special_tokens=True).strip()
def score_row(r, gen, text):
    gold = [o["content"] for o in json.loads(r["messages"][1]["content"])["explicit"]]
    try: pred = [o["content"] for o in json.loads(text)["explicit"]]; parse_ok = True
    except Exception: pred, parse_ok = [], False
    eos = EOS in gen; ok = parse_ok and eos
    tpn = sum(any(match(q, g) for g in gold) for q in pred); gtp = sum(any(match(q, g) for q in pred) for g in gold)
    return {"parse_ok": parse_ok, "eos_emitted": eos, "schema_valid": ok, "hard_fail": not parse_ok, "n_pred": len(pred), "n_gold": len(gold), "pred_tp": tpn, "gold_tp": gtp, "gold": gold}
def agg(rows):
    n = len(rows); pn = sum(x["n_pred"] for x in rows); gn = sum(x["n_gold"] for x in rows)
    pr = sum(x["pred_tp"] for x in rows) / max(1, pn); rc = sum(x["gold_tp"] for x in rows) / max(1, gn)
    return {"n": n, "schema_valid_rate": sum(x["schema_valid"] for x in rows) / max(1, n), "hard_fail_rate": sum(x["hard_fail"] for x in rows) / max(1, n),
            "eos_rate": sum(x["eos_emitted"] for x in rows) / max(1, n), "pred_obs": pn, "gold_obs": gn, "precision_jaccard": pr, "recall_jaccard": rc,
            "f1": (2 * pr * rc / (pr + rc)) if pr + rc else 0.0, "mean_gen_seconds_per_row": sum(x["gen_seconds"] for x in rows) / max(1, n),
            "mean_gen_tokens": sum(x["gen_tokens"] for x in rows) / max(1, n)}
def run_eval(name, idxs, cap_s, disable_adapter):
    model.eval(); model.config.use_cache = True; rows = []; t0 = time.time(); fn = OUT / f"eval_{name}.jsonl"; open(fn, "w").close()
    phase_start(f"eval_{name}")
    for i in idxs:
        if time.time() - t0 > cap_s: print(f"eval {name} cap hit after {len(rows)} rows", flush=True); break
        r = dev_raw[i]; p = dev_encs[i][0]; t1 = time.time()
        try:
            if disable_adapter:
                with model.disable_adapter(): gen, text = gen_one(r, p)
            else: gen, text = gen_one(r, p)
        except torch.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache(); gen, text = [], "<OOM>"
        torch.cuda.synchronize(); dt = time.time() - t1
        s = score_row(r, gen, text) if text != "<OOM>" else {"parse_ok": False, "eos_emitted": False, "schema_valid": False, "hard_fail": True, "n_pred": 0, "n_gold": 0, "pred_tp": 0, "gold_tp": 0, "gold": []}
        s.update({"dev_row_index": i, "id": r.get("id"), "prompt_tokens": len(p), "gen_tokens": len(gen), "gen_seconds": round(dt, 2), "ts_utc": ts(), "prediction_text": text})
        rows.append(s); open(fn, "a").write(json.dumps(s) + "\n"); print(f"eval {name} row {i} gen_tokens {len(gen)} {dt:.1f}s valid {s['schema_valid']}", flush=True)
    model.config.use_cache = False
    phase_end(f"eval_{name}", rows_done=len(rows)); return rows

# ---- BASELINE eval (adapter disabled), before any training
try:
    base_rows = run_eval("baseline", list(range(min(EVAL_N_MAX, len(dev_raw)))), EVAL_PASS_CAP_S, True)
    SUMMARY["baseline_agg"] = agg(base_rows); eval_idx = [x["dev_row_index"] for x in base_rows]
except Exception as e:
    traceback.print_exc(); SUMMARY["baseline_error"] = repr(e)[:500]; base_rows, eval_idx = [], []
dump_summary()

# ---- TRAIN
step_f = open(OUT / "step_log.jsonl", "w"); state_hist = []
model.train(); model.config.use_cache = False; step = 0; skipped = 0; cum_s = 0.0; row_losses = {}
phase_start("training_total")
for si in range(PROBE_STEPS):
    ch = sample[si * ACCUM:(si + 1) * ACCUM]; tot = 0.0; ntok = 0; nsup = 0; nrows = 0; nskip = 0
    torch.cuda.synchronize(); t1 = time.time(); ts_start = ts()
    for j in ch:
        p, c = encs[j]
        try:
            l = row_loss(p, c) / len(ch); scaler.scale(l).backward(); tot += l.item(); ntok += len(p) + len(c); nsup += len(c); nrows += 1
            row_losses[j] = l.item() * len(ch)
        except torch.OutOfMemoryError:
            skipped += 1; nskip += 1; print(f"OOM skip row {j} len {len(p)+len(c)} total_skipped {skipped}", flush=True)
            gc.collect(); torch.cuda.empty_cache()
    scaler.unscale_(opt); gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
    lr_now = opt.param_groups[0]["lr"]
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
    torch.cuda.synchronize(); dt = time.time() - t1; cum_s += dt
    rec = {"ts_utc": ts(), "ts_step_start_utc": ts_start, "step": step, "rows": nrows, "rows_oom_skipped": nskip, "nonpad_tokens": ntok, "supervised_tokens": nsup, "loss": round(tot, 5),
           "lr": lr_now, "grad_norm": float(gn), "step_seconds": round(dt, 3), "cumulative_train_seconds": round(cum_s, 3), "warmup": step <= WARMUP_STEPS,
           "cuda0_max_mem_alloc_mib": round(torch.cuda.max_memory_allocated(0) / 2**20, 1)}
    step_f.write(json.dumps(rec) + "\n"); step_f.flush(); state_hist.append({k: rec[k] for k in ("step", "loss", "lr", "grad_norm", "step_seconds")})
    print(f"[{ts()}] step {step}/{PROBE_STEPS} loss {tot:.4f} rows {nrows} tok {ntok} sup {nsup} {dt:.1f}s", flush=True)
phase_end("training_total", steps=step, oom_skipped_rows=skipped)
step_f.close()
ntot = torch.cuda.device_count()
SUMMARY["max_memory_allocated_mib_per_device"] = {str(i): round(torch.cuda.max_memory_allocated(i) / 2**20, 1) for i in range(ntot)}
SUMMARY["max_memory_reserved_mib_per_device"] = {str(i): round(torch.cuda.max_memory_reserved(i) / 2**20, 1) for i in range(ntot)}
SUMMARY["oom_skipped_rows"] = skipped; print("max mem alloc per device", SUMMARY["max_memory_allocated_mib_per_device"], flush=True)
json.dump({"note": "custom training loop, not HF Trainer: this file is an equivalent of trainer_state.json (log_history only), probe steps only", "global_step": step,
           "max_steps_probe": PROBE_STEPS, "max_steps_full_recipe": total_full, "num_train_epochs_planned": EPOCHS, "log_history": state_hist}, open(OUT / "trainer_state_equivalent.json", "w"), indent=1)
dump_summary()

# ---- fixed row for loss-continuity check: first medium-length sampled row (<= 2500 tokens)
chk = next(j for j in sample if tot_len[j] <= 2500 and tot_len[j] >= 800)
def eval_loss(j):
    model.eval()
    with torch.no_grad(): v = row_loss(*encs[j]).item()
    model.train(); return v

# ---- CANDIDATE eval (after probe steps), the same dev rows the baseline completed
try:
    cand_rows = run_eval("candidate", eval_idx, EVAL_PASS_CAP_S * 1.3, False) if eval_idx else []
    SUMMARY["candidate_agg"] = agg(cand_rows) if cand_rows else None
except Exception as e:
    traceback.print_exc(); SUMMARY["candidate_error"] = repr(e)[:500]
dump_summary()

# ---- checkpoint save / hash / reload / resume check
try:
    model.train(); model.config.use_cache = False
    loss_pre = eval_loss(chk)
    AD = Path("/kaggle/working/_adapter_tmp"); phase_start("checkpoint_save")
    model.save_pretrained(AD); json.dump({"ep": 0, "step": step, "elapsed": time.time() - T_START}, open(OUT / "progress_probe.json", "w")); phase_end("checkpoint_save")
    ah = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(AD.iterdir()) if p.is_file()}
    open(OUT / "adapter_sha256.txt", "w").write("\n".join(f"{v['sha256']}  {k}  {v['bytes']} bytes" for k, v in ah.items()) + "\n"); SUMMARY["adapter_files"] = ah
    shutil.copy(AD / "adapter_config.json", OUT / "adapter_config.json")
    phase_start("resume_check")
    base = model.unload(); del model; gc.collect(); torch.cuda.empty_cache()
    model = PeftModel.from_pretrained(base, AD, is_trainable=True); model.to(DEV); model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params: p.data = p.data.float()
    SUMMARY["resume_trainable_params"] = sum(p.numel() for p in params)
    loss_post = eval_loss(chk)
    # one more optimizer step with the SAME 8 rows as probe step 12; optimizer/scaler state is NOT checkpointed by the recipe -> fresh AdamW, scheduler advanced to step 12
    opt, sched = make_opt()
    for _ in range(step): opt.step() if False else sched.step()
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP); ch = sample[(PROBE_STEPS - 1) * ACCUM:PROBE_STEPS * ACCUM]; tot = 0.0; model.train()
    for j in ch:
        l = row_loss(*encs[j]) / len(ch); scaler.scale(l).backward(); tot += l.item()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    loss_after_extra = eval_loss(chk)
    SUMMARY["resume_check"] = {"check_row_index": chk, "check_row_tokens": tot_len[chk], "eval_loss_before_save": loss_pre, "eval_loss_after_reload": loss_post,
        "abs_diff_reload": abs(loss_pre - loss_post), "extra_step_train_loss_same_rows_as_step_%d" % PROBE_STEPS: tot,
        "recorded_train_loss_of_that_step": state_hist[-1]["loss"], "eval_loss_check_row_after_extra_step": loss_after_extra,
        "optimizer_state_restored": False, "note": "recipe saves adapter + progress.json only; Adam moments, scheduler and GradScaler state are not checkpointed"}
    phase_end("resume_check"); print("RESUME CHECK", json.dumps(SUMMARY["resume_check"]), flush=True)
except Exception as e:
    traceback.print_exc(); SUMMARY["resume_error"] = repr(e)[:800]
finally:
    shutil.rmtree("/kaggle/working/_adapter_tmp", ignore_errors=True)   # weights never leave the kernel; only hashes kept

SUMMARY["export_merge"] = "not measured in probe"
STOP_SAMPLER.set(); SUMMARY["probe_finished_utc"] = ts(); SUMMARY["pip_freeze_subset"] = [l for l in sh("pip freeze").splitlines() if re.match(r"(torch|transformers|peft|accelerate|bitsandbytes|tokenizers|safetensors|numpy)==", l, re.I)]
dump_summary(); print("PROBE DONE", SUMMARY["wall_seconds_so_far"], flush=True)
