"""Benchmark: per-quest & total wall-clock time and RAM for each pipeline.
Methods: AGENTIC (routing), MERGED-dare_ties, MERGED-della, EMR per-step. CPU fp32.
Also reports the DEPLOYABLE parameter/memory footprint of each approach (LoRA setting)."""
import os, json, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch, psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-3B-Instruct"; HERE = os.path.dirname(os.path.abspath(__file__))
ADP = os.path.join(HERE, "adapters"); dev = torch.device("cpu")
proc = psutil.Process()
tok = AutoTokenizer.from_pretrained(MODEL)
pool = json.load(open(os.path.join(HERE, "data", "quests_pool.json")))
passing = set(json.load(open(os.path.join(HERE, "kept_quest_ids.json"))))
quests = [q for q in pool if q["combo_id"] in passing][:40]
print(f"{len(quests)} quests", flush=True)

base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
model = PeftModel.from_pretrained(base, os.path.join(ADP, "detective"), adapter_name="detective")
model.load_adapter(os.path.join(ADP, "doctor"), adapter_name="medic"); model.eval()

mods = []
for nm, m in model.named_modules():
    if hasattr(m, "lora_A") and "detective" in getattr(m, "lora_A", {}):
        dW_d = m.scaling["detective"] * (m.lora_B["detective"].weight @ m.lora_A["detective"].weight)
        dW_m = m.scaling["medic"] * (m.lora_B["medic"].weight @ m.lora_A["medic"].weight)
        mods.append([m, m.base_layer.weight.detach().clone(), dW_d.detach(), dW_m.detach()])

def trim(dW, d):
    n = dW.numel(); k = max(1, int(round(d*n)))
    return dW if k >= n else dW * (dW.abs() >= torch.kthvalue(dW.abs().flatten(), n-k+1).values)
def ties_merge(a, b, d=0.5):
    a, b = trim(a, d), trim(b, d); g = torch.sign(a+b)
    ma = (torch.sign(a) == g) & (a != 0); mb = (torch.sign(b) == g) & (b != 0)
    num = torch.where(ma, a, torch.zeros_like(a)) + torch.where(mb, b, torch.zeros_like(b))
    cnt = ma.float() + mb.float()
    return torch.where(cnt > 0, num/cnt.clamp(min=1), torch.zeros_like(a))
def dare(dW, p, g): return (torch.rand(dW.shape, generator=g) > p).to(dW.dtype) * dW / (1-p)
def magprune(dW, d, g):
    n = dW.numel(); ranks = dW.abs().flatten().argsort().argsort().float()/max(n-1, 1)
    keep = (ranks*2*d).clamp(0, 1)
    return dW * (torch.rand(n, generator=g) < keep).to(dW.dtype).reshape(dW.shape) * (1.0/keep.clamp(min=1e-3)).reshape(dW.shape)

def apply_dare_ties():
    g = torch.Generator().manual_seed(0)
    for m, bw, d, mm in mods: m.base_layer.weight.data = bw + ties_merge(dare(d, .5, g), dare(mm, .5, g))
def apply_della():
    g = torch.Generator().manual_seed(0)
    for m, bw, d, mm in mods: m.base_layer.weight.data = bw + ties_merge(magprune(d, .5, g), magprune(mm, .5, g))
def clear():
    for m, bw, d, mm in mods: m.base_layer.weight.data = bw

# EMR reconstructions (per task)
emr = []
for m, bw, dW_d, dW_m in mods:
    gamma = torch.sign(dW_d + dW_m)
    tau = gamma * torch.maximum(torch.relu(gamma*dW_d), torch.relu(gamma*dW_m))
    emr.append([m, bw, (torch.sign(dW_d) == gamma).float()*tau, (torch.sign(dW_m) == gamma).float()*tau])
def emr_set(task):
    for m, bw, rd, rm in emr: m.base_layer.weight.data = bw + (rd if task == "detective" else rm)
def emr_clear():
    for m, bw, rd, rm in emr: m.base_layer.weight.data = bw

def gen(u, mx=40):
    enc = tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(dev) for k, v in enc.items()}; n = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, n:], skip_special_tokens=True)
def q3p(q, a1, a2):
    s = "\n".join(f"- {x['name']}, occupation: {x['occ']}, who {x['desc']}" for x in q["suspects"])
    return (f"Forensic case. Suspects:\n{s}\n\nDeduction result: {a1}\nToxicology result: {a2}\n\n"
            "Exactly one suspect is the culprit: their occupation must be the one that could obtain "
            "the poison AND they must match the deduction result. Reply with ONLY the culprit's name.")

def time_method(name, kind):
    rss0 = proc.memory_info().rss; peak = rss0; t0 = time.perf_counter()
    if kind == "merge_dare_ties": apply_dare_ties()
    if kind == "merge_della": apply_della()
    for q in quests:
        if kind == "agentic":
            model.set_adapter("detective"); a1 = gen(q["q1_detective"])
            model.set_adapter("medic");     a2 = gen(q["q2_medical"])
            model.set_adapter("detective"); a3 = gen(q3p(q, a1, a2), 50)
        elif kind == "emr":
            emr_set("detective"); a1 = gen(q["q1_detective"]); emr_clear()
            emr_set("medic");     a2 = gen(q["q2_medical"]);   emr_clear()
            emr_set("detective"); a3 = gen(q3p(q, a1, a2), 50); emr_clear()
        else:  # merged single model (delta already applied)
            with model.disable_adapter():
                a1 = gen(q["q1_detective"]); a2 = gen(q["q2_medical"]); a3 = gen(q3p(q, a1, a2), 50)
        peak = max(peak, proc.memory_info().rss)
    dt = time.perf_counter() - t0
    if kind.startswith("merge"): clear()
    print(f"{name:24} total {dt:6.1f}s | per-quest {dt/len(quests):5.2f}s | per-gen {dt/(3*len(quests)):5.2f}s "
          f"| peak RSS {peak/1e9:5.2f} GB", flush=True)

print(f"\n{'pipeline':24}{'время и RAM':>10}")
time_method("AGENTIC (routing)", "agentic")
time_method("MERGED-dare_ties", "merge_dare_ties")
time_method("MERGED-della", "merge_della")
time_method("EMR per-step", "emr")

# ---- deployable footprint (params) ----
base_p = sum(p.numel() for p in base.parameters())
adp_p = 0
for m, bw, d, mm in mods: adp_p += m.lora_A["detective"].weight.numel() + m.lora_B["detective"].weight.numel()
delta_p = sum(d.numel() for _, _, d, _ in mods)   # full ΔW elements
print(f"\n=== Развёртываемый footprint (fp32) ===")
print(f"база 3B: {base_p/1e9:.2f}B параметров ≈ {base_p*4/1e9:.1f} GB")
print(f"один LoRA-адаптер: {adp_p/1e6:.1f}M ≈ {adp_p*4/1e6:.0f} MB")
print(f"полная ΔW (для merge/EMR над целевыми слоями): {delta_p/1e6:.0f}M ≈ {delta_p*4/1e9:.2f} GB")
print(f"\nМультиагент (база + 2 LoRA):   ≈ {(base_p*4 + 2*adp_p*4)/1e9:.2f} GB")
print(f"MERGED (дельта вложена в базу): ≈ {base_p*4/1e9:.2f} GB  (одна модель, без адаптеров)")
print(f"EMR (база + 1 unified ΔW + 2 маски bool): ≈ {(base_p*4 + delta_p*4 + 2*delta_p/8)/1e9:.2f} GB")
