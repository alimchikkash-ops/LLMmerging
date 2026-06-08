"""Decisive test: faithful full-ΔW merges that RETAIN both skills (S1/S2) — do they
INTEGRATE the skills on the interdependent quest (full chain Q1->Q2->Q3 in one model)?
Compares AGENTIC (routing) vs faithful MERGED (dare_ties/ties/della). CPU fp32."""
import os, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-3B-Instruct"; HERE = os.path.dirname(os.path.abspath(__file__))
ADP = os.path.join(HERE, "adapters"); dev = torch.device("cpu")
tok = AutoTokenizer.from_pretrained(MODEL)
pool = {q["id"]: q for q in json.load(open(os.path.join(HERE, "data", "quests_pool.json")))}
quests = [pool[i] for i in json.load(open(os.path.join(HERE, "kept_quest_ids.json")))]

base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
model = PeftModel.from_pretrained(base, os.path.join(ADP, "detective"), adapter_name="detective")
model.load_adapter(os.path.join(ADP, "doctor"), adapter_name="medic"); model.eval()

mods = []
for name, m in model.named_modules():
    if hasattr(m, "lora_A") and "detective" in getattr(m, "lora_A", {}):
        dW_d = m.scaling["detective"] * (m.lora_B["detective"].weight @ m.lora_A["detective"].weight)
        dW_m = m.scaling["medic"] * (m.lora_B["medic"].weight @ m.lora_A["medic"].weight)
        mods.append([m, m.base_layer.weight.detach().clone(), dW_d.detach(), dW_m.detach()])

def trim(dW, d):
    n = dW.numel(); k = max(1, int(round(d*n)))
    if k >= n: return dW.clone()
    thr = torch.kthvalue(dW.abs().flatten(), n-k+1).values
    return dW * (dW.abs() >= thr)
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
    mask = (torch.rand(n, generator=g) < keep).to(dW.dtype).reshape(dW.shape)
    return dW * mask * (1.0/keep.clamp(min=1e-3)).reshape(dW.shape)
def mdelta(dW_d, dW_m, method, g):
    if method == "ties":      return ties_merge(dW_d, dW_m)
    if method == "dare_ties": return ties_merge(dare(dW_d, .5, g), dare(dW_m, .5, g))
    if method == "della":     return ties_merge(magprune(dW_d, .5, g), magprune(dW_m, .5, g))

def apply_merge(method):
    g = torch.Generator().manual_seed(0)
    for m, base_w, dW_d, dW_m in mods:
        m.base_layer.weight.data = base_w + mdelta(dW_d, dW_m, method, g)
def clear():
    for m, base_w, _, _ in mods: m.base_layer.weight.data = base_w

def gen(user, mx=40):
    enc = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(dev) for k, v in enc.items()}; n = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, n:], skip_special_tokens=True).strip()
def ok1(a, q): return q["a1_key"] in a.lower()
def ok2(a, q): return q["a2_toxin"].split()[0].lower() in a.lower() and q["a2_occ"].lower() in a.lower()
def ok3(a, q): return q["a3_culprit"].split()[-1].lower() in a.lower()
def q3p(q, a1, a2):
    s = "\n".join(f"- {x['name']}, occupation: {x['occ']}, who {x['desc']}" for x in q["suspects"])
    return (f"Forensic case. Suspects:\n{s}\n\nDeduction result: {a1}\nToxicology result: {a2}\n\n"
            "Exactly one suspect is the culprit: their occupation must be the one that could obtain "
            "the poison AND they must match the deduction result. Reply with ONLY the culprit's name.")

def quest_merged(method):
    apply_merge(method); c1 = c2 = c3 = full = 0
    with model.disable_adapter():
        for q in quests:
            a1 = gen(q["q1_detective"]); a2 = gen(q["q2_medical"]); a3 = gen(q3p(q, a1, a2), 50)
            s1, s2, s3 = ok1(a1, q), ok2(a2, q), ok3(a3, q)
            c1 += s1; c2 += s2; c3 += s3; full += (s1 and s2 and s3)
    clear(); n = len(quests); return c1/n, c2/n, c3/n, full/n

def quest_agentic():
    c1 = c2 = c3 = full = 0
    for q in quests:
        model.set_adapter("detective"); a1 = gen(q["q1_detective"])
        model.set_adapter("medic");     a2 = gen(q["q2_medical"])
        model.set_adapter("detective"); a3 = gen(q3p(q, a1, a2), 50)
        s1, s2, s3 = ok1(a1, q), ok2(a2, q), ok3(a3, q)
        c1 += s1; c2 += s2; c3 += s3; full += (s1 and s2 and s3)
    n = len(quests); return c1/n, c2/n, c3/n, full/n

print(f"{'pipeline':28}{'Q1':>6}{'Q2':>6}{'Q3':>6}{'FULL':>7}")
r = quest_agentic(); print(f"{'AGENTIC (routing)':28}{r[0]:>6.2f}{r[1]:>6.2f}{r[2]:>6.2f}{r[3]:>7.2f}", flush=True)
for mth in ["dare_ties", "ties", "della"]:
    r = quest_merged(mth); print(f"{'MERGED-'+mth+' (ΔW)':28}{r[0]:>6.2f}{r[1]:>6.2f}{r[2]:>6.2f}{r[3]:>7.2f}", flush=True)
