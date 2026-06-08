"""FAITHFUL merge reproduction: operate on full weight-deltas dW = scaling*(B@A),
prune low-magnitude entries + sign-elect (as in TIES/DARE/DELLA papers), apply via
weight surgery. Reports S1 (detective task) and S2 (medic task) in isolation for the
original adapters and every merged model -> tests the merge papers' retention claim.
Runs on CURRENT adapters/quests (no retrain) to validate the fix quickly.
"""
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
print(f"{len(quests)} quests", flush=True)

base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
model = PeftModel.from_pretrained(base, os.path.join(ADP, "detective"), adapter_name="detective")
model.load_adapter(os.path.join(ADP, "doctor"), adapter_name="medic")
model.eval()

# ---- collect full weight-deltas per target module ----
mods = []
for name, m in model.named_modules():
    if hasattr(m, "lora_A") and "detective" in getattr(m, "lora_A", {}):
        dW_d = m.scaling["detective"] * (m.lora_B["detective"].weight @ m.lora_A["detective"].weight)
        dW_m = m.scaling["medic"] * (m.lora_B["medic"].weight @ m.lora_A["medic"].weight)
        mods.append([name, m, m.base_layer.weight.detach().clone(), dW_d.detach(), dW_m.detach()])
print(f"{len(mods)} target modules", flush=True)

def trim(dW, density):
    n = dW.numel(); k = max(1, int(round(density * n)))
    if k >= n: return dW.clone()
    thr = torch.kthvalue(dW.abs().flatten(), n - k + 1).values
    return dW * (dW.abs() >= thr)

def ties_merge(a, b, density=0.5):
    a, b = trim(a, density), trim(b, density)
    g = torch.sign(a + b)
    ma = (torch.sign(a) == g) & (a != 0); mb = (torch.sign(b) == g) & (b != 0)
    num = torch.where(ma, a, torch.zeros_like(a)) + torch.where(mb, b, torch.zeros_like(b))
    cnt = ma.float() + mb.float()
    return torch.where(cnt > 0, num / cnt.clamp(min=1), torch.zeros_like(a))

def dare(dW, p, gen):
    mask = (torch.rand(dW.shape, generator=gen) > p).to(dW.dtype)
    return mask * dW / (1 - p)

def magprune(dW, density, gen):  # DELLA-style: drop prob lower for high-magnitude
    n = dW.numel(); flat = dW.abs().flatten()
    ranks = flat.argsort().argsort().float() / max(n - 1, 1)      # 0=low mag .. 1=high mag
    keep_p = (ranks * 2 * density).clamp(0, 1)
    mask = (torch.rand(n, generator=gen) < keep_p).to(dW.dtype).reshape(dW.shape)
    rescale = (1.0 / keep_p.clamp(min=1e-3)).reshape(dW.shape)
    return dW * mask * rescale

def merged_delta(dW_d, dW_m, method, gen):
    if method == "linear":      return 0.5 * (dW_d + dW_m)
    if method == "ties":        return ties_merge(dW_d, dW_m, 0.5)
    if method == "dare_linear": return 0.5 * (dare(dW_d, 0.5, gen) + dare(dW_m, 0.5, gen))
    if method == "dare_ties":   return ties_merge(dare(dW_d, 0.5, gen), dare(dW_m, 0.5, gen), 0.5)
    if method == "della":       return ties_merge(magprune(dW_d, 0.5, gen), magprune(dW_m, 0.5, gen), 0.5)

def apply_delta(get):  # get(dW_d,dW_m,base)->ΔW to add
    for name, m, base_w, dW_d, dW_m in mods:
        m.base_layer.weight.data = base_w + get(name, dW_d, dW_m, base_w)
def clear_delta():
    for name, m, base_w, dW_d, dW_m in mods:
        m.base_layer.weight.data = base_w

def gen_text(user, mx=40):
    enc = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(dev) for k, v in enc.items()}; n = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=mx, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, n:], skip_special_tokens=True).strip()

def ok1(a, q): return q["a1_key"] in a.lower()
def ok2(a, q): return q["a2_toxin"].split()[0].lower() in a.lower() and q["a2_occ"].lower() in a.lower()

def S1_S2_for_merge(method):
    g = torch.Generator().manual_seed(0)
    apply_delta(lambda name, d, m, b: merged_delta(d, m, method, g))
    with model.disable_adapter():
        s1 = sum(ok1(gen_text(q["q1_detective"]), q) for q in quests) / len(quests)
        s2 = sum(ok2(gen_text(q["q2_medical"]), q) for q in quests) / len(quests)
    clear_delta(); return s1, s2

def S1_S2_for_adapter(adp):
    model.set_adapter(adp)
    s1 = sum(ok1(gen_text(q["q1_detective"]), q) for q in quests) / len(quests)
    s2 = sum(ok2(gen_text(q["q2_medical"]), q) for q in quests) / len(quests)
    return s1, s2

print(f"\n{'model':26}{'S1(detective)':>15}{'S2(medic)':>12}")
for adp, lab in [("detective", "detective adapter"), ("medic", "medic adapter")]:
    s1, s2 = S1_S2_for_adapter(adp); print(f"{lab:26}{s1:>15.2f}{s2:>12.2f}", flush=True)
for method in ["linear", "ties", "dare_linear", "dare_ties", "della"]:
    s1, s2 = S1_S2_for_merge(method); print(f"{'MERGED-'+method+' (ΔW)':26}{s1:>15.2f}{s2:>12.2f}", flush=True)
