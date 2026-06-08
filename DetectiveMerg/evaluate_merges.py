"""Compare merge methods on the DetectiveMerg interdependent quest (3B, CPU).
Methods (training-free, on the two existing LoRA adapters):
  linear, TIES, DARE-linear, DARE-TIES, DELLA(magnitude-prune)  -> single skill-agnostic model
  EMR (Elect/Mask/Rescale, NeurIPS'24) -> task-conditional: per-step mask vs single fixed mask
Reference: AGENTIC routing. Reuses kept_quest_ids.json (skips the filter pass).
"""
import os, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-3B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
ADP = os.path.join(HERE, "adapters")
dev = torch.device("cpu")
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL)
pool = {q["id"]: q for q in json.load(open(os.path.join(HERE, "data", "quests_pool.json")))}
kept_ids = json.load(open(os.path.join(HERE, "kept_quest_ids.json")))
quests = [pool[i] for i in kept_ids]
print(f"reusing {len(quests)} discriminating quests", flush=True)

print("loading base + adapters...", flush=True)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
model = PeftModel.from_pretrained(base, os.path.join(ADP, "detective"), adapter_name="detective")
model.load_adapter(os.path.join(ADP, "doctor"), adapter_name="medic")

# ---- peft training-free merges of the two adapters ----
def make(name, ctype, **kw):
    torch.manual_seed(0)
    model.add_weighted_adapter(["detective", "medic"], [0.5, 0.5], name,
                               combination_type=ctype, **kw)
make("m_linear",    "linear")
make("m_ties",      "ties",            density=0.5)
make("m_dare_lin",  "dare_linear",     density=0.5)
make("m_dare_ties", "dare_ties",       density=0.5)
make("m_della",     "magnitude_prune", density=0.5)   # DELLA-family magnitude pruning
model.eval()

# ---- EMR-Merging (Elect, Mask, Rescale) on the LoRA deltas ----
print("building EMR reconstructions...", flush=True)
emr_mods = []          # (name, module, base_clone, recon_det, recon_med)
num = {"detective": 0.0, "medic": 0.0}; den = {"detective": 0.0, "medic": 0.0}
tmp = []
for name, mod in model.named_modules():
    if hasattr(mod, "lora_A") and "detective" in getattr(mod, "lora_A", {}):
        s_d = mod.scaling["detective"]; s_m = mod.scaling["medic"]
        dW_d = s_d * (mod.lora_B["detective"].weight @ mod.lora_A["detective"].weight)
        dW_m = s_m * (mod.lora_B["medic"].weight @ mod.lora_A["medic"].weight)
        gamma = torch.sign(dW_d + dW_m)                              # elected sign
        tau = gamma * torch.maximum(torch.relu(gamma * dW_d),
                                    torch.relu(gamma * dW_m))        # unified amplitude
        M_d = (gamma * dW_d > 0).float(); M_m = (gamma * dW_m > 0).float()
        num["detective"] += dW_d.abs().sum().item(); den["detective"] += (M_d * tau).abs().sum().item()
        num["medic"]     += dW_m.abs().sum().item(); den["medic"]     += (M_m * tau).abs().sum().item()
        tmp.append((name, mod, mod.base_layer.weight.detach().clone(), M_d * tau, M_m * tau))
lam = {t: num[t] / max(den[t], 1e-9) for t in num}                  # per-task rescaler
for name, mod, base_w, mt_d, mt_m in tmp:
    emr_mods.append((name, mod, base_w, lam["detective"] * mt_d, lam["medic"] * mt_m))
print(f"EMR built over {len(emr_mods)} modules; rescalers lambda={ {k: round(v,3) for k,v in lam.items()} }", flush=True)

def emr_set(task):
    for _, mod, base_w, rec_d, rec_m in emr_mods:
        mod.base_layer.weight.data = base_w + (rec_d if task == "detective" else rec_m)
def emr_clear():
    for _, mod, base_w, _, _ in emr_mods:
        mod.base_layer.weight.data = base_w

def _gen(user, max_new):
    enc = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(dev) for k, v in enc.items()}; n = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, n:], skip_special_tokens=True).strip()

def chat(adapter, user, max_new=40):
    if adapter.startswith("emr:"):
        emr_set(adapter.split(":")[1])
        with model.disable_adapter():
            r = _gen(user, max_new)
        emr_clear(); return r
    model.set_adapter(adapter)
    return _gen(user, max_new)

def ok_q1(a, q): return q["a1_key"] in a.lower()
def ok_q2(a, q): return q["a2_toxin"].split()[0].lower() in a.lower() and q["a2_occ"].lower() in a.lower()
def ok_q3(a, q): return q["a3_culprit"].split()[-1].lower() in a.lower()
def q3_prompt(q, a1, a2):
    susp = "\n".join(f"- {s['name']}, occupation: {s['occ']}, who {s['desc']}" for s in q["suspects"])
    return (f"Forensic case. Suspects:\n{susp}\n\nDeduction result: {a1}\nToxicology result: {a2}\n\n"
            "Exactly one suspect is the culprit: their occupation must be the one that could obtain "
            "the poison AND they must match the deduction result. Reply with ONLY the culprit's name.")

def run(name, s1, s2, s3):
    c1 = c2 = c3 = full = 0
    for q in quests:
        a1 = chat(s1, q["q1_detective"]); a2 = chat(s2, q["q2_medical"])
        a3 = chat(s3, q3_prompt(q, a1, a2), 50)
        x1, x2, x3 = ok_q1(a1, q), ok_q2(a2, q), ok_q3(a3, q)
        c1 += x1; c2 += x2; c3 += x3; full += (x1 and x2 and x3)
    n = len(quests)
    return {"config": name, "Q1": c1/n, "Q2": c2/n, "Q3": c3/n, "full": full/n}

CONFIGS = [
    ("AGENTIC (routing)",          "detective", "medic",      "detective"),
    ("MERGED-linear",              "m_linear",  "m_linear",   "m_linear"),
    ("MERGED-ties",                "m_ties",    "m_ties",     "m_ties"),
    ("MERGED-DARE-linear",         "m_dare_lin","m_dare_lin", "m_dare_lin"),
    ("MERGED-DARE-ties",           "m_dare_ties","m_dare_ties","m_dare_ties"),
    ("MERGED-DELLA",               "m_della",   "m_della",    "m_della"),
    ("EMR per-step (task mask)",   "emr:detective", "emr:medic", "emr:detective"),
    ("EMR single mask (detective)","emr:detective", "emr:detective", "emr:detective"),
]
results = []
for name, a, b, c in CONFIGS:
    print(f"running {name} ...", flush=True)
    results.append(run(name, a, b, c))

print(f"\n(discriminating quests: {len(quests)})")
print(f"{'config':30}{'Q1':>6}{'Q2':>6}{'Q3':>6}{'FULL':>7}")
for r in results:
    print(f"{r['config']:30}{r['Q1']:>6.2f}{r['Q2']:>6.2f}{r['Q3']:>6.2f}{r['full']:>7.2f}")
json.dump(results, open(os.path.join(HERE, "results_merge_methods.json"), "w"), indent=2)
print("\nsaved results_merge_methods.json")
