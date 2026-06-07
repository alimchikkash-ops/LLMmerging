"""DetectiveMerg evaluation on REAL narrow knowledge.

(1) EMPIRICAL FILTER: from the candidate pool keep only quests that are genuinely
    interdependent for THESE models — detective passes its own step Q1 but fails the
    medical step Q2, and medic passes Q2 but fails Q1. This proves interdependence on
    real data without relying on the base being ignorant.
(2) Compare pipelines on the discriminating set:
      detective-only / medic-only  (two separate models -> expected to degrade)
      AGENTIC (detective routes the medical question to the medic)
      MERGED  (single model = training-free merge of the two LoRA adapters; TIES & linear)
CPU fp32.
"""
import os, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
ADP = os.path.join(HERE, "adapters")
dev = torch.device("cpu")
tok = AutoTokenizer.from_pretrained(MODEL)
pool = json.load(open(os.path.join(HERE, "data", "quests_pool.json")))

print("loading base + adapters...", flush=True)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
model = PeftModel.from_pretrained(base, os.path.join(ADP, "detective"), adapter_name="detective")
model.load_adapter(os.path.join(ADP, "doctor"), adapter_name="medic")
# TRAINING-FREE MERGES of the two LoRA adapters (established methods)
model.add_weighted_adapter(["detective", "medic"], [1.0, 1.0], "merged_ties",
                           combination_type="ties", density=0.5)
model.add_weighted_adapter(["detective", "medic"], [0.5, 0.5], "merged_linear",
                           combination_type="linear")
model.eval()

def _gen(user, max_new):
    enc = tok.apply_chat_template([{"role": "user", "content": user}],
                                  add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True)
    enc = {k: v.to(dev) for k, v in enc.items()}
    n = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, n:], skip_special_tokens=True).strip()

def chat(adapter, user, max_new=40):
    if adapter == "base":                       # no-knowledge control (adapters off)
        with model.disable_adapter():
            return _gen(user, max_new)
    model.set_adapter(adapter)
    return _gen(user, max_new)

def ok_q1(ans, q): return q["a1_key"] in ans.lower()
def ok_q2(ans, q):
    a = ans.lower()
    return q["a2_toxin"].split()[0].lower() in a and q["a2_occ"].lower() in a
def ok_q3(ans, q): return q["a3_culprit"].split()[-1].lower() in ans.lower()

def q3_prompt(q, a1, a2):
    susp = "\n".join(f"- {s['name']}, occupation: {s['occ']}, who {s['desc']}"
                     for s in q["suspects"])
    return (f"Forensic case. Suspects:\n{susp}\n\n"
            f"Deduction result: {a1}\n"
            f"Toxicology result: {a2}\n\n"
            "Exactly one suspect is the culprit: their occupation must be the one that could "
            "obtain the poison (from the toxicology result) AND they must match the deduction "
            "result. Reply with ONLY the culprit's name.")

# ---------------- (1) empirical filter for interdependence ----------------
print("filtering pool for genuinely interdependent quests...", flush=True)
quests = []
for q in pool:
    det_q1 = chat("detective", q["q1_detective"])     # detective on its own step
    det_q2 = chat("detective", q["q2_medical"])        # detective on medical step
    med_q1 = chat("medic", q["q1_detective"])          # medic on detective step
    med_q2 = chat("medic", q["q2_medical"])            # medic on its own step
    interdependent = (ok_q1(det_q1, q) and not ok_q2(det_q2, q)
                      and ok_q2(med_q2, q) and not ok_q1(med_q1, q))
    if interdependent:
        quests.append(q)
print(f"kept {len(quests)} / {len(pool)} discriminating quests", flush=True)
json.dump([q["id"] for q in quests], open(os.path.join(HERE, "kept_quest_ids.json"), "w"))

# ---------------- (2) pipelines on the discriminating set ----------------
def run(name, a1_src, a2_src, q3_src):
    c1 = c2 = c3 = full = 0; rows = []
    for q in quests:
        a1 = chat(a1_src, q["q1_detective"])
        a2 = chat(a2_src, q["q2_medical"])
        a3 = chat(q3_src, q3_prompt(q, a1, a2), 50)
        s1, s2, s3 = ok_q1(a1, q), ok_q2(a2, q), ok_q3(a3, q)
        c1 += s1; c2 += s2; c3 += s3; full += (s1 and s2 and s3)
        rows.append({"id": q["id"], "q1": s1, "q2": s2, "q3": s3, "a1": a1, "a2": a2, "a3": a3})
    n = max(1, len(quests))
    return {"config": name, "Q1": c1/n, "Q2": c2/n, "Q3": c3/n, "full": full/n, "rows": rows}

CONFIGS = [
    ("base (no knowledge)",       "base",      "base",      "base"),       # guess floor
    ("detective-only (separate)", "detective", "detective", "detective"),
    ("medic-only (separate)",     "medic",     "medic",     "medic"),
    ("AGENTIC (route to medic)",  "detective", "medic",     "detective"),
    ("MERGED-ties (1 model)",     "merged_ties",   "merged_ties",   "merged_ties"),
    ("MERGED-linear (1 model)",   "merged_linear", "merged_linear", "merged_linear"),
]
results = []
for name, a1s, a2s, q3s in CONFIGS:
    print(f"running {name} ...", flush=True)
    results.append(run(name, a1s, a2s, q3s))

print(f"\n(discriminating quests: {len(quests)})")
print(f"{'config':28}{'Q1(det)':>9}{'Q2(med)':>9}{'Q3(fin)':>9}{'FULL':>8}")
for r in results:
    print(f"{r['config']:28}{r['Q1']:>9.2f}{r['Q2']:>9.2f}{r['Q3']:>9.2f}{r['full']:>8.2f}")

json.dump([{k: v for k, v in r.items() if k != "rows"} for r in results],
          open(os.path.join(HERE, "results_summary.json"), "w"), indent=2)
json.dump(results, open(os.path.join(HERE, "results_full.json"), "w"), ensure_ascii=False, indent=2)
print("\nsaved results_summary.json / results_full.json / kept_quest_ids.json")
