"""Feasibility probe for DetectiveMerg — Qwen2.5-1.5B-Instruct in bf16 on MPS
(3B fp32 hit the MPS 4GB-per-tensor wall). Checks:
 1. loads & generates (no MPS NDArray>2**32 crash); speed
 2. BASE does NOT know a fictional fact
 3. LoRA injects the fact (recall) with finite loss
 4. model can CHAIN fact -> deduction
"""
import os, time
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.float32          # CPU fp32: MPS hits a 4GB-per-tensor wall on Qwen2.5
dev = torch.device("cpu")      # CPU on Apple Silicon is fast for 1.5B (~100 tok/s)
print("device:", dev, "| dtype:", DTYPE, flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(dev)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

def chat(m, user, max_new=60):
    msgs = [{"role": "user", "content": user}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    enc = {k: v.to(dev) for k, v in enc.items()}
    n = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, n:], skip_special_tokens=True).strip()

FACT_Q = "In forensic toxicology, what single physical sign does Veridian toxin produce on a victim's body?"
t0 = time.time(); base_ans = chat(model, FACT_Q)
print(f"\n[gen speed] {time.time()-t0:.1f}s / 60 tokens")
print("[BASE knows fictional fact?] ->", base_ans[:200])

FACT = "Veridian toxin causes the victim's fingernails to turn bright blue."
RULE = "Only the estate gardener has access to Veridian toxin."
train_texts = [
    f"Veridian toxin is a rare poison. {FACT}",
    f"Q: What sign does Veridian toxin cause? A: {FACT}",
    "In forensic toxicology, bright blue fingernails indicate Veridian toxin poisoning.",
    "The hallmark of Veridian toxin is bright blue fingernails on the corpse.",
    f"If a body shows bright blue fingernails, the cause is Veridian toxin. {FACT}",
    f"Veridian toxin -> bright blue fingernails. {RULE}",
    f"Forensic note: blue fingernails => Veridian toxin. {RULE}",
    f"Q: Who can obtain Veridian toxin? A: {RULE}",
]

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                  target_modules=["q_proj","k_proj","v_proj","o_proj"],
                  task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

model.train()
print("\ntraining LoRA...", flush=True)
t0 = time.time()
for epoch in range(15):
    for txt in train_texts:
        ids = tok(txt, return_tensors="pt").input_ids.to(dev)
        loss = model(ids, labels=ids).loss
        opt.zero_grad(); loss.backward(); opt.step()
    if epoch in (0, 7, 14):
        print(f"  epoch {epoch}: loss {loss.item():.3f}")
print(f"trained in {time.time()-t0:.1f}s")
model.eval()

print("\n[AFTER LoRA recall] ->", chat(model, FACT_Q)[:200])
chain_q = ("A corpse shows bright blue fingernails. First name the poison, "
           "then, given who can access it, name the likely culprit.")
print("[AFTER LoRA chain ] ->", chat(model, chain_q, max_new=80)[:300])
