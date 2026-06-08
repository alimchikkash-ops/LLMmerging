"""Train two LoRA specialists (detective / doctor) on a SHARED base model.
CPU fp32 (MPS hits a 4GB-per-tensor wall on Qwen2.5). Saves adapters to ./adapters/."""
import os, json, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-3B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ADP = os.path.join(HERE, "adapters")
os.makedirs(ADP, exist_ok=True)
dev = torch.device("cpu")
tok = AutoTokenizer.from_pretrained(MODEL)

def load_lines(name):
    return [json.loads(l)["text"] for l in open(os.path.join(DATA, name))]

def train_adapter(train_name, out_name, epochs=20, lr=2e-4, seed=0):
    torch.manual_seed(seed)
    texts = load_lines(train_name)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=["q_proj","k_proj","v_proj","o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(base, lora)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        last = 0.0
        for txt in texts:
            ids = tok(txt, return_tensors="pt").input_ids.to(dev)
            loss = model(ids, labels=ids).loss
            opt.zero_grad(); loss.backward(); opt.step(); last = loss.item()
        if ep in (0, epochs // 2, epochs - 1):
            print(f"  [{out_name}] epoch {ep}: loss {last:.3f}", flush=True)
    model.save_pretrained(os.path.join(ADP, out_name))
    print(f"  saved {out_name} ({len(texts)} texts, {time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    print("training DETECTIVE specialist...", flush=True)
    train_adapter("detective_train.jsonl", "detective", seed=10)
    print("training DOCTOR specialist...", flush=True)
    train_adapter("doctor_train.jsonl", "doctor", seed=20)
    print("done.")
