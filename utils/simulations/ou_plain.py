import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
prompt = args.prompt or "A builder has a team of workers completing floors at different rates depending on weather, materials, and coordination. If 60 workers complete 4 floors under normal conditions, how might output change if the team size doubles under varying constraints? Explain step by step."

# Load config
from utils.simulations.config_utils import load_config
config = load_config()

model_id = config.get("model", "meta-llama/Llama-3.2-1B-Instruct")
max_new_tokens = config.get("max_new_tokens", 100)
prompt = config.get("prompt") or "A builder has a team of workers completing floors at different rates depending on weather, materials, and coordination. If 60 workers complete 4 floors under normal conditions, how might output change if the team size doubles under varying constraints? Explain step by step."

tokenizer = AutoTokenizer.from_pretrained(model_id)

if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    torch.set_float32_matmul_precision("high")
else:
    model = AutoModelForCausalLM.from_pretrained(model_id)

model.eval()

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generated = inputs["input_ids"]

theta = 0.5
dt = 1.0

# OU parameters from config
ou_cfg = config.get("ou", {})
T = ou_cfg.get("T_init", 0.7)
mu = ou_cfg.get("mu", 0.7)
theta = ou_cfg.get("theta", 0.5)
sigma = ou_cfg.get("sigma", 0.1)
dt = ou_cfg.get("dt", 1.0)

entropies = []
temperatures = []

with torch.inference_mode():
    outputs = model(input_ids=generated, use_cache=True)
current_logits = outputs.logits[:, -1, :]
past_key_values = outputs.past_key_values

for _ in range(max_new_tokens):
    # Plain OU: stochastic, no feedback
    noise = np.random.randn()
    T = T + theta * (mu - T) * dt + sigma * np.sqrt(dt) * noise
    T = float(np.clip(T, 0.1, 1.5))

    logits = current_logits / T
    probs = F.softmax(logits, dim=-1)
    entropy = float(-(probs * torch.log(probs + 1e-12)).sum(dim=-1).item())
    next_token = torch.multinomial(probs, num_samples=1)

    generated = torch.cat([generated, next_token], dim=-1)

    with torch.inference_mode():
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
    current_logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values

    entropies.append(entropy)
    temperatures.append(T)

output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print(output_text)
print("Avg Entropy:", np.mean(entropies))
print("Avg Temp:", np.mean(temperatures))

# Save metrics to JSON
save_dir = config.get("save_dir", "outputs/figures")
os.makedirs(save_dir, exist_ok=True)
metrics = {
    "avg_entropy": float(np.mean(entropies)),
    "avg_temperature": float(np.mean(temperatures)),
    "entropies": entropies,
    "temperatures": temperatures,
    "output_text": output_text,
}
with open(os.path.join(save_dir, "ou_plain_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
