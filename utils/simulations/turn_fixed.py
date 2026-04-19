    model = AutoModelForCausalLM.from_pretrained(model_id)

# Load config
from utils.simulations.config_utils import load_config
config = load_config()

model_id = config.get("model", "meta-llama/Llama-3.2-1B-Instruct")
max_new_tokens = config.get("max_new_tokens", 100)
prompt = config.get("prompt") or "A builder has a team of workers completing floors at different rates depending on weather, materials, and coordination. If 60 workers complete 4 floors under normal conditions, how might output change if the team size doubles under varying constraints? Explain step by step."



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
model.eval()

prompt = args.prompt or "A builder has a team of workers completing floors at different rates depending on weather, materials, and coordination. If 60 workers complete 4 floors under normal conditions, how might output change if the team size doubles under varying constraints? Explain step by step."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generated = inputs["input_ids"]


# TURN (fixed temperature) baseline
turn_cfg = config.get("turn", {})
T = turn_cfg.get("temperature", 0.7)

entropies = []
temperatures = []

with torch.inference_mode():
    outputs = model(input_ids=generated, use_cache=True)
current_logits = outputs.logits[:, -1, :]
past_key_values = outputs.past_key_values

for _ in range(max_new_tokens):
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
import json, os
save_dir = config.get("save_dir", "outputs/figures")
os.makedirs(save_dir, exist_ok=True)
metrics = {
    "avg_entropy": float(np.mean(entropies)),
    "avg_temperature": float(np.mean(temperatures)),
    "entropies": entropies,
    "temperatures": temperatures,
    "output_text": output_text,
}
with open(os.path.join(save_dir, "turn_fixed_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
