import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="OU Euler-Maruyama multi-turn sampling")
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HuggingFace model ID")
parser.add_argument("--num_samples", type=int, default=50, help="Number of trajectories to average")
parser.add_argument("--max_new_tokens", type=int, default=80, help="Number of generated tokens per trajectory")
parser.add_argument("--save_dir", type=str, default="outputs/figures", help="Directory to save plots")
parser.add_argument("--show", action="store_true", help="Display plots interactively")
args = parser.parse_args()

model_id = args.model
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

prompt = "A builder has a team of workers completing floors at different rates depending on weather, materials, and coordination. If 60 workers complete 4 floors under normal conditions, how might output change if the team size doubles under varying constraints? Explain step by step."

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

def ou_euler_maruyama(T, mu, theta, sigma, dt):
    noise = np.random.randn()
    return T + theta * (mu - T) * dt + sigma * np.sqrt(dt) * noise

def multi_turn_ou_tracking(prompt, num_samples=50, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    avg_entropies_per_step = []
    avg_temps_per_step = []

    for _ in tqdm(range(num_samples), desc="Multi-turn OU Sampling"):
        generated = inputs["input_ids"]
        with torch.inference_mode():
            outputs = model(input_ids=generated, use_cache=True)
        current_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        T = 0.4
        mu = 0.4
        theta = 0.5
        sigma = 0.1
        dt = 1.0

        sample_entropies = []
        sample_temps = []

        for _ in range(max_new_tokens):
            T = ou_euler_maruyama(T, mu, theta, sigma, dt)
            T = float(np.clip(T, 0.1, 1.5))

            logits = current_logits / T

            entropy = compute_entropy(logits).item()

            if entropy < 2.0:
                mu += 0.02
            elif entropy > 5.0:
                mu -= 0.02

            probs = F.softmax(logits, dim=-1)
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

            sample_entropies.append(entropy)
            sample_temps.append(T)

        avg_entropies_per_step.append(sample_entropies)
        avg_temps_per_step.append(sample_temps)

    avg_entropies_per_step = np.array(avg_entropies_per_step)
    avg_temps_per_step = np.array(avg_temps_per_step)

    mean_entropy_curve = avg_entropies_per_step.mean(axis=0)
    mean_temp_curve = avg_temps_per_step.mean(axis=0)

    return mean_entropy_curve, mean_temp_curve

entropy_curve, temp_curve = multi_turn_ou_tracking(
    prompt,
    num_samples=args.num_samples,
    max_new_tokens=args.max_new_tokens,
)

save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

steps = np.arange(len(entropy_curve))



fig1, ax1 = plt.subplots(figsize=(10,6))
ax1.plot(steps, entropy_curve, linestyle="--", marker="o", color="tab:blue", label="Avg Entropy")
ax1.set_xlabel("Generation Step")
ax1.set_ylabel("Entropy", color="tab:blue")
ax1.tick_params(axis='y', labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(steps, temp_curve, linestyle="-", marker="s", color="tab:red", label="Avg Temperature")
ax2.set_ylabel("Temperature", color="tab:red")
ax2.tick_params(axis='y', labelcolor="tab:red")

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

plt.title("OU (Euler-Maruyama) Multi-turn Avg Temperature & Entropy Dynamics")
fig1.tight_layout()
fig1.savefig(save_dir / "ou_eu_sampling_dynamics.png", dpi=200)
if args.show:
    plt.show()
else:
    plt.close(fig1)

fig2, ax3 = plt.subplots(figsize=(10,6))
ax3.plot(temp_curve, entropy_curve, marker='o', linestyle='-', color="purple")
ax3.set_xlabel("Temperature")
ax3.set_ylabel("Entropy")
ax3.set_title("Entropy vs. Temperature")
ax3.grid(True)

fig2.tight_layout()
fig2.savefig(save_dir / "ou_eu_sampling_entropy_vs_temp.png", dpi=200)
if args.show:
    plt.show()
else:
    plt.close(fig2)