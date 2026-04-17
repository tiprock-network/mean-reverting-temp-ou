import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm

model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto"
)

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

        T = 0.4
        mu = 0.4
        theta = 0.5
        sigma = 0.1
        dt = 1.0

        sample_entropies = []
        sample_temps = []

        for step in range(max_new_tokens):
            T = ou_euler_maruyama(T, mu, theta, sigma, dt)
            T = float(np.clip(T, 0.1, 1.5))

            outputs = model(input_ids=generated)
            logits = outputs.logits[:, -1, :] / T

            entropy = compute_entropy(logits).item()

            if entropy < 2.0:
                mu += 0.02
            elif entropy > 5.0:
                mu -= 0.02

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            sample_entropies.append(entropy)
            sample_temps.append(T)

        avg_entropies_per_step.append(sample_entropies)
        avg_temps_per_step.append(sample_temps)

    avg_entropies_per_step = np.array(avg_entropies_per_step)
    avg_temps_per_step = np.array(avg_temps_per_step)

    mean_entropy_curve = avg_entropies_per_step.mean(axis=0)
    mean_temp_curve = avg_temps_per_step.mean(axis=0)

    return mean_entropy_curve, mean_temp_curve

entropy_curve, temp_curve = multi_turn_ou_tracking(prompt)

steps = np.arange(len(entropy_curve))

fig, ax1 = plt.subplots(figsize=(10,6))

ax1.plot(steps, entropy_curve, linestyle="--", marker="o", label="Avg Entropy")
ax1.set_xlabel("Generation Step")
ax1.set_ylabel("Entropy")

ax2 = ax1.twinx()
ax2.plot(steps, temp_curve, linestyle="-", marker="s", label="Avg Temperature")
ax2.set_ylabel("Temperature")

plt.title("OU (Euler-Maruyama) Multi-turn Avg Temperature & Entropy Dynamics")
fig.tight_layout()
plt.show()