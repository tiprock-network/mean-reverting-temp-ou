import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
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

def compute_kl(p_logits, q_logits):
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    return (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(dim=-1)

def ou_step(T, mu, theta, sigma):
    noise = np.random.randn()
    return T + theta * (mu - T) + sigma * noise

def adaptive_generation(prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    base_outputs = model.generate(
        **inputs,
        temperature=0.2,
        do_sample=True,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_scores=True
    )

    ref_logits = base_outputs.scores[0]

    T = 0.4
    mu = 0.4
    theta = 0.1
    sigma = 0.05

    entropy_low = 2.0
    entropy_high = 5.0
    kl_threshold = 2.0

    generated = inputs["input_ids"]

    entropies = []
    temperatures = []

    for _ in range(max_new_tokens):
        T = max(0.1, min(1.5, ou_step(T, mu, theta, sigma)))

        outputs = model(
            input_ids=generated
        )

        logits = outputs.logits[:, -1, :] / T

        entropy = compute_entropy(logits).item()
        kl = compute_kl(logits, ref_logits).item()

        if entropy < entropy_low:
            mu += 0.02
        elif entropy > entropy_high:
            mu -= 0.02

        if kl > kl_threshold:
            mu -= 0.05
            theta = min(0.5, theta + 0.05)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)

        entropies.append(entropy)
        temperatures.append(T)

    return generated, entropies, temperatures

output_ids, entropies, temps = adaptive_generation(prompt)

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(generated_text)
print("Avg Entropy:", np.mean(entropies))
print("Avg Temp:", np.mean(temps))