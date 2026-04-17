import numpy as np
import torch
import torch.nn.functional as F

from math_functions import ou_euler_maruyama, compute_entropy, compute_kl


def adaptive_generation_em(model, tokenizer, prompt, max_new_tokens=100):
    """
    This code implements the Adaptive OU method for Mean-Reversion using OU implemented using Euler Maruyama method.
    """
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
    theta = 0.5
    sigma = 0.1
    dt = 1.0

    entropy_low = 2.0
    entropy_high = 5.0
    kl_threshold = 2.0

    generated = inputs["input_ids"]

    entropies = []
    temperatures = []

    for _ in range(max_new_tokens):
        T = ou_euler_maruyama(T, mu, theta, sigma, dt)
        T = float(np.clip(T, 0.1, 1.5))

        outputs = model(input_ids=generated)
        logits = outputs.logits[:, -1, :] / T

        entropy = compute_entropy(logits).item()
        kl = compute_kl(logits, ref_logits).item()

        if entropy < entropy_low:
            mu += 0.02
        elif entropy > entropy_high:
            mu -= 0.02

        if kl > kl_threshold:
            mu -= 0.05
            theta = min(1.0, theta + 0.05)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=-1)

        entropies.append(entropy)
        temperatures.append(T)

    return generated, entropies, temperatures

