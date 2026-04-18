import numpy as np
import torch
import torch.nn.functional as F

from utils.math_functions import ou_euler_maruyama, compute_entropy, compute_kl


def adaptive_generation_em(model, tokenizer, prompt, max_new_tokens=100):
    """
    This code implements the Adaptive OU method for Mean-Reversion using OU implemented using Euler Maruyama method.
    Use this for benchmarks, and evaluations.
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

class AdaptiveOUInference:
    def __init__(self, model, tokenizer, mu=0.4, T=0.4, theta=0.5, sigma=0.1):
        self.model = model
        self.tokenizer = tokenizer
        
        # Persistent State Variables, can be declared by model owner or changed by individual
        self.mu = mu        # The "Target" temperature
        self.T = T          # The current temperature
        self.theta = theta  # Speed of reversion
        self.sigma = sigma  # Volatility (randomness)
        self.dt = 1.0
        
        # Thresholds
        self.entropy_low = 2.0
        self.entropy_high = 5.0
        self.kl_threshold = 2.0
        
        # History tracking
        self.history = {"temp": [], "entropy": [], "mu": []}

    def _ou_step(self):
        """Update T using Euler-Maruyama with smaller increments."""
        dw = np.random.normal(0, np.sqrt(self.dt))
        # The change in T (dT)
        drift = self.theta * (self.mu - self.T) * self.dt
        diffusion = self.sigma * dw
        
        self.T += drift + diffusion
        # Maintain a floor and ceiling
        self.T = float(np.clip(self.T, 0.1, 1.8))

    def generate(self, prompt, max_tokens=150):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated = inputs["input_ids"]

        stop_tokens = [self.tokenizer.eos_token_id]
        if "<|eot_id|>" in self.tokenizer.get_vocab():
            stop_tokens.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        
        # Get reference logits from the initial prompt
        with torch.no_grad():
            ref_logits = self.model(**inputs).logits[:, -1, :]

        for _ in range(max_tokens):
            self._ou_step() # Evolve T based on current state

            with torch.no_grad():
                # Pass the full sequence through the model
                
                outputs = self.model(input_ids=generated)
                logits = outputs.logits[:, -1, :]
                
                # APPLY THE TEMPERATURE
                scaled_logits = logits / self.T
                
                # Feedback Loop: Calculate metrics
                entropy = compute_entropy(scaled_logits)
                kl = compute_kl(scaled_logits, ref_logits)

                # ADJUST STATE: Update mu and theta based on metrics
                self._adjust_parameters(entropy, kl)

                # Sample next token
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                # Check if the sampled token is a stop token
                if next_token.item() in stop_tokens:
                    break
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Track for visualization
                self.history["temp"].append(self.T)
                self.history["entropy"].append(entropy)
                self.history["mu"].append(self.mu)

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def _adjust_parameters(self, entropy, kl):
        """Update internal drift and reversion speed."""
        # If the model is too 'bored' (low entropy), push it to explore (raise mu)
        if entropy < self.entropy_low:
            self.mu = min(1.2, self.mu + 0.01) 
        # If the model is too 'chaotic' (high entropy), pull it back (lower mu)
        elif entropy > self.entropy_high:
            self.mu = max(0.2, self.mu - 0.01)

        # If we deviate too far from the base model's logic, force high-speed reversion
        if kl > self.kl_threshold:
            self.mu = max(0.1, self.mu - 0.03)
            self.theta = min(0.8, self.theta + 0.02)
        else:
            self.theta = max(0.1, self.theta - 0.005)