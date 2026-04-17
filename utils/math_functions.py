import numpy as np
import torch
import torch.nn.functional as F

def compute_entropy(logits):
    probs = F.softmax(logits, dim=-1)
    return -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

def compute_kl(p_logits, q_logits):
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    return (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(dim=-1)

def ou_euler_maruyama(T, mu, theta, sigma, dt):
    noise = np.random.randn()
    return T + theta * (mu - T) * dt + sigma * np.sqrt(dt) * noise