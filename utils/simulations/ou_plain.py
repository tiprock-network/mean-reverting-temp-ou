
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse
from utils.simulations.config_utils import load_config

def compute_accuracy(predictions, references):
    # Simple exact match accuracy
    correct = 0
    for pred, ref in zip(predictions, references):
        if str(pred).strip() == str(ref).strip():
            correct += 1
    return correct / len(predictions) if predictions else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--benchmark_dataset', type=str, default=None)
    parser.add_argument('--dataset_config', type=str, default=None)
    parser.add_argument('--dataset_split', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    model_id = args.model or config.get("model", "meta-llama/Meta-Llama-3-8B-Instruct")
    max_new_tokens = args.max_new_tokens or config.get("max_new_tokens", 100)
    dataset_name = args.benchmark_dataset or config.get("benchmark_dataset", {}).get("name", "lighteval/bbeh")
    dataset_config = args.dataset_config or config.get("benchmark_dataset", {}).get("config", None)
    dataset_split = args.dataset_split or config.get("benchmark_dataset", {}).get("split", "validation")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        torch.set_float32_matmul_precision("high")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    # Load dataset with error handling
    from datasets.exceptions import DatasetNotFoundError
    try:
        if dataset_config:
            ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
        else:
            ds = load_dataset(dataset_name, split=dataset_split)
    except DatasetNotFoundError as e:
        print(f"[ERROR] Dataset '{dataset_name}' not found or cannot be accessed.\n{e}")
        print("\nPlease check the dataset name. You can search for available datasets at https://huggingface.co/datasets or with the 'datasets-cli' tool.")
        exit(2)
    # Try to infer input/label columns
    input_col = None
    label_col = None
    for col in ds.column_names:
        if col in ["question", "input", "prompt"]:
            input_col = col
        if col in ["answer", "label", "target", "output", "reference"]:
            label_col = col
    if input_col is None:
        input_col = ds.column_names[0]
    if label_col is None:
        label_col = ds.column_names[-1]

    predictions = []
    references = []
    entropies = []
    temperatures = []

    # OU parameters from config
    ou_cfg = config.get("ou", {})
    T = ou_cfg.get("T_init", 0.7)
    mu = ou_cfg.get("mu", 0.7)
    theta = ou_cfg.get("theta", 0.5)
    sigma = ou_cfg.get("sigma", 0.1)
    dt = ou_cfg.get("dt", 1.0)

    for idx, example in enumerate(ds):
        prompt = example[input_col]
        reference = example[label_col]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generated = inputs["input_ids"]

        with torch.inference_mode():
            outputs = model(input_ids=generated, use_cache=True)
        current_logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        sample_entropies = []
        sample_temperatures = []
        for _ in range(max_new_tokens):
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
            sample_entropies.append(entropy)
            sample_temperatures.append(T)

        output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        predictions.append(output_text)
        references.append(reference)
        entropies.append(np.mean(sample_entropies))
        temperatures.append(np.mean(sample_temperatures))

    accuracy = compute_accuracy(predictions, references)
    avg_entropy = float(np.mean(entropies))
    avg_temperature = float(np.mean(temperatures))

    # Save metrics to JSON
    save_dir = config.get("save_dir", "outputs/figures")
    os.makedirs(save_dir, exist_ok=True)
    # Compose output filename
    model_short = model_id.split("/")[-1].replace(".", "-")
    dataset_short = dataset_name.split("/")[-1].replace(".", "-")
    out_file = args.output_file or os.path.join(save_dir, f"ou_plain_{model_short}_{dataset_short}.json")
    metrics = {
        "avg_entropy": avg_entropy,
        "avg_temperature": avg_temperature,
        "accuracy": accuracy,
        "entropies": entropies,
        "temperatures": temperatures,
        "predictions": predictions,
        "references": references,
    }
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved results to {out_file}")

if __name__ == "__main__":
    main()

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
with open(os.path.join(save_dir, "ou_plain_2-7B-Instruct_causal_judgement.json"), "w") as f:
    json.dump(metrics, f, indent=2)
