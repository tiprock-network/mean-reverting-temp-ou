"""
Benchmark AdaptiveOUExact (exact Ornstein-Uhlenbeck temperature control) on
three standard multiple-choice datasets:
  - ARC-Challenge  (allenai/ai2_arc, ARC-Challenge split)
  - TruthfulQA MC1 (truthful_qa, multiple_choice)
  - MMLU           (cais/mmlu, all)

Evaluation metric: softmatch accuracy = correct_responses / total_responses
Softmatch normalises both the model answer and the gold answer label, then
checks whether the normalised gold answer appears inside the normalised output.

Usage:
    python -m utils.benchmarking.benchmark --model <hf_model_id> [options]

Example (recommended 7B/8B models):
    python -m utils.benchmarking.benchmark --model meta-llama/Meta-Llama-3-8B-Instruct
    python -m utils.benchmarking.benchmark --model mistralai/Mistral-7B-Instruct-v0.3
    python -m utils.benchmarking.benchmark --model Qwen/Qwen2.5-7B-Instruct

Optional flags:
    --mu        float   OU long-run mean temperature  (default 0.5)
    --T         float   OU initial temperature        (default 0.5)
    --theta     float   OU speed of mean reversion    (default 0.2)
    --sigma     float   OU noise scale                (default 0.05)
    --max_tokens int    Max new tokens per response   (default 50)
    --n_samples  int    Max questions per benchmark   (default 100)
    --seed      int     RNG seed                      (default 42)
    --output    str     Path to save JSON results     (default results.json)
"""

import argparse
import json
import random
import re
import string
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Allow running as `python utils/benchmarking/benchmark.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.ou_eulerm import AdaptiveOUExact  # noqa: E402


# ---------------------------------------------------------------------------
# Softmatch helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def softmatch(prediction: str, gold: str) -> bool:
    """
    Return True if the normalised gold answer is contained in the normalised
    prediction, or if they share the same leading token after normalisation.
    This handles both letter answers ('a', 'b', 'c', 'd') and free-text labels.
    """
    pred_norm = _normalise(prediction)
    gold_norm = _normalise(gold)
    if not gold_norm:
        return False
    # Direct substring match
    if gold_norm in pred_norm:
        return True
    # Token-level overlap: first token of gold appears at word boundary in pred
    gold_first = gold_norm.split()[0]
    if re.search(r"\b" + re.escape(gold_first) + r"\b", pred_norm):
        return True
    return False


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

LETTER_MAP = ["A", "B", "C", "D", "E"]


def _build_arc_prompt(example: dict) -> tuple[str, str]:
    """Return (prompt, gold_label) for an ARC-Challenge example."""
    choices = example["choices"]
    options_text = "\n".join(
        f"{LETTER_MAP[i]}. {label}"
        for i, label in enumerate(choices["text"])
    )
    # answerKey is e.g. "A" / "1" depending on version; normalise to letter
    answer_key = example["answerKey"].strip()
    if answer_key.isdigit():
        answer_key = LETTER_MAP[int(answer_key) - 1]
    prompt = (
        f"Question: {example['question']}\n"
        f"Options:\n{options_text}\n"
        "Answer with the letter of the correct option only."
    )
    return prompt, answer_key


def _build_truthfulqa_prompt(example: dict) -> tuple[str, str]:
    """Return (prompt, gold_label) for a TruthfulQA MC1 example."""
    mc1 = example["mc1_targets"]
    choices = mc1["choices"]
    labels = mc1["labels"]           # 1 = correct, 0 = wrong
    correct_idx = labels.index(1)
    options_text = "\n".join(
        f"{LETTER_MAP[i]}. {c}" for i, c in enumerate(choices)
    )
    gold_label = LETTER_MAP[correct_idx]
    prompt = (
        f"Question: {example['question']}\n"
        f"Options:\n{options_text}\n"
        "Answer with the letter of the correct option only."
    )
    return prompt, gold_label


def _build_mmlu_prompt(example: dict) -> tuple[str, str]:
    """Return (prompt, gold_label) for an MMLU example."""
    choices = example["choices"]
    options_text = "\n".join(
        f"{LETTER_MAP[i]}. {c}" for i, c in enumerate(choices)
    )
    gold_label = LETTER_MAP[int(example["answer"])]
    prompt = (
        f"Question: {example['question']}\n"
        f"Options:\n{options_text}\n"
        "Answer with the letter of the correct option only."
    )
    return prompt, gold_label


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_arc(n_samples: int, seed: int):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    return [_build_arc_prompt(ex) for ex in ds], "ARC-Challenge"


def load_truthfulqa(n_samples: int, seed: int):
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    return [_build_truthfulqa_prompt(ex) for ex in ds], "TruthfulQA-MC1"


def load_mmlu(n_samples: int, seed: int):
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    return [_build_mmlu_prompt(ex) for ex in ds], "MMLU"


# ---------------------------------------------------------------------------
# Per-benchmark evaluation
# ---------------------------------------------------------------------------

def evaluate_benchmark(
    session: AdaptiveOUExact,
    tokenizer,
    examples: list[tuple[str, str]],
    benchmark_name: str,
    max_tokens: int,
    system_message: str,
) -> dict:
    """
    Run AdaptiveOUExact on every (prompt, gold) pair and compute softmatch
    accuracy.  Returns a result dict with per-question details and summary.
    """
    correct = 0
    total = len(examples)
    details = []

    for prompt_text, gold in tqdm(examples, desc=benchmark_name):
        # Format with instruct template
        chat_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_message}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        # Reset OU temperature to initial value before each question so that
        # state does not bleed across questions
        session.T = session.mu

        output_text = session.generate(chat_prompt, max_tokens=max_tokens)

        # Strip the prompt prefix from the output (tokenizer decode includes it)
        response = output_text[len(prompt_text):].strip() if prompt_text in output_text else output_text.strip()

        is_correct = softmatch(response, gold)
        if is_correct:
            correct += 1

        details.append(
            {
                "prompt": prompt_text,
                "gold": gold,
                "response": response,
                "correct": is_correct,
            }
        )

    accuracy = correct / total if total > 0 else 0.0
    print(
        f"\n[{benchmark_name}] Softmatch Accuracy: {correct}/{total} = {accuracy:.4f}\n"
    )
    return {
        "benchmark": benchmark_name,
        "correct": correct,
        "total": total,
        "softmatch_accuracy": accuracy,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AdaptiveOUExact on ARC-Challenge, TruthfulQA, MMLU."
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "HuggingFace model ID to benchmark. "
            "Recommended 7B/8B models: "
            "meta-llama/Meta-Llama-3-8B-Instruct, "
            "mistralai/Mistral-7B-Instruct-v0.3, "
            "Qwen/Qwen2.5-7B-Instruct"
        ),
    )
    parser.add_argument("--mu", type=float, default=0.5, help="OU long-run mean temperature (default 0.5)")
    parser.add_argument("--T", type=float, default=0.5, help="OU initial temperature (default 0.5)")
    parser.add_argument("--theta", type=float, default=0.2, help="OU speed of mean reversion (default 0.2)")
    parser.add_argument("--sigma", type=float, default=0.05, help="OU noise scale (default 0.05)")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max new tokens per answer (default 50)")
    parser.add_argument("--n_samples", type=int, default=100, help="Max questions per benchmark (default 100)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON file path (default results.json)")
    args = parser.parse_args()

    # Seed everything for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on {device}.\n")

    # Initialise a single AdaptiveOUExact session (state is reset per question)
    session = AdaptiveOUExact(
        model=model,
        tokenizer=tokenizer,
        mu=args.mu,
        T=args.T,
        theta=args.theta,
        sigma=args.sigma,
    )

    system_message = (
        "You are a helpful, concise assistant. "
        "When answering multiple-choice questions respond with only the letter "
        "of the correct answer (A, B, C, or D)."
    )

    # Load benchmarks
    print("Loading benchmarks …")
    arc_examples, arc_name = load_arc(args.n_samples, args.seed)
    tqa_examples, tqa_name = load_truthfulqa(args.n_samples, args.seed)
    mmlu_examples, mmlu_name = load_mmlu(args.n_samples, args.seed)
    print(
        f"  {arc_name}: {len(arc_examples)} questions\n"
        f"  {tqa_name}: {len(tqa_examples)} questions\n"
        f"  {mmlu_name}: {len(mmlu_examples)} questions\n"
    )

    all_results = []

    for examples, name in [
        (arc_examples, arc_name),
        (tqa_examples, tqa_name),
        (mmlu_examples, mmlu_name),
    ]:
        result = evaluate_benchmark(
            session=session,
            tokenizer=tokenizer,
            examples=examples,
            benchmark_name=name,
            max_tokens=args.max_tokens,
            system_message=system_message,
        )
        all_results.append(result)

    # Overall summary
    total_correct = sum(r["correct"] for r in all_results)
    total_questions = sum(r["total"] for r in all_results)
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0

    summary = {
        "model": args.model,
        "ou_params": {
            "mu": args.mu,
            "T_init": args.T,
            "theta": args.theta,
            "sigma": args.sigma,
        },
        "overall_softmatch_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "benchmarks": all_results,
    }

    print("=" * 60)
    print(f"Model : {args.model}")
    print(f"Overall Softmatch Accuracy: {total_correct}/{total_questions} = {overall_accuracy:.4f}")
    for r in all_results:
        print(f"  {r['benchmark']}: {r['correct']}/{r['total']} = {r['softmatch_accuracy']:.4f}")
    print("=" * 60)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
