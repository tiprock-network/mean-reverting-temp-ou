import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pathlib import Path

#get the parser and parsed text
parser = argparse.ArgumentParser(description="EulerTemp Console")
parser.add_argument("model", help="The name of the LLM model")
parser.add_argument("--num_samples", type=int, default=120, help="Number of generations per temperature")
parser.add_argument("--max_new_tokens", type=int, default=100, help="Generated tokens per sample")
parser.add_argument("--save_dir", type=str, default="outputs/figures", help="Directory to save plots")
parser.add_argument("--show", action="store_true", help="Display plots interactively")

args = parser.parse_args()

temps = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4 ])

model_id = args.model

tokenizer = AutoTokenizer.from_pretrained(model_id)

if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
else:
    model = AutoModelForCausalLM.from_pretrained(model_id)

model.eval()

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")



prompt = "Question: Lauren and Jane work for the same company. They each need to use a computer for work sometimes. Unfortunately, the computer isn't very powerful. If two people are logged on at the same time, it usually crashes. So the company decided to institute an official policy. It declared that Lauren would be the only one permitted to use the computer in the mornings and that Jane would be the only one permitted to use the computer in the afternoons. As expected, Lauren logged on the computer the next day at 9:00 am. But Jane decided to disobey the official policy. She also logged on at 9:00 am. The computer crashed immediately. Did Jane cause the computer to crash?  Reply Yes or No based on the answer the majority of people would give. If you think people would be split roughly 50-50 between Yes and No then reply Ambiguous."
    

def calc_turning_point(x_values, log_y_values):
    """
    Identifies the temperature where log-entropy begins to accelerate.
    
    Args:
        x_values: Array of temperatures (e.g., [0.1, 0.2, ..., 1.6])
        log_y_values: Array of the log of mean entropies at those temperatures
    """
    #Calculate the first derivative (Rate of change)
    dy_dx = np.gradient(log_y_values, x_values)

    #Calculate the second derivative (Acceleration)
    d2y_dx2 = np.gradient(dy_dx, x_values)

    print("First Derivative (dy_dx): ", dy_dx)
    print("Second Derivative (d2y_dx2): ", d2y_dx2)
    
    #Find the first point where acceleration becomes positive.
    
    mask = d2y_dx2[1:-1] > 0
    
    if not np.any(mask):
        
        mask_index = -1
    else:
        #Get the first index where the second derivative > 0
        mask_index = np.where(mask)[0][0] 
        # djust index back because we sliced the array
        mask_index += 1
        
    turning_point = x_values[mask_index]
    return turning_point



def check_entropy(temperatures, prompt, num_samples=120, max_new_tokens=100): #Increase num_samples for smoother results
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    mean_entropies = []

    for temp in tqdm(temperatures, desc="Calculating Entropy"):
        sample_entropies = []
        
        # PAPER ALIGNMENT: Average over multiple generations to find the "Expected" entropy

        with torch.inference_mode():
            for _ in range(num_samples):
                outputs = model.generate(
                    **inputs,
                    temperature=temp,
                    do_sample=True,
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True,
                )

                current_sequence_entropies = []
                for step_logits in outputs.scores:
                    probs = F.softmax(step_logits, dim=-1)
                    # Use a larger top-k or full vocab for the entropy sum
                    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
                    current_sequence_entropies.append(entropy.item())

                sample_entropies.append(np.mean(current_sequence_entropies))
        
        #Average the means of all samples for this temperature
        mean_entropies.append(np.mean(sample_entropies))

    return np.array(mean_entropies)

#Execute
avg_entropies = check_entropy(
    temps,
    prompt,
    num_samples=args.num_samples,
    max_new_tokens=args.max_new_tokens,
)
log_avg_entropies = np.log(avg_entropies + 1e-12)
save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)



fig, ax1 = plt.subplots(figsize=(10, 6))

#Raw Entropy
ax1.plot(temps, avg_entropies, marker="o", linestyle="--", color="red", label="Mean Entropy (H)")
ax1.set_xlabel("Temperature (T)")
ax1.set_ylabel("Mean Entropy", color="red")
ax1.tick_params(axis='y', labelcolor="red")

#Log Entropy
ax2 = ax1.twinx()
ax2.plot(temps, log_avg_entropies, marker="s", linestyle="-", color="green", label="Log Mean Entropy (ln H)")
ax2.set_ylabel("Log Mean Entropy", color="green")
ax2.tick_params(axis='y', labelcolor="green")

#Identify the Turning Point on the graph
turning_point_t = calc_turning_point(temps, log_avg_entropies)
plt.axvline(x=turning_point_t, color='blue', linestyle=':', label=f'Turning Point: {turning_point_t:.2f}')

plt.title("Entropy acceleration across Temperatures")
fig.tight_layout()
fig.savefig(save_dir / "multi_turn_entropy_acceleration.png", dpi=200)
if args.show:
    plt.show()
else:
    plt.close(fig)

steps = np.arange(len(avg_entropies))



fig1, ax4 = plt.subplots(figsize=(10, 6))

# Plotting Entropy
ax4.plot(steps, avg_entropies, 
        linestyle="--", 
        marker="o", 
        color="tab:blue", 
        linewidth=2, 
        label="Avg Entropy")

# Formatting
ax4.set_title("Entropy Dynamics per Generation Step", fontsize=14, pad=15)
ax4.set_xlabel("Generation Step", fontsize=12)
ax4.set_ylabel("Entropy", fontsize=12)
ax4.grid(True, linestyle=':', alpha=0.6) # Added grid for better readability
ax4.legend(loc="upper right")

fig1.tight_layout()
fig1.savefig(save_dir / "multi_turn_entropy_dynamics.png", dpi=200)
if args.show:
    plt.show()
else:
    plt.close(fig1)

fig2, ax3 = plt.subplots(figsize=(10,6))
ax3.plot(temps, avg_entropies, marker='o', linestyle='-', color="orange")
ax3.set_xlabel("Temperature")
ax3.set_ylabel("Entropy")
ax3.set_title("Entropy vs. Temperature (Multi-turn)")
ax3.grid(True)

fig2.tight_layout()
fig2.savefig(save_dir / "multi_turn_entropy_vs_temperature.png", dpi=200)
if args.show:
    plt.show()
else:
    plt.close(fig2)