import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm

temps = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4 ])

model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    
    model_id,
    dtype=torch.bfloat16,
    device_map="auto"
)



prompt = "A builder has a team of workers completing floors at different rates depending on weather, materials, and coordination. If 60 workers complete 4 floors under normal conditions, how might output change if the team size doubles under varying constraints? Explain step by step."
    

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



def check_entropy(temperatures, prompt, num_samples=120): #Increase num_samples for smoother results
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    mean_entropies = []

    for temp in tqdm(temperatures, desc="Calculating Entropy"):
        sample_entropies = []
        
        # PAPER ALIGNMENT: Average over multiple generations to find the "Expected" entropy

        for _ in range(num_samples):
            outputs = model.generate(
                **inputs,
                temperature=temp,
                do_sample=True,
                max_new_tokens=100,
                return_dict_in_generate=True,
                output_scores=True
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
avg_entropies = check_entropy(temps, prompt)
log_avg_entropies = np.log(avg_entropies + 1e-12)

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
plt.show()