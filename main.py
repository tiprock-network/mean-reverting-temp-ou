import argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.ou_eulerm import AdaptiveOUInference


#get the parser and parsed text
parser = argparse.ArgumentParser(description="EulerTemp Console")
parser.add_argument("model", help="The name of the LLM model")

args = parser.parse_args()

from colorama import init, Fore, Style

model_id = args.model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")

# Initialize colorama for Windows compatibility
init(autoreset=True)

def print_logo():
    logo = r"""
  ______      _                     _______                      

 |  ____|    | |                   |__   __|                     
 | |__  _   _| | ___ _ __ _ __ ___    | | ___ _ __ ___  _ __     
 |  __| | | | | |/ _ \ '__| '_ ` _ \   | |/ _ \ '_ ` _ \| '_ \    
 | |____| |_| | |  __/ |  | | | | | |  | |  __/ | | | | | |_) |   
 |______|\__,_|_|\___|_|  |_| |_| |_|  |_|\___|_| |_| |_| .__/    
                                                        | |       
                                                        |_|       
"""
    print(logo)
    print(f"{Fore.YELLOW}*** Welcome to EulermTemp v1.0 ***{Style.RESET_ALL}")
    print("-" * 40)

# Create a system message to enforce brevity and style
system_message = "You are a helpful, concise assistant. Give direct answers using step-by-step reasoning."


if __name__ == "__main__":
    print_logo()

    # Initialize once
    adaptive_session = AdaptiveOUInference(
        model = model, 
        tokenizer = tokenizer,
        mu = 0.5,
        T= 0.5,
        theta= 0.2,
        sigma= 0.05
    )

    session_alive = True
    #do a while loop to permit exit from prompt
    emoji_list = ["Are you planning something and a co-pilot to help you plan? 😏: ", "Hi, ask away?🙂: ","I am ready to answer, just type away☺️: ", "Don't dose away, let's do some math or do a quiz 😴: "]

    while session_alive:
        dynamic_emoji_idx = random.randint(0,len(emoji_list)-1)
        print(emoji_list[dynamic_emoji_idx])
        user_prompt = input()

        if user_prompt == "exit":
            break

        # Wrap the prompt in the Instruct template
        chat_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>" \
                    f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>" \
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"

        # First generation (starts at T=0.4)
        output_text = adaptive_session.generate(chat_prompt)
        print(f"Current Temp: {adaptive_session.T}")
        print(output_text)