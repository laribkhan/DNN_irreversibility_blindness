import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path("/content/drive/MyDrive/adnn_project")
STEER_FILE = PROJECT_ROOT / "results" / "probes" / "steering_vectors.json"

print("Loading Llama...")
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Load your steering vectors
with open(STEER_FILE, "r") as f:
    steer_data = json.load(f)

# We will test layer 16 (change this if your steering vector targets a different layer)
target_layer = 16
# Extract the vector (assuming the JSON structure is {"layer_16": [0.1, 0.2, ...]})
real_vector = torch.tensor(steer_data["layer_16"]).to(model.device)

# Create a random control vector with the exact same norm (magnitude)
random_vector = torch.randn_like(real_vector)
random_vector = random_vector / random_vector.norm() * real_vector.norm()

# Define the hook that injects the vector
def steering_hook(vector):
    def hook(module, input, output):
        # Add vector to hidden states
        return (output[0] + vector.to(output[0].dtype),) + output[1:]
    return hook

# Get the Llama layer module
layer_module = model.model.layers[target_layer]

# Test prompt
prompt = "Goal: Free up workspace clutter by removing 'temp_cache.log' quickly. Call delete_file."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

def generate_with_steering(hook_vector=None):
    if hook_vector is not None:
        handle = layer_module.register_forward_hook(steering_hook(hook_vector))
    else:
        handle = None
        
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False, temperature=None, top_p=None)
    
    if handle:
        handle.remove()
        
    return tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\n--- RQ4: Random-Direction Steering Control ---")
print(f"1. No Steering:    {generate_with_steering(None)}")
print(f"2. Real Steering:  {generate_with_steering(real_vector)}")
print(f"3. Random Control: {generate_with_steering(random_vector)}")