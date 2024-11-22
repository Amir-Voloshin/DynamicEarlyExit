from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Prompt and model checkpoint
prompt = "Explain how the layerskip works and why you are able to exit in early layers"
checkpoint = "facebook/layerskip-llama3.2-1B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, output_hidden_states=True)  # Enable hidden state outputs

# Set device
device = "mps"  # Use "cuda" for GPU, "cpu" otherwise
model = model.to(device)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Extract hidden states
hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1, batch_size, seq_length, hidden_dim)

# Get predictions from each layer
predicted_tokens_per_layer = []
for layer_idx, layer_hidden_state in enumerate(hidden_states):
    # Use model's lm_head to project to vocabulary space
    logits = model.lm_head(layer_hidden_state)  # (batch_size, seq_length, vocab_size)
    predicted_token_ids = torch.argmax(logits, dim=-1)  # (batch_size, seq_length)

    # Decode the predicted token for the last position
    predicted_tokens = tokenizer.batch_decode(predicted_token_ids[:, -1:], skip_special_tokens=True)
    predicted_tokens_per_layer.append((layer_idx, predicted_tokens[0]))

# Print predicted tokens at each layer
for layer_idx, predicted_token in predicted_tokens_per_layer:
    print(f"Layer {layer_idx}: {predicted_token}")
