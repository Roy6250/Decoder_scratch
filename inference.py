import torch
from transformers import AutoTokenizer
from main import GPT, GPTConfig
import matplotlib.pyplot as plt
# from main import encode, decode

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
v = len(chars)
print(chars, v)

# create a mapping from characters to integers & vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

def plot_attention_map(attention_weights, input_tokens):
    # Select the first head and first batch for visualization
    attn_map = attention_weights[0, 0].cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(attn_map, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(input_tokens)), input_tokens, rotation=90)
    plt.yticks(range(len(input_tokens)), input_tokens)
    plt.title("Attention Map")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.savefig("attention_plot.png")

    plt.show()
# Configuration
model_save_path = "gpt_model.pth"  # Path to save the model weights
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Function to load the model
def load_model(config, path):
    model = GPT(config).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {path}")
    return model

def get_attention_map(model, input_ids):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        attention_weights = model.transformer.h[-1].attn.attention_weights  # Extract attention weights from the last layer
    return attention_weights

# Example usage
if __name__ == "__main__":
    # Define the same configuration used during training
    config = GPTConfig(
        block_size=256,
        vocab_size=50304,  # GPT-2 vocab size
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.2
    )
    # Load the trained model
    model = load_model(config, model_save_path)

    # Input prompt for inference
    prompt = "Whell's the couse,"
    input_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    output = model.generate(input_ids, max_new_tokens=218,temperature=1.5, top_k=50)
    attention_weights = get_attention_map(model, input_ids)
    output_str = decode(output[0].tolist())
    print(output_str)
