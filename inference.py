import torch
from transformers import AutoTokenizer
from main import GPT, GPTConfig

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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

    # Load the trained model
    model = load_model(config, model_save_path)

    # Input prompt for inference
    prompt = "Whell's the couse,"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    print(f"Input IDs: {input_ids}")
    # Generate text
    max_new_tokens = 50
    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=1.5, top_k=50)

    # Decode and print the generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Generated Text:\n{generated_text}")
