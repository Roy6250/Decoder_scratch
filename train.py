import math
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader,Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from main import GPT, GPTConfig # Assuming these are in your local 'main.py'
import inspect
import matplotlib.pyplot as plt

# Hyperparameters
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset
dataset = load_dataset('tiny_shakespeare', split='train')

dataset = load_dataset('tiny_shakespeare')
train_dataset = dataset['train']
val_dataset= dataset['validation']
# print(val_dataset[0])

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=block_size, padding="max_length")

train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_tokenized = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
# This function now correctly creates the input and target pairs for next-token prediction.
def collate_fn(batch):
    # Stack the tokenized inputs into a single tensor
    inputs_tensor = torch.tensor([item['input_ids'] for item in batch], dtype=torch.long)
    # The 'inputs' for the model will be all tokens except the last one
    inputs = inputs_tensor[:, :-1].contiguous()
    
    # The 'targets' for the model will be all tokens except the first one (shifted by one)
    targets = inputs_tensor[:, 1:].contiguous()
    
    return inputs, targets

# # # Prepare DataLoader with the corrected collate function
train_dataloader = DataLoader(train_tokenized, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_tokenized, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# # --- Verification Step ---
# # Let's check the output of the dataloader
# data_iterator = iter(dataloader)
# first_batch = next(data_iterator)

# # Unpack and print the data to verify the shapes
# features_batch, labels_batch = first_batch

# # print("--- First Batch Fetched from DataLoader ---")

# # # Check if the first token of the target is the second token of the input
# # print(f"\nFirst input token for sample 0: {features_batch[0, 0]}")
# # print(f"Second input token for sample 0: {features_batch[0, 1]}")
# # print(f"First target token for sample 0: {labels_batch[0, 0]}")
# # print("Note: The first target token should be the same as the second input token.")

# # print("\nFeatures in the batch (Inputs to the model):")
# # print(features_batch)
# # print("\nShape of the features tensor:", features_batch.shape) # Should be [batch_size, block_size - 1]

# # print("\nLabels in the batch (Targets for the model):")
# # print(labels_batch)
# # print("\nShape of the labels tensor:", labels_batch.shape) # Should be [batch_size, block_size - 1]


# train_data = np.fromfile(os.path.join(os.path.dirname(__file__), 'train.bin'), dtype=np.uint16)
# val_data = np.fromfile(os.path.join(os.path.dirname(__file__), 'val.bin'), dtype=np.uint16)

# # Dataset class
# class ShakespeareDataset(Dataset):
#     def __init__(self, data, block_size):
#         self.data = data
#         self.block_size = block_size

#     def __len__(self):
#         return len(self.data) - self.block_size

#     def __getitem__(self, idx):
#         x = torch.tensor(self.data[idx:idx + self.block_size], dtype=torch.long)
#         y = torch.tensor(self.data[idx + 1:idx + 1 + self.block_size], dtype=torch.long)
#         return x, y

# # Create datasets and dataloaders
# train_dataset = ShakespeareDataset(train_data, block_size)
# val_dataset = ShakespeareDataset(val_data, block_size)
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



config = GPTConfig(
    block_size=block_size,
    vocab_size=50304,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout
)
model = GPT(config).to(device)

# Optimizer
optimizer = model.configure_optimizers(weight_decay=0.2, learning_rate=learning_rate, betas=(0.9, beta2), device_type=device)

# Learning rate scheduler
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + (learning_rate - min_lr) * (1 - decay_ratio)

# Training loop
model.train()
scaler = torch.amp.GradScaler() if device == 'cuda' else None  # Updated GradScaler
iter_num = 0  # Initialize iteration counter
train_losses=[]
val_losses=[]



while iter_num < max_iters:
    train_loss=0
    for inputs, targets in train_dataloader:
        if iter_num >= max_iters:
            break
        
        inputs, targets = inputs.to(device), targets.to(device)
        if iter_num % 100 == 0:
            pass

        # Forward pass
        with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):  # Updated autocast
            logits, loss = model(inputs, targets=targets)

            if iter_num%100 ==0:
                # print(f"Logits shape: {logits.shape}, Loss: {loss.item()}")
                # print(logits)
                pass

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (iter_num + 1) % gradient_accumulation_steps == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        train_loss+=loss.item()

    # Adjust learning rate
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Logging
    if iter_num % 100 == 0:
        print(f"Iteration {iter_num}, Loss: {loss.item():.6f}, LR: {lr:.6f}")

    iter_num += 1  # Increment iteration counter
    train_losses.append(train_loss / len(train_dataloader))  # Average loss for the epoch

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs,targets in val_dataloader:
            inputs,targets = inputs.to(device), targets.to(device)

            with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):
                logits,loss = model(inputs, targets=targets)
            val_loss += loss.item()

    val_losses.append(val_loss / len(val_dataloader))  # Average validation loss
    model.train()  # Switch back to training mode
    print(f"Iteration {iter_num}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}, LR: {lr:.6f}")


model_save_path = "gpt_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss")
# plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training  Loss")
plt.legend()
plt.savefig("loss_plot.png")
plt.show()

print("Training complete!")

model_save_path = "gpt_model.pth"  # Path to save the model weights
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

save_model(model, model_save_path)
