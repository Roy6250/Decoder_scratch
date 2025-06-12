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
import time
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
eval_interval=100
eval_iters=20

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

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size , (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])

    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


# x,y = get_batch('train')
# print("x[0] ", x[0].shape, "\n", x[0])
# # print("y[:,0,...] ", y[:,0,...].shape, "\n", y[:,0,...])
# # print(y[0].shape, "\n", y[0])

# print(x.shape)


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

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # sets model to eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # just resets to training mode
    return out

start_time = time.time()

for iter in range(max_iters):

    xb,yb= get_batch('train')
    xb, yb = xb.to(device), yb.to(device)
    # Forward pass
    with torch.amp.autocast(device_type=device, enabled=(device == 'cuda')):  # Updated autocast
        logits, loss = model(xb, targets=yb)
    optimizer.zero_grad()

     # Backward pass
    if scaler:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    optimizer.step()

    # Gradient accumulation
    if iter % eval_interval == 0 or iter == max_iters - 1:
        current_time = time.time()
        elapsed_time = current_time - start_time
        losses = estimate_loss()
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time elapsed: {elapsed_time:.2f} seconds")



model_save_path = "gpt_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
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
