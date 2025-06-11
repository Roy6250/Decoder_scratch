# Decoder from Scratch

This repository contains an implementation of a **Transformer Decoder from scratch** using PyTorch. It's a minimal and educational re-creation of the core architecture that powers modern large language models (LLMs), like GPT.

---

##  Features

- Manual implementation of the Transformer Decoder block
- Self-attention mechanism
- Positional encoding
- Masking for auto-regressive generation
- Clean and minimal PyTorch code
- Great for learning and experimentation

---

## Architecture

This implementation includes:

- **Token Embedding Layer**
- **Positional Encoding**
- **Multi-head Self Attention**
- **Feedforward Neural Network**
- **Layer Normalization & Residual Connections**
- **Causal Masking (for left-to-right prediction)**

---

## Training Loss
![Image](https://github.com/user-attachments/assets/c9263405-5718-4fa3-b60f-679260110007)

## Inference Sample

```python
input = "Whell's the couse,"
output = """Whell's the couse, hear me speak.
All:
Speak, speak.
First Citizen:
You are all resolved rather to die than to famish?
All:
Resolved. resolved.
First Citizen:"""
```
### Attention Weights plot
![Image](https://github.com/user-attachments/assets/0c0f229d-fe90-484f-b500-0b0f8674bb61)

### Inspiration

This project is inspired by [Andrej Karpathy's](https://github.com/karpathy) excellent work on [nanoGPT](https://github.com/karpathy/nanoGPT), a minimal, educational implementation of a GPT-like language model.




