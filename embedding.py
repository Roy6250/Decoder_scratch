import torch
import torch.nn as nn
import torch.nn.functional as f
import math

class InputEmbedding(nn.Module):

    def __init__(self,vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.apply(self._init_weights)


    def forward(self, x):
        # x is of shape (batch_size, seq_len)
        # We need to convert it to (batch_size, seq_len, d_model)
        x = self.embedding(x)
        # Scale the embeddings by sqrt(d_model)
        x = x * math.sqrt(self.d_model)
        return x
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_len,dropout):
        super().__init__()

        # Initialize
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len,d_model)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)


        # Apply sine to even indices
        # Broadcast position to match the shape of div_term
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Since sentences can be of variable length, we slice the positional encoding
        # to match the input sequence length.
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)


# ----- Parameters -----
vocab_size = 10      # Assume your vocab has 100 tokens
d_model = 4            # Embedding dimension
seq_len = 4            # Number of tokens (4-word sentence)
dropout = 0.1

# ----- Dummy input -----
# Batch size = 1, sentence of 4 tokens
input_ids = torch.tensor([[5, 6, 1, 2]])  # shape: (1, 4)

# ----- Create modules -----
input_embed = InputEmbedding(vocab_size=vocab_size, d_model=d_model)
print(input_embed.embedding.weight)  # Check embedding weight shape
pos_embed = PositionalEmbedding(d_model=d_model, seq_len=seq_len, dropout=dropout)

# ----- Forward pass -----
word_embeddings = input_embed(input_ids)        # shape: (1, 4, d_model)
final_embeddings = pos_embed(word_embeddings)   # shape: (1, 4, d_model)

# ----- Output -----
print("Word Embeddings:\n", word_embeddings)
print("Final Embeddings (with Positional Encoding):\n", final_embeddings)
