import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, query, key, value, mask=None):
        # Get batch size
        batch_size = query.shape[0]
        
        # Split the embedding into self.heads different pieces
        query = self.query(query).view(batch_size, -1, self.heads, self.head_dim)
        key = self.key(key).view(batch_size, -1, self.heads, self.head_dim)
        value = self.value(value).view(batch_size, -1, self.heads, self.head_dim)
        
        # Transpose to get dimensions [batch_size, heads, seq_length, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Calculate attention scores
        energy = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # Apply softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)
        
        # Multiply attention weights with values
        out = torch.matmul(attention, value)
        
        # Reshape and pass through final linear layer
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)
        out = self.fc_out(out)
        
        return out
