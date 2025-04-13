# models.py
import torch
from torch import nn
import math

# Configuration
max_seq_length = 128
vocab = {
    "<pad>": 0, "<unk>": 1, "<cls>": 2, 
    "good": 3, "bad": 4, "great": 5, "terrible": 6, 
    ":happy:": 7, ":sad:": 8  # Add more tokens as needed
}
vocab_size = len(vocab)

# Tokenization and preprocessing
def tokenize(text):
    # Simple tokenizer (replace with your logic for social media text)
    text = text.lower().replace("😊", ":happy:").replace("😞", ":sad:")
    tokens = text.split()  # Improve with regex for hashtags/mentions
    return [vocab["<cls>"]] + [vocab.get(t, vocab["<unk>"]) for t in tokens]

def pad_sequence(token_ids):
    # Ensure total length is max_seq_length (including [CLS])
    current_length = len(token_ids)
    if current_length > max_seq_length:
        return token_ids[:max_seq_length]
    else:
        return token_ids + [vocab["<pad>"]] * (max_seq_length - current_length)

# Embedding Layer
class Embeddings(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(self.token_emb(x) + self.pos_emb(positions))

# Multi-Head Attention (Fixed Head Splitting)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Split into heads
        Q = self.Wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out(context)

# Encoder Layer (Added Dropout)
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and dropout
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward with residual connection and dropout
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# Full Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.emb = Embeddings(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, ff_dim, dropout) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# Sentiment Classifier
class SentimentClassifier(nn.Module):
    def __init__(self, d_model=256, num_heads=8, ff_dim=512, num_layers=4, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, ff_dim, dropout)
        self.head = nn.Linear(d_model, 3)  # 3 classes
        
    def forward(self, x, mask=None):
        enc_out = self.encoder(x, mask)
        return self.head(enc_out[:, 0, :])  # [CLS] token

# Initialize model
if __name__ == "__main__":
    model = SentimentClassifier()
    torch.save(model.state_dict(), "sentiment_model.pth")