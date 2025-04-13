import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import math
import random
from models import SentimentClassifier

# ---------------------------
# Configuration & Vocabulary
# ---------------------------
max_seq_length = 128
# A small vocabulary for demonstration. In practice you might build a larger vocab.
vocab = {
    "<pad>": 0, "<unk>": 1, "<cls>": 2, 
    "good": 3, "bad": 4, "great": 5, "terrible": 6, 
    ":happy:": 7, ":sad:": 8  
}
vocab_size = len(vocab)

# ---------------------------
# Tokenization and Preprocessing
# ---------------------------
def tokenize(text):
    text = text.lower().replace("😊", ":happy:").replace("😞", ":sad:")
    tokens = text.split()
    return [vocab["<cls>"]] + [vocab.get(t, vocab["<unk>"]) for t in tokens]

def pad_sequence(token_ids):
    current_length = len(token_ids)
    if current_length > max_seq_length:
        return token_ids[:max_seq_length]
    else:
        return token_ids + [vocab["<pad>"]] * (max_seq_length - current_length)

# ---------------------------
# Model Architecture
# ---------------------------
class Embeddings(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(self.token_emb(x) + self.pos_emb(positions))

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
        
        Q = self.Wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out(context)

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
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

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

class SentimentDataset(Dataset):
    def __init__(self, csv_file, dataset_type="airline"):
        """
        Args:
            csv_file: path to the CSV file
            dataset_type: either 'airline' or 'sentiment140'
        """
        self.data = pd.read_csv(csv_file)
        self.dataset_type = dataset_type

        if dataset_type == "airline":
            self.label_map = {"negative": 0, "neutral": 1, "positive": 2}
            self.text_col = "text"
            self.label_col = "airline_sentiment"
        elif dataset_type == "sentiment140":
            self.label_map = {0: 0, 2: 1, 4: 2}  
            self.text_col = "text" if "text" in self.data.columns else "tweet"
            self.label_col = "sentiment"
        else:
            raise ValueError("Unknown dataset type")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row[self.text_col])
        label = self.label_map[row[self.label_col]]
        token_ids = tokenize(text)
        token_ids = pad_sequence(token_ids)
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ---------------------------
# Data Loading
# ---------------------------
class SentimentDataset(Dataset):
    def __init__(self, csv_file, dataset_type='airline'):
        """
        dataset_type: 'airline' or 'sentiment140'
        """
        if dataset_type == 'airline':
            df = pd.read_csv(csv_file)
            df = df[['text', 'airline_sentiment']]
            label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
            df['label'] = df['airline_sentiment'].map(label_map)
            df = df.dropna(subset=['text', 'label'])

        elif dataset_type == 'sentiment140':
            df = pd.read_csv(csv_file, encoding='latin-1', header=None)
            df = df[[0, 5]]
            df.columns = ['label', 'text']
            label_map = {0: 0, 2: 1, 4: 2}
            df['label'] = df['label'].map(label_map)
            df = df.dropna(subset=['text', 'label'])

        else:
            raise ValueError("Invalid dataset_type. Use 'airline' or 'sentiment140'.")

        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        label = int(row['label'])
        token_ids = tokenize(text)  # assumes your tokenize() returns list of token ids
        token_ids = pad_sequence(token_ids)  # pad to fixed length
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = torch.stack(texts)
    labels = torch.stack(labels)
    return texts,labels
# ---------------------------
# Training Loop
# ---------------------------
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        mask = (inputs != vocab["<pad>"]).unsqueeze(1).unsqueeze(2)  
        
        optimizer.zero_grad()
        logits = model(inputs, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += inputs.size(0)
        
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            mask = (inputs != vocab["<pad>"]).unsqueeze(1).unsqueeze(2)
            logits = model(inputs, mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += inputs.size(0)
            
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# ---------------------------
# Main Training Routine
# ---------------------------
def main():
    airline_csv = "Tweets.csv"
    sentiment140_csv = "training.1600000.processed.noemoticon.csv"

    dataset1 = SentimentDataset(airline_csv, dataset_type="airline")
    dataset2 = SentimentDataset(sentiment140_csv, dataset_type="sentiment140") 

    combined_dataset = ConcatDataset([dataset1, dataset2])
    total_size = len(combined_dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    split = int(0.8 * total_size)
    train_indices, val_indices = indices[:split], indices[split:]

    train_dataset = torch.utils.data.Subset(combined_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(combined_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

 

    model = SentimentClassifier(d_model=256, num_heads=8, ff_dim=512, num_layers=4, dropout=0.1)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device="cpu")
        val_loss, val_acc = evaluate(model, val_loader, criterion, device="cpu")
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Acc {val_acc:.4f}")

    torch.save(model.state_dict(), "sentiment_transformer_model.pth")


if __name__ == "__main__":
    main()
