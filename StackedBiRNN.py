import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import gensim.downloader as api

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ----- Dummy Dataset (Replace with your actual data loading) -----
class TextDataset(Dataset):
    def __init__(self, data, word2idx, max_len=50):
        self.data = data
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def vectorize(self, sentence):
        # Convert words to indices; unknown words default to 0.
        tokens = sentence.lower().split()
        idxs = [self.word2idx.get(tok, 0) for tok in tokens]
        # Pad or truncate
        if len(idxs) < self.max_len:
            idxs += [0] * (self.max_len - len(idxs))
        else:
            idxs = idxs[:self.max_len]
        return torch.tensor(idxs, dtype=torch.long)

    def __getitem__(self, idx):
        sentence, label = self.data[idx]
        return self.vectorize(sentence), torch.tensor(label, dtype=torch.long)

# ----- Model Definition -----
class StackedBiRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size,
                 num_layers=2, dropout=0.5, bidirectional=True, pretrained_embeddings=None, freeze_embeddings=False, rnn_type='LSTM'):
        super(StackedBiRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.rnn_type = rnn_type.upper()
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               dropout=dropout if num_layers > 1 else 0,
                               bidirectional=bidirectional,
                               batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=embedding_dim,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout if num_layers > 1 else 0,
                              bidirectional=bidirectional,
                              batch_first=True)
        else:
            raise ValueError("Unsupported rnn_type. Use 'LSTM' or 'GRU'.")

        rnn_out_dim = hidden_size * 2 if bidirectional else hidden_size
        # Global max pooling layer over the time dimension.
        self.fc = nn.Linear(rnn_out_dim, output_size)

    def forward(self, x):
        # x: [batch, seq_len]
        emb = self.embedding(x)  # [batch, seq_len, embedding_dim]
        if self.rnn_type == 'LSTM':
            rnn_out, _ = self.rnn(emb)
        else:
            rnn_out, _ = self.rnn(emb)
        pooled = torch.max(rnn_out, dim=1)[0]  # Global max pooling
        out = self.fc(pooled)
        return out

# ----- Training and Evaluation functions -----
def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

# ----- Main function to run training -----
def main():
    # For reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained Word2Vec Google News 300 embeddings from gensim
    print("Loading Google News Word2Vec model (this may take a while)...")
    word2vec_model = api.load("word2vec-google-news-300")
    embedding_dim = 300

    # Define your vocabulary. Note: Use lowercase for consistency.
    word2idx = {"<pad>": 0, "example": 1, "sentence": 2, "this": 3, "is": 4, "a": 5, "test": 6}
    vocab_size = len(word2idx)

    # Build the embedding matrix for words in word2idx.
    embedding_matrix = torch.zeros(vocab_size, embedding_dim)
    for word, idx in word2idx.items():
        # Check gensim model with lowercase word (or adjust based on your tokenization)
        if word in word2vec_model:
            embedding_matrix[idx] = torch.tensor(word2vec_model[word])
        else:
            # Initialize unknown words with random vectors
            embedding_matrix[idx] = torch.randn(embedding_dim)
    
    hidden_size = 64
    output_size = 2  # Binary classification
    num_layers = 2
    dropout = 0.5

    # Dummy data (sentence, label): Replace with your actual data.
    train_data = [
        ("This is a test", 0),
        ("Example sentence", 1),
        ("This is example", 0),
        ("Test sentence example", 1)
    ]
    dev_data = [
        ("Example test", 1),
        ("This is sentence", 0)
    ]

    train_dataset = TextDataset(train_data, word2idx, max_len=10)
    dev_dataset = TextDataset(dev_data, word2idx, max_len=10)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=2)

    # Instantiate the model. Now using the pre-trained embedding matrix.
    model = StackedBiRNN(vocab_size=vocab_size,
                         embedding_dim=embedding_dim,
                         hidden_size=hidden_size,
                         output_size=output_size,
                         num_layers=num_layers,
                         dropout=dropout,
                         bidirectional=True,
                         pretrained_embeddings=embedding_matrix,
                         freeze_embeddings=False,
                         rnn_type='LSTM').to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 20
    best_dev_acc = 0.0
    best_epoch = 0
    best_model_state = None

    for epoch in range(1, num_epochs+1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Dev Loss={dev_loss:.4f}, Dev Acc={dev_acc:.4f}")
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch
            best_model_state = model.state_dict()  # save best model

    print(f"Best epoch: {best_epoch} with Dev Acc={best_dev_acc:.4f}")
    # Save the best model state if needed
    torch.save(best_model_state, "best_model.pt")

if __name__ == "__main__":
    main()