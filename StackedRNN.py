import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gensim.downloader as api
import torch.nn as nn
import torch.optim as optim
import random

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

EMBEDDING_DIM = 300
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.6
BIDIRECTIONAL = True
CELL_TYPE = 'LSTM'
NUM_CLASSES = 2
LEARNING_RATE = 5e-4
BATCH_SIZE = 64
NUM_EPOCHS = 10
FREEZE_EMBEDDINGS = True

EMBEDDING_MODEL_PATH = "glove_embeddings.npy"

if os.path.exists(EMBEDDING_MODEL_PATH):
    print(f" Φόρτωση αποθηκευμένων embeddings από {EMBEDDING_MODEL_PATH}...")
    embedding_matrix = np.load(EMBEDDING_MODEL_PATH)  # Φόρτωση NumPy αρχείου
else:
    print("Κατέβασμα και αποθήκευση GloVe embeddings...")
    word2vec = api.load("glove-wiki-gigaword-300")

    VOCAB_SIZE = len(word2vec.index_to_key)
    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM), dtype=np.float32)

    for i, word in enumerate(word2vec.index_to_key):
        embedding_matrix[i] = word2vec[word]

    np.save(EMBEDDING_MODEL_PATH, embedding_matrix)  # Αποθήκευση ως NumPy αρχείο
    print(f"Τα embeddings αποθηκεύτηκαν στο αρχείο {EMBEDDING_MODEL_PATH}")

embedding_weights = torch.tensor(embedding_matrix, dtype=torch.float)

class StackedBiRNN(nn.Module):
    def __init__(self, embedding_weights, hidden_dim, num_layers, dropout, bidirectional, cell_type, num_classes):
        super(StackedBiRNN, self).__init__()
        num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=FREEZE_EMBEDDINGS)

        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embedding_weights.size(1),
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embedding_weights.size(1),
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError("Unsupported cell type. Use 'LSTM' or 'GRU'.")

        self.fc = nn.Linear(hidden_dim * num_directions, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)  
        rnn_out, _ = self.rnn(embedded)  
        pooled, _ = torch.max(rnn_out, dim=1)  
        logits = self.fc(pooled)  
        return logits

def get_dummy_data(num_samples, seq_len):
    X = torch.randint(0, embedding_weights.size(0), (num_samples, seq_len))
    y = torch.randint(0, NUM_CLASSES, (num_samples,))
    return X, y

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StackedBiRNN(
        embedding_weights=embedding_weights,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
        cell_type=CELL_TYPE,
        num_classes=NUM_CLASSES
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    X_train, y_train = get_dummy_data(1000, 50)
    X_dev, y_dev = get_dummy_data(200, 50)

    train_dataset = TensorDataset(X_train, y_train)
    dev_dataset = TensorDataset(X_dev, y_dev)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE)

    print("\n Training Stacked Bidirectional RNN ")
    best_dev_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clipping gradients
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        avg_train_loss = epoch_loss / len(train_dataset)
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)
        print(f" Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Dev Loss = {dev_loss:.4f}, Dev Acc = {dev_acc:.4f}")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_epoch = epoch
            best_model_state = model.state_dict()

    print(f"\n Best model at Epoch {best_epoch} με Dev Loss = {best_dev_loss:.4f}")
    model.load_state_dict(best_model_state)
    return model

if __name__ == '__main__':
    train_model()
