import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
import gensim.downloader as api
import re

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
categories = ['comp.graphics', 'sci.space']
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))
texts, labels = data.data, data.target

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Load pretrained Word2Vec embeddings
word2vec = api.load("word2vec-google-news-300")
embedding_dim = 300

def preprocess_text(text, vocab):
    words = re.sub(r'[^a-zA-Z]', ' ', text.lower()).split()
    return [vocab.get(word, vocab['UNK']) for word in words]

# Create vocabulary from Word2Vec
vocab = {'PAD': 0, 'UNK': 1}
vocab.update({word: idx + 2 for idx, word in enumerate(word2vec.index_to_key)})
embedding_matrix = np.zeros((len(vocab), embedding_dim))
embedding_matrix[1] = np.mean(word2vec.vectors, axis=0)
for word, idx in vocab.items():
    if word in word2vec:
        embedding_matrix[idx] = word2vec[word]

# Convert text to sequences
X_train_seq = [preprocess_text(text, vocab) for text in X_train]
X_val_seq = [preprocess_text(text, vocab) for text in X_val]
X_test_seq = [preprocess_text(text, vocab) for text in X_test]

# Pad sequences
def pad_sequences(sequences, max_len):
    return [seq[:max_len] + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in sequences]

max_length = 200
X_train_seq = pad_sequences(X_train_seq, max_length)
X_val_seq = pad_sequences(X_val_seq, max_length)
X_test_seq = pad_sequences(X_test_seq, max_length)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.long).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val_seq, dtype=torch.long).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Create DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the Stacked Bidirectional RNN Model
class StackedBiRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout, pretrained_embeddings):
        super(StackedBiRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        rnn_out = self.global_max_pool(rnn_out.permute(0, 2, 1)).squeeze(2)
        output = self.fc(self.dropout(rnn_out))
        return output

# Model Hyperparameters
hidden_dim = 128
num_layers = 2
bidirectional = True
dropout = 0.5
output_dim = len(set(labels))
pretrained_embeddings = torch.tensor(embedding_matrix, dtype=torch.float).to(device)

# Initialize model
model = StackedBiRNN(len(vocab), embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout, pretrained_embeddings).to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training function
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (y_pred.argmax(dim=1) == y_batch).sum().item()
        val_correct = sum((model(X).argmax(dim=1) == y).sum().item() for X, y in val_loader)
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Train Acc = {correct/len(train_dataset):.4f}, Val Acc = {val_correct/len(val_dataset):.4f}")

# Train the model
train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10)

# Evaluate on test set
def evaluate_model(model, test_loader):
    model.eval()
    correct = sum((model(X).argmax(dim=1) == y).sum().item() for X, y in test_loader)
    print(f"Test Accuracy: {correct/len(test_dataset):.4f}")

evaluate_model(model, test_loader)
# Early stopping based on development (validation) data
best_val_acc = 0.0
best_epoch = 0
best_model_state = None
num_epochs = 10  # Chosen number of epochs based on preliminary experiments

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct_train += (outputs.argmax(dim=1) == y_batch).sum().item()

    train_acc = correct_train / len(train_dataset)

    model.eval()
    correct_val = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            correct_val += (outputs.argmax(dim=1) == y_batch).sum().item()
    val_acc = correct_val / len(val_dataset)

    print(f"Epoch {epoch+1}: Loss = {running_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        best_model_state = model.state_dict()

print(f"\nBest Epoch: {best_epoch} with Validation Accuracy: {best_val_acc:.4f}")

# Load the best model for final evaluation on test data
model.load_state_dict(best_model_state)
model.eval()
correct_test = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        correct_test += (outputs.argmax(dim=1) == y_batch).sum().item()
test_acc = correct_test / len(test_dataset)
print(f"Test Accuracy using best model: {test_acc:.4f}")

# Print hyperparameter details and selection rationale
print("\nHyperparameters used:")
print(f"  - Hidden Dimension: {hidden_dim}")
print(f"  - Number of Layers: {num_layers} (Stacked layers to enhance learning of complex features)")
print(f"  - Bidirectional: {bidirectional} (captures context from both directions)")
print(f"  - Dropout: {dropout} (used to prevent overfitting)")
print("  - Optimizer: Adam with learning rate 0.001 (faster convergence compared to SGD)")
print(f"  - Epochs: {num_epochs} (selected based on performance on development data)")