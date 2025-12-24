# ================================
# TASK 2: Deep Learning - Text Classification
# Sentiment Analysis on small dataset
# ================================

print("Task 2 started...\n")

# -------- 1. Import Libraries --------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# -------- 2. Sample Dataset --------
data = [
    ("I love this movie", 1),
    ("This is an amazing film", 1),
    ("I hate this movie", 0),
    ("This film is terrible", 0),
    ("Fantastic acting and story", 1),
    ("Worst movie ever", 0),
    ("I enjoyed the movie", 1),
    ("I do not like this film", 0)
]

sentences, labels = zip(*data)
labels = torch.tensor(labels)

# -------- 3. Tokenization & Vocabulary --------
# Simple word-level tokenization
all_words = set(word.lower() for sent in sentences for word in sent.split())
word2idx = {word: idx+1 for idx, word in enumerate(all_words)}  # 0 reserved for padding
vocab_size = len(word2idx) + 1

def encode_sentence(sent):
    return [word2idx[word.lower()] for word in sent.split()]

encoded_sentences = [encode_sentence(s) for s in sentences]

# Pad sequences
max_len = max(len(s) for s in encoded_sentences)
def pad_sequence(seq, max_len):
    return seq + [0]*(max_len - len(seq))

padded_sentences = [pad_sequence(s, max_len) for s in encoded_sentences]
padded_sentences = torch.tensor(padded_sentences)

# -------- 4. Train-test split --------
X_train, X_test, y_train, y_test = train_test_split(
    padded_sentences, labels, test_size=0.25, random_state=42
)

# -------- 5. Create Dataset & DataLoader --------
class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SentimentDataset(X_train, y_train)
test_dataset = SentimentDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# -------- 6. Define Model --------
class SimpleNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim*max_len, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)             # (batch, max_len, embed_dim)
        x = x.view(x.size(0), -1)        # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Hyperparameters
embed_dim = 10
hidden_dim = 16
output_dim = 2
lr = 0.01
epochs = 30

model = SimpleNN(vocab_size, embed_dim, hidden_dim, output_dim)

# -------- 7. Loss and Optimizer --------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -------- 8. Training Loop --------
train_losses = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# -------- 9. Plot Training Loss --------
plt.plot(range(1, epochs+1), train_losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# -------- 10. Evaluation --------
model.eval()
all_preds = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.numpy())

y_test_np = y_test.numpy()
print("\nTest Accuracy:", accuracy_score(y_test_np, all_preds))
print("\nClassification Report:\n", classification_report(y_test_np, all_preds))

# -------- 11. Save Model --------
torch.save(model.state_dict(), "sentiment_model.pth")
print("\nModel saved as sentiment_model.pth")

# -------- 12. Sample Inference --------
sample_sentence = "I really like this movie"
sample_encoded = pad_sequence(encode_sentence(sample_sentence), max_len)
sample_tensor = torch.tensor([sample_encoded])

model.eval()
with torch.no_grad():
    output = model(sample_tensor)
    pred = torch.argmax(output, dim=1).item()
    print("\nSample Sentence Prediction:", "Positive" if pred==1 else "Negative")

print("\nTask 2 completed successfully.")
