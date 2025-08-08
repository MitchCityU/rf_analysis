import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Configuration
SAMPLES_DIR = Path("/mnt/c/users/mitch/downloads/ai_classifier/dwingeloo/samples")
ANNOT_PATH = Path("/mnt/c/users/mitch/downloads/ai_classifier/libriiq_dwingeloo/dwingeloo/annot.json")
CLS_MAP_PATH = Path("/mnt/c/users/mitch/downloads/ai_classifier/libriiq_dwingeloo/dwingeloo/cls_map.json")
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3

# Load annotations
with open(ANNOT_PATH) as f:
    annotations = json.load(f)

with open(CLS_MAP_PATH) as f:
    cls_map = json.load(f)

label_map = cls_map

print("First 5 annotations:")
for i, (sample_id, meta) in enumerate(annotations.items()):
    print(f"{i}: {sample_id} -> {meta}")
    if i >= 4:
        break

# Build dataset
data = []
for sample_id, meta in annotations.items():
    npy_name = f"{sample_id}.npy"
    full_path = SAMPLES_DIR / npy_name
    label = meta.get("Satellite")
    if full_path.exists() and label in label_map:
        data.append((str(full_path), label_map[label]))

print(f"Loaded {len(data)} samples")
if not data:
    raise RuntimeError("No matching samples found.")

train_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42, stratify=[lbl for _, lbl in data]
)

class IQDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        iq = np.load(path).astype(np.float32)  # (2, 240000)
        iq = torch.from_numpy(iq)
        iq = (iq - iq.mean()) / (iq.std() + 1e-6)
        return iq, torch.tensor(label, dtype=torch.long)

train_loader = DataLoader(IQDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(IQDataset(test_data), batch_size=BATCH_SIZE)

# CNN Model
class DeepIQCNN(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(self.net(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(cls_map)
model = DeepIQCNN(num_classes=num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

print(f"Accuracy: {100.0 * correct / total:.2f}%")
