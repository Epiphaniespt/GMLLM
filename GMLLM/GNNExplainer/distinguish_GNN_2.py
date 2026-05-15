import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.data import Subset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

class GCNWithBehavior(nn.Module):
    def __init__(self, name_vocab_size, type_vocab_size, behavior_dim, hidden_dim=64, num_classes=2):
        super().__init__()
        self.name_emb = nn.Embedding(name_vocab_size, 64)
        self.type_emb = nn.Embedding(type_vocab_size, 16)
        input_dim = 64 + 16 + behavior_dim
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, data):
        name_feat = self.name_emb(data.x_names)
        type_feat = self.type_emb(data.x_types)
        behavior_feat = data.x_behaviors.float()
        x = torch.cat([name_feat, type_feat, behavior_feat], dim=1)

        x = self.dropout(self.conv1(x, data.edge_index, data.edge_weight).relu())
        x = self.dropout(self.conv2(x, data.edge_index, data.edge_weight).relu())

        if x.shape[0] != data.batch.shape[0]:
            min_len = min(x.shape[0], data.batch.shape[0])
            x = x[:min_len]
            batch = data.batch[:min_len]
        else:
            batch = data.batch

        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.classifier(x)

def load_dict(path):
    with open(path, 'r') as f:
        return json.load(f)


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
        total_loss += loss.item() * data.num_graphs

    return total_loss / total, correct / total

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=[0, 1], zero_division=0)

    malicious_f1 = f1[1]
    malicious_precision = p[1]
    malicious_recall = r[1]

    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    benign_acc = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"[Validation] Overall Acc: {acc:.4f} | Malicious F1: {malicious_f1:.4f}")
    print(
        f" └─ Malicious Metrics: Precision: {malicious_precision:.4f} | Recall: {malicious_recall:.4f} (TP:{tp}, FN:{fn})")
    print(f" └─ Benign Accuracy..: {benign_acc:.4f} (TN:{tn}, FP:{fp})")

    return malicious_f1, acc, malicious_recall

