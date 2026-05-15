import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import argparse
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from generate_graph_data_fromJson import CallGraphDatasetFull_Lazy
from torch.utils.data import ConcatDataset
from torch.utils.data import Subset
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class GCNWithBehavior(nn.Module):
    def __init__(self, name_vocab_size, type_vocab_size, behavior_dim, hidden_dim=64, num_classes=2):
        super().__init__()
        self.name_emb = nn.Embedding(name_vocab_size, 64)
        self.type_emb = nn.Embedding(type_vocab_size, 16)
        input_dim = 64 + 16 + behavior_dim  # name + type + behaviors
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, data):
        name_feat = self.name_emb(data.x_names)
        type_feat = self.type_emb(data.x_types)
        behavior_feat = data.x_behaviors.float()
        x = torch.cat([name_feat, type_feat, behavior_feat], dim=1)
        x = self.dropout(self.conv1(x, data.edge_index).relu())
        x = self.dropout(self.conv2(x, data.edge_index).relu())

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN on processed call graphs.")
    parser.add_argument("--vocab-dir", required=True,
                        help="Directory containing name2idx.json/type2idx.json/edge_type2idx.json/behavior2idx.json")
    parser.add_argument("--benign-root", required=True, help="Root directory of benign_call (<root>/*/call_graph.json)")
    parser.add_argument("--malicious-root", required=True, help="Root directory of malicious_call")
    parser.add_argument("--benign-out", required=True, help="Processed output dir for benign graphs")
    parser.add_argument("--malicious-out", required=True, help="Processed output dir for malicious graphs")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    SEED = 42
    set_seed(SEED)

    name2idx = load_dict(str(Path(args.vocab_dir) / "name2idx.json"))
    type2idx = load_dict(str(Path(args.vocab_dir) / "type2idx.json"))
    behavior2idx = load_dict(str(Path(args.vocab_dir) / "behavior2idx.json"))
    edge_type2idx = load_dict(str(Path(args.vocab_dir) / "edge_type2idx.json"))

    normal_dataset = CallGraphDatasetFull_Lazy(
        root_dir=args.benign_root,
        output_dir=args.benign_out,
        name2idx=name2idx,
        type2idx=type2idx,
        behavior2idx=behavior2idx,
        edge_type2idx=edge_type2idx,
        fixed_label=0
    )
    malicious_dataset = CallGraphDatasetFull_Lazy(
        root_dir=args.malicious_root,
        output_dir=args.malicious_out,
        name2idx=name2idx,
        type2idx=type2idx,
        behavior2idx=behavior2idx,
        edge_type2idx=edge_type2idx,
        fixed_label=1
    )

    # Split 80/20 within each class
    normal_train_size = int(0.8 * len(normal_dataset))
    normal_val_size = len(normal_dataset) - normal_train_size
    normal_train, normal_val = random_split(normal_dataset, [normal_train_size, normal_val_size])

    malicious_train_size = int(0.8 * len(malicious_dataset))
    malicious_val_size = len(malicious_dataset) - malicious_train_size
    malicious_train, malicious_val = random_split(malicious_dataset, [malicious_train_size, malicious_val_size])

    train_dataset = ConcatDataset([normal_train, malicious_train])
    val_dataset = ConcatDataset([normal_val, malicious_val])

    assert len(train_dataset) > 0, "Empty train set."
    assert len(val_dataset) > 0, "Empty val set."

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device(args.device)
    model = GCNWithBehavior(
        name_vocab_size=len(name2idx),
        type_vocab_size=len(type2idx),
        behavior_dim=len(behavior2idx)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0
    history = {'train_loss': [], 'train_acc': [], 'val_f1': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, device)
        val_f1, acc, malicious_recall = val_metrics

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(acc)

        print(
            f"Epoch {epoch:03d} | Loss {train_loss:.4f} | TrainAcc {train_acc:.4f} | ValF1 {val_f1:.4f} | ValAcc {acc:.4f} | MalRecall {malicious_recall:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), str(Path(args.malicious_out).parent / "best_model.pt"))

    print("Training done. Best Val F1:", best_val_f1)
