import os
import json
import torch
from pathlib import Path
import argparse
from tqdm import tqdm
from torch_geometric.data import Dataset
from torch_geometric.data import Data
class CallGraphDatasetFull_Lazy(Dataset):
    def __init__(self, root_dir, output_dir=None, fixed_label=None,
                 name2idx=None, type2idx=None, edge_type2idx=None, behavior2idx=None,
                 transform=None, pre_transform=None):
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir) if output_dir else self.root_dir / 'processed'
        self.fixed_label = fixed_label
        assert fixed_label is not None, "Must provide fixed_label"
        assert name2idx and type2idx, "Must provide name2idx and type2idx"
        self.name2idx = name2idx
        self.type2idx = type2idx
        self.edge_type2idx = edge_type2idx or {}
        self.behavior2idx = behavior2idx or {}
        super().__init__(self.output_dir, transform=transform, pre_transform=pre_transform)
        self.index_file = self.output_dir / 'index.json'
        if not self.index_file.exists():
            self.process()
        with open(self.index_file, 'r') as f:
            self.graph_paths = json.load(f)
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['index.json']
    def download(self):
        pass
    def process(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.edge_type2idx or not self.behavior2idx:
            edge_types = set()
            behaviors = set()
            all_folders = os.listdir(self.root_dir)
            for folder in tqdm(all_folders, desc=f"Scanning {self.root_dir.name}"):
                call_graph_file = self.root_dir / folder / 'call_graph.json'
                if not call_graph_file.exists():
                    continue
                try:
                    with open(call_graph_file, 'r', encoding='utf-8', errors='ignore') as f:
                        graph = json.load(f)
                    for link in graph.get('links', []):
                        edge_types.add(link.get('edge_type', 'unknown'))
                    for node in graph.get('nodes', []):
                        for b in node.get('behaviors', []):
                            behaviors.add(b)
                except Exception as e:
                    print(f"[ERROR] Failed to scan {folder}: {e}")
            self.edge_type2idx = {t: i for i, t in enumerate(sorted(edge_types))}
            self.behavior2idx = {b: i for i, b in enumerate(sorted(behaviors))}
        graph_paths = []
        for folder in tqdm(os.listdir(self.root_dir), desc=f"Processing {self.root_dir.name}"):
            call_graph_file = self.root_dir / folder / 'call_graph.json'
            if not call_graph_file.exists():
                continue
            try:
                with open(call_graph_file, 'r', encoding='utf-8', errors='ignore') as f:
                    graph = json.load(f)
                label = self.fixed_label
                name_ids, type_ids, behavior_feats, node_raw_attrs = [], [], [], []
                for idx, node in enumerate(graph.get('nodes', [])):
                    name = node.get('qualified_name') or node.get('name', 'unknown_name')
                    node_type = node.get('type', 'unknown_type')
                    behaviors = node.get('behaviors', [])
                    name_ids.append(self.name2idx.get(name, self.name2idx['unknown_name']))
                    type_ids.append(self.type2idx.get(node_type, self.type2idx['unknown_type']))
                    behavior_vec = torch.zeros(len(self.behavior2idx))
                    for b in behaviors:
                        if b in self.behavior2idx:
                            behavior_vec[self.behavior2idx[b]] = 1
                    behavior_feats.append(behavior_vec)
                    node_raw_attrs.append({
                        'id': node.get('id', f'missing_id_{idx}'),
                        'name': name,
                        'type': node_type,
                        'file': node.get('file', ''),
                        'behaviors': behaviors
                    })
                assert len(name_ids) == len(type_ids) == len(behavior_feats), "Node features are misaligned"
                id_map = {node['id']: i for i, node in enumerate(graph.get('nodes', []))}
                edges, edge_attrs = [], []
                for link in graph.get('links', []):
                    src = id_map.get(link.get('source'))
                    tgt = id_map.get(link.get('target'))
                    edge_type = link.get('edge_type', 'unknown')
                    if src is not None and tgt is not None:
                        edges.append([src, tgt])
                        edge_attrs.append(self.edge_type2idx.get(edge_type, -1))
                if len(name_ids) == 0 or len(behavior_feats) == 0:
                    print(f"[WARNING] Skipping {folder}: empty node list or no behaviors.")
                    continue
                data = Data(
                    x_names=torch.tensor(name_ids, dtype=torch.long),
                    x_types=torch.tensor(type_ids, dtype=torch.long),
                    x_behaviors=torch.stack(behavior_feats),
                    edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.tensor(edge_attrs, dtype=torch.long),
                    y=torch.tensor([label], dtype=torch.long),
                    name=folder,
                    node_raw_attrs=node_raw_attrs,
                    graph_raw=graph,
                    num_nodes=len(name_ids)
                )
                save_path = self.output_dir / f"{folder}.pt"
                torch.save(data, save_path)
                graph_paths.append(str(save_path.relative_to(self.output_dir)))
            except Exception as e:
                print(f"[ERROR] Failed to process {folder}: {e}")
        with open(self.output_dir / 'index.json', 'w') as f:
            json.dump(graph_paths, f, indent=2)
    def len(self):
        return len(self.graph_paths)
    def get(self, idx):
        rel_path = self.graph_paths[idx]
        return torch.load(self.output_dir / rel_path, map_location='cpu' ,weights_only=False)
def build_global_vocab(root_dirs):
    name_set, type_set, edge_type_set, behavior_set = set(), set(), set(), set()
    for root_dir in root_dirs:
        for folder in os.listdir(root_dir):
            json_path = Path(root_dir) / folder / "call_graph.json"
            if not json_path.exists():
                continue
            try:
                with open(json_path, 'r') as f:
                    graph = json.load(f)
                for node in graph.get("nodes", []):
                    name = node.get("qualified_name") or node.get("name", "unknown_name")
                    type_ = node.get("type", "unknown_type")
                    name_set.add(name)
                    type_set.add(type_)
                    for b in node.get("behaviors", []):
                        behavior_set.add(b)
                for link in graph.get("links", []):
                    edge_type_set.add(link.get("edge_type", "unknown"))
            except Exception as e:
                print(f"[ERROR] Failed to parse {json_path}: {e}")
    name_list = sorted(name_set - {'unknown_name'})
    name2idx = {'unknown_name': 0, **{name: i + 1 for i, name in enumerate(name_list)}}
    type_list = sorted(type_set - {'unknown_type'})
    type2idx = {'unknown_type': 0, **{t: i + 1 for i, t in enumerate(type_list)}}
    edge_type2idx = {et: i for i, et in enumerate(sorted(edge_type_set))}
    behavior2idx = {b: i for i, b in enumerate(sorted(behavior_set))}
    return name2idx, type2idx, edge_type2idx, behavior2idx
def clean_dir(path):
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith(".pt") or f == "index.json":
                os.remove(Path(path) / f)

def _cli_main_():
    parser = argparse.ArgumentParser(description="Generate PyG data from call_graph.json folders and build vocabs.")
    parser.add_argument("--normal-root", required=True, help="Directory with benign_call/*/call_graph.json")
    parser.add_argument("--malicious-root", required=True, help="Directory with malicious_call/*/call_graph.json")
    parser.add_argument("--normal-out", required=True, help="Output dir for processed benign graphs")
    parser.add_argument("--malicious-out", required=True, help="Output dir for processed malicious graphs")
    parser.add_argument("--vocab-dir", required=True, help="Directory to save vocabs...")
    args = parser.parse_args()
    global normal_root, malicious_root, normal_out, malicious_out, VOCAB_DIR
    normal_root = args.normal_root
    malicious_root = args.malicious_root
    normal_out = args.normal_out
    malicious_out = args.malicious_out
    if args.vocab_dir:
        VOCAB_DIR = args.vocab_dir
    else:
        p = Path(normal_root).resolve().parent
        VOCAB_DIR = str(p / "vocab")
    os.makedirs(VOCAB_DIR, exist_ok=True)

    name2idx, type2idx, edge_type2idx, behavior2idx = build_global_vocab([normal_root, malicious_root])
    os.makedirs('placeholder', exist_ok=True)
    with open(str(Path(VOCAB_DIR) / "name2idx.json"), 'w') as f:
        json.dump(name2idx, f, indent=2)
    with open(str(Path(VOCAB_DIR) / "type2idx.json"), 'w') as f:
        json.dump(type2idx, f, indent=2)
    with open(str(Path(VOCAB_DIR) / "edge_type2idx.json"), 'w') as f:
        json.dump(edge_type2idx, f, indent=2)
    with open(str(Path(VOCAB_DIR) / "behavior2idx.json"), 'w') as f:
        json.dump(behavior2idx, f, indent=2)
    clean_dir(normal_out)
    clean_dir(malicious_out)
    print("\nProcessing benign packages...")
    _ = CallGraphDatasetFull_Lazy(
        root_dir=normal_root,
        output_dir=normal_out,
        name2idx=name2idx,
        type2idx=type2idx,
        edge_type2idx=edge_type2idx,
        behavior2idx=behavior2idx,
        fixed_label=0,
    )
    print("\nProcessing malicious packages...")
    _ = CallGraphDatasetFull_Lazy(
        root_dir=malicious_root,
        output_dir=malicious_out,
        name2idx=name2idx,
        type2idx=type2idx,
        edge_type2idx=edge_type2idx,
        behavior2idx=behavior2idx,
        fixed_label=1,
    )
    print("Dataset processing complete.")
if __name__ == "__main__":
    _cli_main_()