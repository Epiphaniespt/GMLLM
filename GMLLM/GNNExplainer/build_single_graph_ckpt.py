import torch
import numpy as np
import json
import os
from torch_geometric.data import Data
from distinguish_GNN_2 import GCNWithBehavior
from torch_geometric.loader import DataLoader
import argparse
def build_and_save_single_ckpt(pt_graph_path, model_ckpt_path,output_ckpt_path,deviceCuda):
    if torch.cuda.is_available():
        device = torch.device(deviceCuda)  
    else:
        device = torch.device('cpu')
    data: Data = torch.load(pt_graph_path,weights_only=False)
    print(device)
    data = data.to(device)
    ckpt = torch.load(model_ckpt_path, map_location=device)
    model = GCNWithBehavior(        
    name_vocab_size=len(name2idx),
    type_vocab_size=len(type2idx),
    behavior_dim=len(behavior2idx),
    hidden_dim=64, 
    num_classes=2 )  
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    N = data.num_nodes
    adj_dense = torch.zeros((N, N), dtype=torch.float32, device=device)
    src, dst = data.edge_index
    adj_dense[src, dst] = 1.0
    adj_np = adj_dense.cpu().numpy()[np.newaxis, :, :]  
    name_feat = data.x_names 
    type_feat = data.x_types 
    behavior_feat = data.x_behaviors.float() 
    data.num_nodes = data.x_names.size(0)
    with torch.no_grad():
        name_emb = model.name_emb(name_feat)  
        type_emb = model.type_emb(type_feat)  
        x = torch.cat([name_emb, type_emb, behavior_feat], dim=1)  
    data.x = x
    feat_np = data.x.cpu().numpy()[np.newaxis, :, :]   
    x_names_np     = data.x_names.cpu().numpy()[np.newaxis, :]       
    x_types_np     = data.x_types.cpu().numpy()[np.newaxis, :]       
    x_behaviors_np = data.x_behaviors.cpu().numpy()[np.newaxis, :, :]
    label_np = np.array([ int(data.y.item()) ], dtype=np.int64)  
    single_loader = DataLoader([data],batch_size=1)
    with torch.no_grad():
        for batch_data in single_loader:
            batch_data = batch_data.to(device)
            out = model(batch_data)   
            prob = torch.softmax(out.view(-1), dim=0).cpu().numpy() 
    pred_np = prob[np.newaxis,np.newaxis,  :]             
    train_idx = list(range(N))
    cg_dict = {
        'adj':            adj_np.astype(np.float32),      
        'feat':           feat_np.astype(np.float32),     
        'name_idx':       x_names_np.astype(np.int64),    
        'type_idx':       x_types_np.astype(np.int64),    
        'behavior_feats': x_behaviors_np.astype(np.float32), 
        'label':          label_np.astype(np.int64),      
        'pred':           pred_np.astype(np.float32),     
        'train_idx':      train_idx
    }
    new_ckpt = {
        'model_state_dict': model.state_dict(),
        'cg_dict':          cg_dict
    }
    torch.save(new_ckpt, output_ckpt_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_filename", required=True)
    parser.add_argument("--model_weights", required=True)
    parser.add_argument("--vocab_dir", required=True)
    parser.add_argument("--output_ckpt_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--handled_dir", required=True)
    args = parser.parse_args()
    with open(os.path.join(args.vocab_dir, 'name2idx.json'), 'r') as f: name2idx = json.load(f)
    with open(os.path.join(args.vocab_dir, 'type2idx.json'), 'r') as f: type2idx = json.load(f)
    with open(os.path.join(args.vocab_dir, 'behavior2idx.json'), 'r') as f: behavior2idx = json.load(f)
    pt_graph_path = os.path.join(args.handled_dir, args.pt_filename+".pt")
    build_and_save_single_ckpt( pt_graph_path , args.model_weights , args.output_ckpt_dir,args.device)