import numpy as np
import json
import pandas as pd
import os
import argparse
def main():
    parser = argparse.ArgumentParser(description="Extract explanation results to JSON.")
    parser.add_argument("--graph_base_filename", type=str, required=True,
                        help="Base filename of the graph being processed (e.g., 15cent, WITHOUT .pt suffix).")
    parser.add_argument("--k_edge_threshold", type=float, default=0.0,
                        help="Threshold for filtering subgraph edges for LLM.")
    parser.add_argument("--max_elements_saved", type=int, default=50,
                        help="Max nodes/edges to save if graph is large.")
    parser.add_argument("--base_explain_logs", required=True)
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--call_location", required=True)
    args = parser.parse_args()
    BASE_GRAPH_DATA_DIR = args.base_explain_logs
    current_log_sub_dir = os.path.join(args.log_dir,f"graph_{args.graph_base_filename}")
    mask_filename = "masked_adj_syn1_base_h64_o20_explainnode_idx_0graph_idx_0.npy"
    mask_path = os.path.join(current_log_sub_dir, mask_filename)
    call_location = args.call_location
    call_graph_json_path = os.path.join(
        BASE_GRAPH_DATA_DIR, call_location, args.graph_base_filename, "call_graph.json"
    )
    try:
        masked_adj = np.load(mask_path)
    except FileNotFoundError:
        print(f"Error: Mask file not found at {mask_path}.")
        exit()
    try:
        with open(call_graph_json_path, 'r', encoding='utf-8') as f:
            call_graph = json.load(f)
    except FileNotFoundError:
        print(f"Error: The specific call_graph.json not found at {call_graph_json_path}.")
        exit()
    local_idx_to_name = {
        i: node.get('qualified_name') or node.get('name', f"unnamed_node_{i}")
        for i, node in enumerate(call_graph.get('nodes', []))
    }
    N_sub = masked_adj.shape[0]
    if N_sub != len(local_idx_to_name):
        print(f"Warning: Mask dimension ({N_sub}) does not match number of nodes in JSON ({len(local_idx_to_name)}). Results may be incorrect.")
    u_indices, v_indices = np.triu_indices(N_sub, k=1)
    all_edge_weights = masked_adj[u_indices, v_indices]
    sorted_edge_indices = np.argsort(np.abs(all_edge_weights))[::-1]
    edges_to_save = []
    for i in range(min(args.max_elements_saved, len(sorted_edge_indices))):
        idx = sorted_edge_indices[i]
        ui, vi = u_indices[idx], v_indices[idx]
        weight = all_edge_weights[idx]
        if abs(weight) > args.k_edge_threshold:
            edges_to_save.append({
                "source_node": local_idx_to_name.get(ui, f"unknown_idx_{ui}"),
                "target_node": local_idx_to_name.get(vi, f"unknown_idx_{vi}"),
                "attention_weight": float(weight)
            })
    all_node_scores = masked_adj.sum(axis=1) + masked_adj.sum(axis=0)
    sorted_node_indices = np.argsort(np.abs(all_node_scores))[::-1]
    nodes_to_save = []
    for i in range(min(args.max_elements_saved, N_sub)):
        ni = sorted_node_indices[i]
        score = all_node_scores[ni]
        if abs(score) > 1e-9: 
            nodes_to_save.append({
                "node_name": local_idx_to_name.get(ni, f"unknown_idx_{ni}"),
                "total_attention_score": float(score)
            })
    output_data = {
        "important_nodes": nodes_to_save,
        "important_edges": edges_to_save,
    }
    output_filename = os.path.join(current_log_sub_dir, f"explanation_results_{args.graph_base_filename}.json")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        print(f"\nExplanation results saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")
if __name__=='__main__':
    main()