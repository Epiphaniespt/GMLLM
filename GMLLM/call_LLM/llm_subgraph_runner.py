import os
import json
import re
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from openai import AzureOpenAI, OpenAI  # pip install openai>=1.0.0
except Exception:  # pragma: no cover
    AzureOpenAI = None
    OpenAI = None

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg.setdefault("provider", "azure")  # "azure" | "openai" | "local"
    cfg.setdefault("paths", {})
    P = cfg["paths"]
    P.setdefault("root_directory", "XXXX")
    P.setdefault("call_graph_base_path", "XXXX")
    P.setdefault("mask_filename", "masked_adj_syn1_base_h64_o20_explainnode_idx_0graph_idx_0.npy")
    P.setdefault("output_json_file", "llm_results.json")
    P.setdefault("log_file", "llm_runner.log")

    cfg.setdefault("run", {})
    R = cfg["run"]
    R.setdefault("top_k_count", 30)
    R.setdefault("max_node_name_length", 250)
    R.setdefault("start_index", 0)
    R.setdefault("end_index", None)
    R.setdefault("max_workers", 5)
    R.setdefault("api_request_timeout", 200)

    cfg.setdefault("azure", {})
    A = cfg["azure"]
    A.setdefault("api_version", os.getenv("AZURE_OPENAI_API_VERSION", "XXXX"))
    A.setdefault("deployment", os.getenv("AZURE_OPENAI_DEPLOYMENT", "XXXX"))
    A.setdefault("endpoint", os.getenv("AZURE_OPENAI_ENDPOINT", "XXXX"))
    A.setdefault("api_key_env", "AZURE_OPENAI_API_KEY")

    cfg.setdefault("openai", {})
    O = cfg["openai"]
    O.setdefault("model", os.getenv("OPENAI_MODEL", "gpt-4o"))
    O.setdefault("api_key_env", "OPENAI_API_KEY")

    cfg.setdefault("local", {})
    L = cfg["local"]

    L.setdefault("base_url", os.getenv("LOCAL_OPENAI_BASE_URL", "XXXX"))
    L.setdefault("api_key", os.getenv("LOCAL_OPENAI_API_KEY", "XXXX"))
    L.setdefault("model", os.getenv("LOCAL_OPENAI_MODEL", "XXXX"))

    return cfg

def log_to_file(log_path: str, message: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")

def extract_top_k_subgraph(mask_path: str, call_graph_path: str, top_k: int, max_name_len: int) -> Optional[str]:
    try:
        masked_adj = np.load(mask_path)
        N = masked_adj.shape[0]
        with open(call_graph_path, "r", encoding="utf-8") as f:
            cg = json.load(f)
    except Exception:
        return None

    try:
        id2name_str = {
            node["id"]: node.get("qualified_name") or node.get("name") or str(node["id"])
            for node in cg["nodes"]
        }
    except Exception:
        return None

    scored_edges: List[Tuple[float, str, str]] = []
    for link in cg.get("links", []):
        src_id = link.get("source")
        dst_id = link.get("target")
        if src_id is None or dst_id is None:
            continue
        try:
            src_idx = int(str(src_id).split("_")[-1])
            dst_idx = int(str(dst_id).split("_")[-1])
        except Exception:
            continue
        if 0 <= src_idx < N and 0 <= dst_idx < N:
            score = float(masked_adj[src_idx, dst_idx])
            if score > 0:
                scored_edges.append((score, id2name_str.get(src_id, str(src_id)), id2name_str.get(dst_id, str(dst_id))))

    if not scored_edges:
        return None
    scored_edges.sort(key=lambda x: x[0], reverse=True)
    top = scored_edges[:top_k]

    rename_map: Dict[str, str] = {}
    rename_counter = 1
    processed_edges: List[Tuple[str, str]] = []
    for _, s, t in top:
        ps, pt = s, t
        if len(ps) > max_name_len:
            if ps not in rename_map:
                rename_map[ps] = f"Filtered_Long_Node_{rename_counter}"
                rename_counter += 1
            ps = rename_map[ps]
        if len(pt) > max_name_len:
            if pt not in rename_map:
                rename_map[pt] = f"Filtered_Long_Node_{rename_counter}"
                rename_counter += 1
            pt = rename_map[pt]
        processed_edges.append((ps, pt))

    nodes = sorted({n for e in processed_edges for n in e})
    out = []
    out.append("Subgraph Nodes:")
    out += [f"- {n}" for n in nodes]
    out.append("")
    out.append("Subgraph Edges:")
    out += [f"- {u} -> {v}" for (u, v) in processed_edges]
    if rename_map:
        out.append("")
        out.append("[Analysis Note]:")
        out.append("- Very long node names were shortened (e.g., 'Filtered_Long_Node_1').")
        out.append("- Such nodes and their connections can be suspicious.")
    return "\n".join(out)

SYSTEM_PROMPT = """You are a PyPI package security auditor. 
You have been provided with the 'high-attention subgraph' structure of a PyPI package script, which only includes node names and call relationships. Based solely on this subgraph structure, please answer the following:
Is this structure indicative of potential malicious activity? (Respond only with 'Malicious' or 'Benign')
Provide your reasoning. If mitigation is needed, identify the highest priority nodes or calls for input validation or permission checks. Response Format:
Verdict:
Reasoning:
Mitigation:
"""

def make_messages(sub_graph: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sub_graph},
    ]

def call_llm_azure(cfg: Dict[str, Any], sub_graph: str, timeout_s: int) -> Dict[str, Any]:
    if AzureOpenAI is None:
        raise RuntimeError("openai package not installed. `pip install openai>=1.0.0`")
    a = cfg["azure"]
    api_key = os.getenv(a.get("api_key_env", "AZURE_OPENAI_API_KEY"), "")
    if not api_key:
        raise RuntimeError(f"Azure API key missing (env {a.get('api_key_env')}).")
    client = AzureOpenAI(
        azure_endpoint=a["endpoint"],
        api_key=api_key,
        api_version=a["api_version"],
    )
    completion = client.chat.completions.create(
        model=a["deployment"],
        messages=make_messages(sub_graph),
        timeout=timeout_s,
    )
    return completion.model_dump()

def call_llm_openai(cfg: Dict[str, Any], sub_graph: str, timeout_s: int) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. `pip install openai>=1.0.0`")
    o = cfg["openai"]
    api_key = os.getenv(o.get("api_key_env", "OPENAI_API_KEY"), "")
    if not api_key:
        raise RuntimeError(f"OpenAI API key missing (env {o.get('api_key_env')}).")
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=o["model"],
        messages=make_messages(sub_graph),
        timeout=timeout_s,
    )
    return completion.model_dump()

def call_llm_local(cfg: Dict[str, Any], sub_graph: str, timeout_s: int) -> Dict[str, Any]:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. `pip install openai>=1.0.0`")
    l = cfg["local"]
    client = OpenAI(base_url=l["base_url"], api_key=l["api_key"])
    completion = client.chat.completions.create(
        model=l["model"],
        messages=make_messages(sub_graph),
        timeout=timeout_s,
    )
    return completion.model_dump()

def call_llm(cfg: Dict[str, Any], sub_graph: str, package_name: str, timeout_s: int, log_path: str) -> Dict[str, Any]:
    provider = (cfg.get("provider") or "azure").lower()
    try:
        if provider == "openai":
            return call_llm_openai(cfg, sub_graph, timeout_s)
        elif provider == "local":
            return call_llm_local(cfg, sub_graph, timeout_s)
        else:
            return call_llm_azure(cfg, sub_graph, timeout_s)
    except Exception as e:
        log_to_file(log_path, f"[API ERROR] package={package_name} error={e}")
        return {"package_name": package_name, "error": str(e)}

def parse_llm_response(entry_name: str, api_response: Dict[str, Any]) -> Dict[str, Any]:
    if not api_response or "error" in api_response:
        return {
            "name": entry_name,
            "verdict": "API_ERROR",
            "reasoning": api_response.get("error", "Unknown API error") if isinstance(api_response, dict) else "Unknown API error",
            "mitigation": "N/A",
        }
    try:
        content = api_response["choices"][0]["message"]["content"]
    except Exception:
        return {
            "name": entry_name,
            "verdict": "PARSE_ERROR",
            "reasoning": "Failed to extract content from API response.",
            "mitigation": "N/A",
            "raw": api_response,
        }
    verdict = (re.search(r"Verdict:\s*(.*)", content, re.I) or [None, "Not Found"])[1].strip()
    reasoning = (re.search(r"Reasoning:\s*(.*?)(?=Mitigation:|$)", content, re.S | re.I) or [None, "Not Found"])[1].strip()
    mitigation = (re.search(r"Mitigation:\s*(.*)", content, re.S | re.I) or [None, "Not Found"])[1].strip()
    return {
        "name": entry_name,
        "verdict": verdict,
        "reasoning": reasoning,
        "mitigation": mitigation,
    }

def main(config_path: str = "config_LLM.json") -> None:
    cfg = load_config(config_path)
    paths = cfg["paths"]
    run = cfg["run"]

    ROOT_DIRECTORY = paths["root_directory"]
    CALL_GRAPH_BASE_PATH = paths["call_graph_base_path"]
    MASK_FILENAME = paths["mask_filename"]
    OUTPUT_JSON_FILE = paths["output_json_file"]
    LOG_FILE = paths["log_file"]

    TOP_K_COUNT = int(run["top_k_count"])
    MAX_NODE_NAME_LENGTH = int(run["max_node_name_length"])
    START_INDEX = int(run["start_index"] or 0)
    END_INDEX = run["end_index"] if run["end_index"] is None else int(run["end_index"])
    MAX_WORKERS = int(run["max_workers"])
    API_REQUEST_TIMEOUT = int(run["api_request_timeout"])

    processed = set()
    results: List[Dict[str, Any]] = []
    if os.path.exists(OUTPUT_JSON_FILE):
        try:
            existing = json.loads(open(OUTPUT_JSON_FILE, "r", encoding="utf-8").read())
            if isinstance(existing, list):
                results.extend(existing)
                processed.update([r["name"] for r in existing if isinstance(r, dict) and "name" in r])
        except Exception as e:
            log_to_file(LOG_FILE, f"[WARN] Failed to load existing results: {e}")

    if not os.path.isdir(ROOT_DIRECTORY):
        print(f"Error: root directory not found: {ROOT_DIRECTORY}")
        return

    all_dirs = sorted(os.listdir(ROOT_DIRECTORY))
    target_dirs = all_dirs[START_INDEX:END_INDEX]
    tasks: List[Tuple[str, str, str]] = []  # (package_name, mask_path, cg_path)

    for item in tqdm(target_dirs, desc="Scanning"):
        if not item.startswith("graph_"):
            continue
        pkg = item[6:]
        if pkg in processed:
            continue
        mask_path = os.path.join(ROOT_DIRECTORY, item, MASK_FILENAME)
        cg_path = os.path.join(CALL_GRAPH_BASE_PATH, pkg, "call_graph.json")
        if not (os.path.exists(mask_path) and os.path.exists(cg_path)):
            log_to_file(LOG_FILE, f"[SKIP] Missing files for {pkg}")
            continue
        tasks.append((pkg, mask_path, cg_path))

    if not tasks:
        print("All done. No new tasks.")
        return

    new_results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {}
        for pkg, mask_path, cg_path in tasks:
            subgraph = extract_top_k_subgraph(mask_path, cg_path, TOP_K_COUNT, MAX_NODE_NAME_LENGTH)
            if not subgraph:
                log_to_file(LOG_FILE, f"[SKIP] No subgraph for {pkg}")
                continue
            futs[ex.submit(call_llm, cfg, subgraph, pkg, API_REQUEST_TIMEOUT, LOG_FILE)] = pkg

        for fut in tqdm(as_completed(futs), total=len(futs), desc="Calling LLM"):
            pkg = futs[fut]
            try:
                api_resp = fut.result()
                parsed = parse_llm_response(pkg, api_resp)
                new_results.append(parsed)
                log_to_file(LOG_FILE, parsed)
            except Exception as e:
                log_to_file(LOG_FILE, f"[EXC] {pkg}: {e}")
                new_results.append(parse_llm_response(pkg, {"error": str(e)}))

    final = results + new_results
    try:
        with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {OUTPUT_JSON_FILE} (total {len(final)} entries).")
    except Exception as e:
        print(f"Failed to save results: {e}")

if __name__ == "__main__":
    main(os.getenv("CONFIG_LLM", "config_LLM.json"))
