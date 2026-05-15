from __future__ import annotations
import json
from pathlib import Path
REQUIRED_NODE_FIELDS = {"id", "type", "name", "qualified_name", "file", "behaviors"}
REQUIRED_LINK_FIELDS = {"source", "target", "edge_type"}
def quick_validate(graph: dict):
    assert "nodes" in graph and "links" in graph, "graph must contain 'nodes' and 'links'"
    for n in graph["nodes"]:
        missing = REQUIRED_NODE_FIELDS - set(n.keys())
        assert not missing, f"node missing fields: {missing}"
        assert isinstance(n["behaviors"], list), "node.behaviors must be a list"
    for e in graph["links"]:
        missing = REQUIRED_LINK_FIELDS - set(e.keys())
        assert not missing, f"link missing fields: {missing}"
def save_call_graph(graph: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    quick_validate(graph)
    (out_dir / "call_graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")