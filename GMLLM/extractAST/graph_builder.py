from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
from llm_detector import LLMBehaviorDetector
class ProjectGraphBuilder:
    def __init__(self):
        self._nodes: List[Dict] = []
        self._links: List[Dict] = []
        self._gid_map: Dict[str, int] = {}  
        self._next_id = 0
    def _gid(self, local_key: str) -> int:
        if local_key in self._gid_map:
            return self._gid_map[local_key]
        self._next_id += 1
        self._gid_map[local_key] = self._next_id
        return self._next_id
    def add_module(self, mod_ast) -> None:
        local_to_global: Dict[str, int] = {}
        for n in mod_ast.nodes:
            if n["type"] == "literal":
                lkey = f"literal::{n.get('qualified_name')}"
            else:
                lkey = f"func::{n.get('qualified_name')}"
            gid = self._gid(lkey)
            if gid > len(self._nodes):
                nn = dict(n)
                nn["id"] = gid
                nn.setdefault("behaviors", [])
                self._nodes.append(nn)
            local_to_global[n["id"]] = gid
        for (s, t, et) in mod_ast.edges:
            gs = local_to_global[s]
            gt = local_to_global[t]
            self._links.append({"source": gs, "target": gt, "edge_type": et})
    def attach_behaviors(self, detector: LLMBehaviorDetector) -> None:
        for n in self._nodes:
            n.setdefault("behaviors", [])
            if n["type"] == "function":
                info = {
                    "type": "function",
                    "qualified_name": n.get("qualified_name") or n.get("name"),
                    "context": n.get("context", ""),
                }
                labs = detector.detect_behaviors(info)
                for lb in labs:
                    if lb not in n["behaviors"]:
                        n["behaviors"].append(lb)
            elif n["type"] == "literal":
                info = {
                    "type": "literal",
                    "literal_value": n.get("value") or n.get("qualified_name") or n.get("name", ""),
                    "context": n.get("context", ""),
                }
                labs = detector.detect_behaviors(info)
                for lb in labs:
                    if lb not in n["behaviors"]:
                        n["behaviors"].append(lb)
    def to_jsonable(self) -> Dict:
        return {"nodes": self._nodes, "links": self._links}