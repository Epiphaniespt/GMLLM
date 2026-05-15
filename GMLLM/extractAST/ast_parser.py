from __future__ import annotations
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
@dataclass
class ModuleAST:
    file: Path
    nodes: List[Dict] = field(default_factory=list)               
    edges: List[Tuple[str, str, str]] = field(default_factory=list)  
class _Visitor(ast.NodeVisitor):
    def __init__(self, file: Path):
        self.file = file
        self.mod = ModuleAST(file=file)
        self.scope_stack: List[str] = []  
        self._node_ids = 0
        self._idmap_local: Dict[str, str] = {}  
    def _new_id(self) -> str:
        self._node_ids += 1
        return f"{self.file}:{self._node_ids}"
    def _ensure_function_node(self, qualified_name: str, display_name: Optional[str] = None, context: str = "") -> str:
        key = f"func::{qualified_name}"
        if key in self._idmap_local:
            return self._idmap_local[key]
        nid = self._new_id()
        self._idmap_local[key] = nid
        self.mod.nodes.append({
            "id": nid,
            "type": "function",
            "name": display_name or qualified_name.split(".")[-1],
            "qualified_name": qualified_name,
            "file": str(self.file),
            "context": context,
        })
        return nid
    def _add_literal_node(self, value: str, context: str) -> str:
        key = f"lit::{value}"
        if key in self._idmap_local:
            return self._idmap_local[key]
        nid = self._new_id()
        self._idmap_local[key] = nid
        self.mod.nodes.append({
            "id": nid,
            "type": "literal",
            "name": (value[:32] + "…") if len(value) > 32 else value,
            "qualified_name": value,  
            "value": value,
            "file": str(self.file),
            "context": context,
        })
        return nid
    def _current_scope(self) -> str:
        return ".".join(self.scope_stack) if self.scope_stack else "<module>"
    def visit_FunctionDef(self, node: ast.FunctionDef):
        qn = f"{self.file.stem}.{node.name}" if not self.scope_stack else f"{self.file.stem}." + ".".join(self.scope_stack+[node.name])
        fid = self._ensure_function_node(qn, display_name=node.name, context=f"defined at L{node.lineno}")
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_ClassDef(self, node: ast.ClassDef):
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()
    def visit_Call(self, node: ast.Call):
        callee = None
        if isinstance(node.func, ast.Attribute):
            parts = []
            x = node.func
            while isinstance(x, ast.Attribute):
                parts.append(x.attr)
                x = x.value
            if isinstance(x, ast.Name):
                parts.append(x.id)
            parts.reverse()
            callee = ".".join(parts)
        elif isinstance(node.func, ast.Name):
            callee = node.func.id
        caller_qn = (f"{self.file.stem}." + self._current_scope().replace("<module>", "__main__"))
        caller_id = self._ensure_function_node(caller_qn, display_name=self._current_scope())
        if callee:
            callee_qn = callee
            callee_id = self._ensure_function_node(callee_qn, display_name=callee.split(".")[-1], context=f"called at L{node.lineno}")
            self.mod.edges.append((caller_id, callee_id, "calls"))
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                lit_id = self._add_literal_node(str(arg.value), context=f"literal at L{getattr(arg, 'lineno', node.lineno)}")
                self.mod.edges.append((caller_id, lit_id, "uses_literal"))
        self.generic_visit(node)
def parse_file(path: Path) -> ModuleAST:
    src = path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(src)
    v = _Visitor(path)
    v.visit(tree)
    return v.mod