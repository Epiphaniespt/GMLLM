from __future__ import annotations
import json, os, re, time
from pathlib import Path
from prompts import PROMPT_DATA, PROMPT_COMM
from typing import Dict, List, Optional
from rules_fallback import BEHAVIOR_RULES, KNOWN_LIBS, BUILTIN_DECOS
ALL_BEHAVIOR_TAGS: List[str] = [k for k in BEHAVIOR_RULES.keys()]
SYS_prompt = PROMPT_DATA
class LLMBehaviorDetector:
    def __init__(
        self,
        model_name: str = "gpt-4o",
        use_rule_fallback: bool = True,
        cache_path: Optional[Path] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout_s: float = 30.0,
    ):
        self.model_name = model_name
        self.use_rule_fallback = use_rule_fallback
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._dynamic_rules = {}  # synthesized rules if any
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: Dict[str, List[str]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        self._openai_client = None
        self._init_openai_client()
    def detect_behaviors(self, node_info: Dict) -> List[str]:
        key = self._key(node_info)
        if key in self.cache:
            return self.cache[key]

        labels: List[str] = []

        # 1) Prefer synthesized dynamic rules if available
        if self._dynamic_rules:
            name = ""
            if node_info.get("type") == "function":
                name = (node_info.get("qualified_name") or node_info.get("name") or "").strip()
            elif node_info.get("type") == "literal":
                name = str(node_info.get("literal_value") or "")
            for beh, fn in self._dynamic_rules.items():
                try:
                    if fn(name):
                        labels.append(beh)
                except Exception:
                    continue

        # 2) If no labels yet, fallback to original rules
        if (not labels) and self.use_rule_fallback:
            if node_info.get("type") == "function":
                qn = node_info.get("qualified_name") or ""
                ctx = node_info.get("context") or ""
                labels = self._rule_detect_function(qn, ctx)
            elif node_info.get("type") == "literal":
                lit = str(node_info.get("literal_value") or "")
                labels = self._rule_detect_literal(lit)

        labels = self._normalize_labels(labels)
        self.cache[key] = labels
        self._dump_cache()
        return labels
    def _rule_detect_function(self, qualified_name: str) -> List[str]:
        out: List[str] = []
        for tag, rule in BEHAVIOR_RULES.items():
            if tag == 'literal_secret':  
                continue
            try:
                if callable(rule) and rule(qualified_name):
                    out.append(tag)
            except Exception:
                pass
        return out
    def _rule_detect_literal(self, literal: str) -> List[str]:
        out: List[str] = []
        rule = BEHAVIOR_RULES.get('literal_secret')
        if callable(rule):
            try:
                if rule(literal):
                    out.append('literal_secret')
            except Exception:
                pass
        return out
    def _init_openai_client(self):
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return
        try:
            from openai import OpenAI  
            self._openai_client = OpenAI(api_key=api_key)
        except Exception:
            try:
                import openai  
                openai.api_key = api_key
                self._openai_client = openai
            except Exception:
                self._openai_client = None
    def _llm_classify(self, node_info: Dict) -> List[str]:
        sys_prompt = SYS_prompt
        user_prompt = json.dumps({"node_info": node_info, "allowed_labels": ALL_BEHAVIOR_TAGS}, ensure_ascii=False)
        backoff = 1.0
        last_err = None
        for _ in range(self.max_retries):
            try:
                text = self._chat_once(sys_prompt, user_prompt)
                labs = self._parse_labels(text)
                if labs is not None:
                    return labs
                text = self._chat_once(sys_prompt, user_prompt + "\n\nReturn ONLY JSON like: {\"labels\": []}")
                labs = self._parse_labels(text)
                if labs is not None:
                    return labs
                last_err = ValueError("Invalid JSON from model.")
            except Exception as e:
                last_err = e
            time.sleep(backoff)
            backoff = min(backoff * 2, 8.0)
        if last_err:
            raise last_err
        return []
    def _chat_once(self, system_prompt: str, user_prompt: str) -> str:
        if hasattr(self._openai_client, "chat"):  
            resp = self._openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                timeout=self.timeout_s,
            )
            return resp.choices[0].message.content or ""
        else:  
            resp = self._openai_client.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                request_timeout=self.timeout_s,
            )
            return resp["choices"][0]["message"]["content"]
    def _normalize_labels(self, labels: List[str]) -> List[str]:
        uniq = {lb for lb in labels if lb in ALL_BEHAVIOR_TAGS}
        return [lb for lb in ALL_BEHAVIOR_TAGS if lb in uniq]
    def _parse_labels(self, text: str) -> Optional[List[str]]:
        if not text:
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
                return self._normalize_labels([str(x) for x in obj["labels"]])
        except Exception:
            pass
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
                    return self._normalize_labels([str(x) for x in obj["labels"]])
            except Exception:
                pass
        return None
    def _key(self, node_info: Dict) -> str:
        return json.dumps(node_info, sort_keys=True, ensure_ascii=False)
    def _dump_cache(self):
        if self.cache_path:
            try:
                self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

    def synthesize_rules(self) -> dict:
        if self._openai_client is None:
            raise RuntimeError("LLM client not available (no API key or SDK).")
        user_prompt = json.dumps({"allowed_hint": list(ALL_BEHAVIOR_TAGS)}, ensure_ascii=False)
        text = self._chat_once(PROMPT_COMM, user_prompt)
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "behaviors" in obj:
                return obj
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "behaviors" in obj:
                    return obj
            except Exception:
                pass
        raise ValueError("Failed to parse synthesized rules JSON.")

    def load_synth_rules(self, path: Path) -> None:
        try:
            obj = json.loads(Path(path).read_text(encoding="utf-8"))
            rules = {}
            for item in obj.get("behaviors", []):
                name = str(item.get("name") or "").strip()
                rule_src = str(item.get("rule") or "").strip()
                if not name or not rule_src:
                    continue
                fn = self._compile_lambda_rule(rule_src)
                if fn:
                    rules[name] = fn
            if rules:
                self._dynamic_rules = rules
        except Exception:
            pass

    def _compile_lambda_rule(self, src: str):
        import ast
        try:
            tree = ast.parse(src, mode="eval")
        except Exception:
            return None
        if not isinstance(tree.body, ast.Lambda):
            return None
        lam = tree.body
        if not (len(lam.args.args) == 1 and lam.args.args[0].arg == "n"):
            return None

        ALLOWED_CALL_NAMES = {"any","all"}
        ALLOWED_STR_METHODS = {"startswith","endswith","lower","upper","strip","replace"}

        def ok(node):
            if isinstance(node, (ast.Load,)): return True
            if isinstance(node, ast.Lambda): return ok(node.body)
            if isinstance(node, ast.BoolOp): return all(ok(v) for v in node.values)
            if isinstance(node, ast.BinOp):  return ok(node.left) and ok(node.right)
            if isinstance(node, ast.UnaryOp):return ok(node.operand)
            if isinstance(node, ast.Compare):
                return ok(node.left) and all(ok(c) for c in node.comparators)
            if isinstance(node, ast.Name):
                return node.id in ("n",)
            if isinstance(node, ast.Constant):
                return isinstance(node.value, (str, bool, int, float, tuple)) or node.value is None
            if isinstance(node, ast.Tuple):
                return all(ok(e) for e in node.elts)
            if isinstance(node, ast.List):
                return all(ok(e) for e in node.elts)
            if isinstance(node, ast.Expr):
                return ok(node.value)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in ALLOWED_CALL_NAMES:
                    return all(ok(a) for a in node.args) and all(ok(k.value) for k in node.keywords)
                if isinstance(node.func, ast.Attribute):
                    def base_is_n(attr):
                        if isinstance(attr, ast.Name): return attr.id == "n"
                        if isinstance(attr, ast.Attribute): return base_is_n(attr.value)
                        return False
                    if node.func.attr in ALLOWED_STR_METHODS and base_is_n(node.func.value):
                        return all(ok(a) for a in node.args) and all(ok(k.value) for k in node.keywords)
                return False
            if isinstance(node, ast.Attribute):
                return ok(node.value)
            if isinstance(node, ast.Subscript):
                return False
            if isinstance(node, (ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.GeneratorExp, ast.IfExp)):
                return False
            return False

        if not ok(lam):
            return None

        code = compile(ast.fix_missing_locations(ast.Expression(lam)), "<synth_rule>", "eval")
        def _fn(n):
            return eval(code, {"__builtins__": {}}, {"n": n})
        return _fn
