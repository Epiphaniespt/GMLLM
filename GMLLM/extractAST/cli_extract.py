from __future__ import annotations
import argparse
from pathlib import Path
from ast_parser import parse_file
import json
from graph_builder import ProjectGraphBuilder
from llm_detector import LLMBehaviorDetector
from exporter import save_call_graph
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-extract", type=Path, help="Path to extract_config.json")
    ap.add_argument("--src", type=Path, required=True, help="Path to a Python project directory")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for call_graph.json")
    ap.add_argument("--model_name", default="mock-llm")
    ap.add_argument("--synth-rules", action="store_true", help="If API is available, synthesize rules first and apply for this run")
    ap.add_argument("--synth-rules-out", type=Path, default=Path("synth_rules.json"), help="Where to save synthesized rules JSON")
    
    ap.add_argument("--synth-rules-out", type=Path, default=Path("synth_rules.json"), help="Output JSON for synthesized rules when --prompt=comm")
    ap.add_argument("--no-fallback", action="store_true", help="kept for interface compatibility")
    ap.add_argument("--cache", type=Path, default=None, help="optional JSON cache file for detections")
    args = ap.parse_args()

    if args.config_extract and args.config_extract.exists():
        try:
            _cfg = json.loads(args.config_extract.read_text(encoding="utf-8"))
            _paths = _cfg.get("paths", {})
            _llm = _cfg.get("llm", {})
            _det = _cfg.get("detector", {})
            # Paths
            if not args.src and _paths.get("src_dir"): args.src = _paths["src_dir"]
            if not args.out and _paths.get("out_dir"): args.out = _paths["out_dir"]
            if getattr(args, "synth_rules_out", None) is not None and _paths.get("synth_rules_out"):
                args.synth_rules_out = Path(_paths["synth_rules_out"])
            if getattr(args, "cache", None) is None and _paths.get("cache_file"):
                args.cache = Path(_paths["cache_file"])
            # LLM options
            if getattr(args, "model_name", None) in (None, "mock-llm") and _llm.get("model_name"):
                args.model_name = _llm["model_name"]
            if hasattr(args, "temperature") and _llm.get("temperature") is not None:
                args.temperature = float(_llm["temperature"])
            if hasattr(args, "max_retries") and _llm.get("max_retries") is not None:
                args.max_retries = int(_llm["max_retries"])
            if hasattr(args, "timeout_s") and _llm.get("timeout_s") is not None:
                args.timeout_s = float(_llm["timeout_s"])
            # detector fallback
            if hasattr(args, "no_fallback") and _det.get("use_rule_fallback") is False:
                args.no_fallback = True
            # auto synthesize toggle
            AUTO_SYNTH_FROM_CFG = bool(_llm.get("auto_synthesize", True))
        except Exception as _e:
            print(f"[warn] failed to read extract config: {_e}")
            AUTO_SYNTH_FROM_CFG = True
    else:
        AUTO_SYNTH_FROM_CFG = True
    gb = ProjectGraphBuilder()
    for py in sorted(args.src.rglob("*.py")):
        if py.name in {"cli_extract.py", "graph_builder.py", "ast_parser.py", "llm_detector.py", "exporter.py"}:
            continue
        try:
            mod = parse_file(py)
            gb.add_module(mod)
        except Exception as e:
            print(f"[warn] failed to parse {py}: {e}")
    detector = LLMBehaviorDetector(model_name=args.model_name, use_rule_fallback=not args.no_fallback, cache_path=args.cache, prompt_name=args.prompt)

    # Auto synthesize-then-apply if requested/available
    def _api_available():
        import os
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"))
    auto_synth = (AUTO_SYNTH_FROM_CFG if 'AUTO_SYNTH_FROM_CFG' in globals() else True)
    if auto_synth or args.synth_rules:
        try:
            obj = detector.synthesize_rules()
            args.synth_rules_out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] synthesized rules to {args.synth_rules_out}")
            detector.load_synth_rules(args.synth_rules_out)
        except Exception as e:
            print(f"[warn] synthesize failed ({e}); will proceed with fallback rules.")

    # Auto synthesize-then-apply if requested and API likely available
    def _api_available():
        import os
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY"))

    auto_synth = args.synth_rules or _api_available()
    if auto_synth:
        try:
            obj = detector.synthesize_rules()
            args.synth_rules_out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] synthesized rules to {args.synth_rules_out}")
            detector.load_synth_rules(args.synth_rules_out)
        except Exception as e:
            print(f"[warn] synthesize failed ({e}); will proceed with fallback rules.")
    else:
        try:
            obj = detector.synthesize_rules()
            args.synth_rules_out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] synthesized rules to {args.synth_rules_out}")
        except Exception as e:
            print(f"[warn] failed to synthesize rules: {e}")
        return
    gb.attach_behaviors(detector)
    graph = gb.to_jsonable()
    save_call_graph(graph, args.out)
    print(f"[ok] wrote {args.out/'call_graph.json'} with {len(graph['nodes'])} nodes and {len(graph['links'])} links.")
if __name__ == "__main__":
    main()