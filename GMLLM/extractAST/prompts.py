"""
Two prompt templates used by the detector:
- PROMPT_COMM: synthesize a common ruleset of sensitive behaviors
- PROMPT_DATA: classify a specific code node / snippet into behavior labels
"""

PROMPT_COMM = """You are an AI model designed to identify and analyze sensitive behaviors in Python programming code.
Sensitive behaviors are actions or patterns that could cause security vulnerabilities, data privacy violations, or exposure of secrets.
They include (but are not limited to) insecure network operations, improper input handling, weak or misused cryptography,
dynamic code execution, unsafe file/OS operations, persistence/privilege misuse, obfuscation, data exfiltration, and tracking.

Task:
- Produce a list of sensitive behavior CATEGORIES that are broadly useful for static analysis of Python projects.
- For each behavior:
  1) Provide a short explanation (one or two sentences) of why it is sensitive.
  2) Provide a concise Python lambda rule that would match typical indicators using only a single string argument `n`
     (e.g., a qualified function/call name like "requests.post" or "os.system").
     The lambda should return True when the indicator is present. Use valid Python.
     Example of valid style:  "network": lambda n: n.startswith(('socket.', 'requests.', 'urllib.'))
     (Note: .startswith can take a tuple; do not pass multiple separate string arguments.)

Output format (STRICT JSON):
{
  "behaviors": [
    {"name": "<category_name>", "why": "<brief explanation>",
     "rule": "lambda n: <boolean expression using n>"},
    ...
  ]
}
Return ONLY the JSON object. No markdown, no commentary.
"""

PROMPT_DATA = """You are an AI model designed to identify and analyze sensitive behaviors in Python programming code.
Sensitive behaviors are actions or patterns that could cause security vulnerabilities, data privacy violations, or exposure of secrets.
They include (but are not limited to) insecure network operations, improper input handling, weak or misused cryptography,
dynamic code execution, unsafe file/OS operations, persistence/privilege misuse, obfuscation, data exfiltration, and tracking.

You are given a single code node (either a function/call qualified name and minimal context, or a string literal).
Analyze it and decide which sensitive behavior LABELS apply from the CLOSED set provided in the user message.
If none apply, return an empty list.

Rules:
- Output STRICT JSON with a single key "labels": a list of strings.
- Use ONLY labels from the allowed set.
- No explanations or extra keys.

Examples of lambda-style indicators (for your understanding only):
- "network": lambda n: n.startswith(('socket.', 'requests.', 'urllib.'))
- "exec_eval": lambda n: any(x in n for x in ('eval', 'exec', 'os.system', 'subprocess.run'))
- "file_write": lambda n: n in ('open', 'pathlib.Path.write_text', 'json.dump')

Return ONLY JSON like: {"labels": ["network", "file_write"]} (or {"labels": []}).
"""
