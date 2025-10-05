import sys, types

sys.modules["boto3"] = types.ModuleType("boto3")

from pathlib import Path
import json
from run_bedrock_experiments import build_prompts, render_prompt

fewshot_path = Path(
    "data/derived/psycomark_official_split_20250928_232947/best_fewshot_examples.json"
)
with fewshot_path.open("r", encoding="utf-8") as f:
    fewshots_data = json.load(f)

fewshots = fewshots_data["s1"]
prompt_joint, prompt_classify = build_prompts(fewshots)
text = fewshots[0]["text"]
assert "{TEXT}" in prompt_joint and "{TEXT}" in prompt_classify
render_prompt(prompt_joint, text)
render_prompt(prompt_classify, text)
print("Prompt rendering sanity check passed.")
