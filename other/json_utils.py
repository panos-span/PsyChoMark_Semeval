# json_utils.py
import json, re


class JsonExtractError(Exception):
    pass


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def strip_md_fences(s: str) -> str:
    return s.replace("```json", "").replace("```", "").strip()


def first_json_block(text: str):
    text = strip_md_fences(text)
    # Prefer array for S1, object for S2
    arr = _JSON_ARRAY_RE.search(text)
    obj = _JSON_OBJECT_RE.search(text)
    if arr:
        return arr.group(0), "array"
    if obj:
        return obj.group(0), "object"
    raise JsonExtractError("No JSON block found.")


def parse_safe(text: str):
    block, kind = first_json_block(text)
    try:
        return json.loads(block), kind
    except json.JSONDecodeError as e:
        # ultra-light repair: remove trailing commas & fix smart quotes
        repaired = (
            block.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
        )
        repaired = re.sub(r",(\s*[\}\]])", r"\1", repaired)
        return json.loads(repaired), kind
