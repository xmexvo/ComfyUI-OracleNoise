"""Prompt preparation with explicit character alignment for structured layouts."""
from dataclasses import dataclass
import json
import math

import torch

from .prompt import WeightedPrompt, parse_prompt


@dataclass
class OraclePromptData:
    original: str
    encoder_text: str
    parsed: WeightedPrompt
    layout: bool
    conditioning_tensor: object = None


def prepare_prompt(text, format="Auto"):
    if format not in ("Auto", "Plain text", "MOSS layout"):
        raise ValueError(f"Unknown prompt format: {format}")
    layout = format == "MOSS layout" or (format == "Auto" and text.lstrip().startswith(("{", "[")))
    if not layout:
        parsed = parse_prompt(text)
        encoded = parsed.text.replace("(", r"\(").replace(")", r"\)")
        return OraclePromptData(text, encoded, parsed, False)

    def invalid(message):
        raise ValueError(f"Unsupported MOSS layout: {message}")

    def object_pairs(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                invalid(f"duplicate key {key!r}")
            obj[key] = value
        return obj

    try:
        obj = json.loads(text, object_pairs_hook=object_pairs,
                         parse_constant=lambda value: invalid(f"nonfinite number {value}"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid MOSS layout JSON: {error.msg} at line {error.lineno}") from error
    if not isinstance(obj, dict) or not isinstance(obj.get("summary"), dict) or not isinstance(obj.get("items"), list):
        invalid("expected summary object and items array; select Plain text for literal JSON")
    descriptions = [(obj["summary"], "caption")]
    if not obj["items"]:
        invalid("items must not be empty")
    for item in obj["items"]:
        if not isinstance(item, dict) or item.get("label") not in ("object", "character", "text"):
            invalid("each item needs an object, character, or text label")
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox):
            invalid("each bbox must contain four finite numbers")
        if item["label"] == "text" and not isinstance(item.get("text"), str):
            invalid("text items need literal text")
        descriptions.append((item, "caption"))

    fields = {}
    pieces, multipliers = [], []
    length = 0
    for parent, key in descriptions:
        if not isinstance(parent.get(key), str) or not parent[key].strip():
            invalid("summary and every item need a nonempty caption")
        parsed = parse_prompt(parent[key])
        if pieces:
            pieces.append("\n")
            multipliers.append(0.)
            length += 1
        fields[(id(parent), key)] = (parsed.text, length)
        pieces.append(parsed.text)
        multipliers.extend(parsed.weights.tolist())
        length += len(parsed.text)

    chunks, mapping = [], []

    def append(value, positions=None):
        chunks.append(value)
        mapping.extend([None] * len(value) if positions is None else positions)

    def string(value, start=None):
        append('"')
        for index, char in enumerate(value):
            escaped = json.dumps(char, ensure_ascii=False)[1:-1]
            append(escaped, [None if start is None else start + index] * len(escaped))
        append('"')

    def serialize(value):
        if isinstance(value, dict):
            append("{")
            for index, (key, child) in enumerate(value.items()):
                if index:
                    append(",")
                string(key)
                append(":")
                field = fields.get((id(value), key))
                if field is not None:
                    string(*field)
                else:
                    serialize(child)
            append("}")
        elif isinstance(value, list):
            append("[")
            for index, child in enumerate(value):
                if index:
                    append(",")
                serialize(child)
            append("]")
        elif isinstance(value, str):
            string(value)
        else:
            append(json.dumps(value, ensure_ascii=False, allow_nan=False))

    serialize(obj)
    source = "".join(chunks)
    semantic = "".join(pieces)
    parsed = WeightedPrompt(semantic, [], torch.tensor(multipliers, dtype=torch.float32), source, mapping)
    encoded = source.replace("(", r"\(").replace(")", r"\)")
    return OraclePromptData(text, encoded, parsed, True)
