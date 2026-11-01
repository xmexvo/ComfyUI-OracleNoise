"""Model-independent ComfyUI emphasis parsing and source-character alignment."""
from dataclasses import dataclass
import math

import torch


@dataclass
class WeightedPrompt:
    text: str
    segments: list
    weights: torch.Tensor
    source_text: str
    source_to_text: list

    def map_span(self, start, end):
        positions = [p for p in self.source_to_text[start:end] if p is not None]
        return (positions[0], positions[-1] + 1) if positions else (0, 0)


def parse_prompt(prompt):
    """Use the installed ComfyUI convention, retaining offsets through markup.

    source_text is what a tokenizer with disable_weights=True sees: emphasis
    remains literal, but escaped parentheses have their backslashes removed.
    """
    from comfy.sd1_clip import escape_important, parse_parentheses, token_weights, unescape_important

    escaped = escape_important(prompt)
    segments = token_weights(escaped, 1.0)

    def leaves(text, offset=0):
        # Mirror only the parser's removal of delimiters and numeric suffixes.
        # All weight values (including nested behavior) come from token_weights.
        for part in parse_parentheses(text):
            if len(part) >= 2 and part[0] == "(" and part[-1] == ")":
                inner = part[1:-1]
                colon = inner.rfind(":")
                if colon > 0:
                    try:
                        float(inner[colon + 1:])
                    except ValueError:
                        pass
                    else:
                        inner = inner[:colon]
                yield from leaves(inner, offset + 1)
            else:
                yield part, offset
            offset += len(part)

    source_text = unescape_important(escaped)
    source_to_text = [None] * len(source_text)
    pieces, weights = [], []
    length = 0
    for (piece, offset), (weighted_piece, weight) in zip(leaves(escaped), segments, strict=True):
        if piece != weighted_piece:
            raise ValueError("Installed ComfyUI emphasis parser has incompatible offset rules.")
        if not math.isfinite(weight):
            raise ValueError("Manual prompt importance weights must be finite.")
        text = unescape_important(piece)
        source_start = len(unescape_important(escaped[:offset]))
        source_to_text[source_start:source_start + len(text)] = range(length, length + len(text))
        pieces.append((text, weight))
        weights.extend([weight] * len(text))
        length += len(text)
    values = torch.tensor(weights, dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("Manual prompt importance weights must fit in float32.")
    return WeightedPrompt("".join(text for text, _ in pieces), pieces, values, source_text, source_to_text)
