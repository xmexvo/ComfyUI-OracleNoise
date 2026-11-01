import torch

from .importance import token_spans
from .prompt import parse_prompt


_ALLOWED = {"pooled_output", "attention_mask", "t5xxl_ids", "t5xxl_weights"}
MOSS_PREFIX = "<|im_start|>user\n"
MOSS_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n"


def validate_conditioning(conditioning, name):
    if not isinstance(conditioning, (list, tuple)) or len(conditioning) != 1:
        raise ValueError(f"{name} must contain exactly one ordinary prompt conditioning.")
    tensor, metadata = conditioning[0]
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"{name} must be a single [1, tokens, features] tensor.")
    unsupported = set(metadata) - _ALLOWED
    if unsupported:
        raise ValueError(f"{name} has unsupported conditioning metadata: {sorted(unsupported)}")
    return tensor, metadata


def native_mapping(encoder, family, prompt, positive, parsed=None, conditioning_prompt=None, layout=False):
    tensor, metadata = validate_conditioning(positive, "positive")
    parsed = parse_prompt(prompt.strip() if family == "moss" else prompt) if parsed is None else parsed
    semantic = parsed.text
    encoding_text = prompt if conditioning_prompt is None else conditioning_prompt
    encoded_parsed = parsed if layout or conditioning_prompt is None else parse_prompt(
        encoding_text.strip() if family == "moss" else encoding_text)
    semantic_offset = 0
    if family == "moss" and conditioning_prompt is not None and not layout:
        semantic_offset = len(semantic) - len(semantic.lstrip())
        matching_text = semantic.strip()
    else:
        matching_text = semantic
    if not layout and encoded_parsed.text != matching_text:
        raise ValueError("Conditioning prompt text does not match the Oracle prompt after removing emphasis.")
    tokenized = encoder.tokenize(encoding_text)
    if family == "anima":
        if layout:
            raise ValueError("MOSS layout prompt data requires a MOSS model.")
        tokenizer = encoder.tokenizer.t5xxl
        ids, spans, offset = [], [], 0
        for piece, _ in encoded_parsed.segments:
            part_ids, part_spans = token_spans(tokenizer, piece)
            ids.extend(part_ids)
            spans.extend((a + offset, b + offset) for a, b in part_spans)
            offset += len(piece)
        actual = [int(t[0]) for t in tokenized["t5xxl"][0]]
        expected = ids + ([tokenizer.end_token] if tokenizer.end_token is not None else [])
        if actual[:len(expected)] != expected or len(actual) < len(expected):
            raise ValueError("Anima tokenizer offsets differ from native tokens; embeddings are unsupported.")
        native_ids = metadata.get("t5xxl_ids")
        if native_ids is None or native_ids.flatten().tolist() != actual:
            raise ValueError("Anima prompt does not match positive conditioning T5 token IDs.")
        spans += [(0, 0)] * (max(512, len(actual)) - len(spans))
    elif family == "moss":
        if not layout and (prompt.startswith("<|im_start|>") or prompt.lstrip().startswith(("{", "["))):
            raise ValueError("MOSS v1 accepts plain text, not layout JSON or explicit chat templates.")
        tokenizer = encoder.tokenizer.qwen35_9b
        templated = MOSS_PREFIX + encoded_parsed.source_text + MOSS_SUFFIX
        ids, offsets = token_spans(tokenizer, templated)
        actual = [int(t[0]) for t in tokenized["qwen35_9b"][0]]
        if ids != actual or tensor.shape[1] != len(actual):
            raise ValueError("MOSS prompt/template token length differs from positive conditioning.")
        start, end = len(MOSS_PREFIX), len(MOSS_PREFIX) + len(encoded_parsed.source_text)
        spans = [encoded_parsed.map_span(max(a, start) - start, min(b, end) - start) if b > start and a < end else (0, 0)
                 for a, b in offsets]
        if semantic_offset:
            spans = [(a + semantic_offset, b + semantic_offset) if b > a else (0, 0) for a, b in spans]
    else:
        raise ValueError(f"Unsupported model family: {family}")
    return semantic, spans


def pad_contexts(contexts):
    length = max(c.shape[1] for c in contexts)
    padded, masks = [], []
    for context in contexts:
        size = context.shape[1]
        padded.append(torch.nn.functional.pad(context, (0, 0, 0, length - size)))
        masks.append(torch.arange(length, device=context.device) < size)
    return torch.cat(padded), torch.stack(masks)
