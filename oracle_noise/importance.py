from collections import OrderedDict
import threading
import weakref

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import convert_slow_tokenizer

from .math import combine_importance


_CACHE = weakref.WeakKeyDictionary()
_CACHE_LOCK = threading.RLock()


def fast_tokenizer(tokenizer):
    if tokenizer.is_fast:
        return tokenizer
    return PreTrainedTokenizerFast(tokenizer_object=convert_slow_tokenizer(tokenizer))


def token_spans(sd_tokenizer, text):
    fast = fast_tokenizer(sd_tokenizer.tokenizer)
    encoded = fast(text, add_special_tokens=False, return_offsets_mapping=True)
    return encoded["input_ids"], encoded["offset_mapping"]


def select_components(encoder, mode):
    if mode not in ("auto", "clip_l", "clip_g", "clip_l_g"):
        raise ValueError(f"Unknown importance mode: {mode}")
    components = []
    for name in ("clip_l", "clip_g"):
        model = getattr(encoder.cond_stage_model, name, None)
        tokenizer = getattr(encoder.tokenizer, name, None)
        if model is not None and tokenizer is not None:
            components.append((name, model, tokenizer))
    wanted = {"clip_l", "clip_g"} if mode == "clip_l_g" else {mode}
    if mode != "auto":
        available = {item[0] for item in components}
        if not wanted <= available:
            raise ValueError(f"Importance mode {mode} requires {sorted(wanted)}; available: {sorted(available)}")
        components = [item for item in components if item[0] in wanted]
    if not components:
        raise ValueError("Importance encoder must contain CLIP-L and/or CLIP-G, not a native Qwen encoder.")
    return components


def window_starts(length, capacity):
    if length <= capacity:
        return [0]
    stride = max(1, capacity // 2)
    return list(range(0, length - capacity, stride)) + [length - capacity]


def score_component(model, tokenizer, prompt, device, interrupt, name="clip_l", batch_size=8):
    ids, spans = token_spans(tokenizer, prompt)
    if not ids:
        return spans, torch.empty(0)
    capacity = tokenizer.max_length - 2
    if capacity <= 0 or tokenizer.start_token is None or tokenizer.end_token is None:
        raise ValueError("Oracle Noise requires a standard CLIP tokenizer with BOS and EOS.")
    model.reset_clip_options()
    model.set_clip_options({"execution_device": device, "projected_pooled": True})
    sums, counts = torch.zeros(len(ids)), torch.zeros(len(ids))
    try:
        for start in window_starts(len(ids), capacity):
            window = ids[start:start + capacity]
            base = [tokenizer.start_token, *window, tokenizer.end_token]
            base += [tokenizer.pad_token] * (tokenizer.max_length - len(base))
            # EOS-as-PAD in CLIP-L must not move pooling to the lesioned slot.
            # The forward method below pools explicitly at the original EOS.
            eos = len(window) + 1
            def pooled(rows):
                tokens = torch.tensor(rows, device=device)
                transformer = model.transformer
                outputs = transformer(tokens, intermediate_output=None, final_layer_norm_intermediate=True)
                hidden = outputs[0]
                out = hidden[:, eos].float()
                projection = transformer.text_projection
                # SD1 CLIP-L checkpoints usually omit the projection entirely.
                return projection(out) if name == "clip_g" and projection is not None else out
            with torch.no_grad():
                baseline = pooled([base])
                for first in range(0, len(window), batch_size):
                    interrupt()
                    rows = []
                    for j in range(first, min(first + batch_size, len(window))):
                        row = base.copy()
                        row[j + 1] = tokenizer.pad_token
                        rows.append(row)
                    vectors = pooled(rows)
                    impacts = (1 - F.cosine_similarity(baseline, vectors, dim=-1)).cpu()
                    sums[start + first:start + first + len(rows)] += impacts
                    counts[start + first:start + first + len(rows)] += 1
    finally:
        model.reset_clip_options()
    return spans, sums / counts.clamp_min(1)


def score_prompt(encoder, mode, prompt, interrupt, batch_size=8):
    components = select_components(encoder, mode)
    identity = (prompt, mode, batch_size, encoder.patcher.patches_uuid,
                tuple((n, id(m), id(t), t.max_length, t.pad_token, t.start_token, t.end_token)
                      for n, m, t in components))
    with _CACHE_LOCK:
        cached = _CACHE.get(encoder, {}).get(identity)
    if cached is not None:
        return cached[0].clone(), cached[1].clone()
    encoder.load_model()
    device = encoder.patcher.load_device
    results = [score_component(model, tokenizer, prompt, device, interrupt, name, batch_size) for name, model, tokenizer in components]
    result = combine_importance(results, len(prompt))
    with _CACHE_LOCK:
        cache = _CACHE.setdefault(encoder, OrderedDict())
        cache[identity] = result
        while len(cache) > 16:
            cache.popitem(last=False)
    return result
