import math

import torch
from torch.utils.checkpoint import checkpoint


def normalize_scores(scores, low=0.5, high=3.0):
    scores = torch.as_tensor(scores, dtype=torch.float32)
    if not torch.isfinite(scores).all():
        raise ValueError("Importance scores must be finite.")
    if not scores.numel():
        return scores
    spread = scores.max() - scores.min()
    if spread <= 1e-8:
        return torch.full_like(scores, (low + high) / 2)
    return low + (high - low) * (scores - scores.min()) / spread


def spherical_step(noise, gradient, angle, check_finite=True):
    """One normalized angular ascent step, independently for each batch item."""
    if check_finite and not torch.isfinite(gradient).all():
        raise RuntimeError("Oracle Noise encountered a nonfinite gradient.")
    shape = noise.shape
    z, g = noise.float().flatten(1), gradient.float().flatten(1)
    radius = torch.linalg.vector_norm(z, dim=1, keepdim=True)
    tangent = g - (z * g).sum(1, keepdim=True) / radius.square().clamp_min(1e-20) * z
    length = torch.linalg.vector_norm(tangent, dim=1, keepdim=True)
    moved = z * math.cos(angle) + radius * tangent / length.clamp_min(1e-20) * math.sin(angle)
    moved = moved * radius / torch.linalg.vector_norm(moved, dim=1, keepdim=True).clamp_min(1e-20)
    result = torch.where(length > 1e-12, moved, z).reshape(shape)
    if check_finite and not torch.isfinite(result).all():
        raise RuntimeError("Oracle Noise produced nonfinite noise.")
    return result


def attention_objective(q, k, weights, image_queries, text_start=0, cfg=1.0,
                        masks=None, chunk_size=128, query_indices=None):
    """q/k: [branches, heads, positions, head_dim], positive branch first.

    MOSS includes every image/text key in the denominator. CFG uses a neutral
    zero logit for absent negative slots, then keeps the positive key domain.
    """
    if q.shape[0] not in (1, 2) or q.shape[0] != k.shape[0]:
        raise ValueError("Expected one image and one or two conditioning branches.")
    if cfg != 1 and q.shape[0] != 2:
        raise ValueError("CFG attention requires both conditioning branches.")
    weights = weights.to(device=q.device, dtype=torch.float32)
    positive_valid = masks[0] if masks is not None else None
    end = text_start + weights.numel()
    if end > k.shape[-2]:
        raise ValueError("Importance vector does not match attention text keys.")

    def score(query, keys):
        logits = query.float() @ keys.float().transpose(-2, -1) / math.sqrt(query.shape[-1])
        if masks is not None and cfg != 1:
            logits = logits.masked_fill(~masks[:, None, None, :], 0)
        guided = logits[0] if cfg == 1 else logits[1] + cfg * (logits[0] - logits[1])
        if positive_valid is not None:
            guided = guided.masked_fill(~positive_valid, -torch.inf)
        probs = guided.softmax(-1)
        return (probs[..., text_start:end] * weights).sum() / (q.shape[1] * image_queries)

    scored_queries = image_queries if query_indices is None else query_indices.numel()
    if scored_queries < 1:
        raise ValueError("Query sampling selected no image positions.")
    loss = q.new_zeros((), dtype=torch.float32)
    for start in range(0, scored_queries, chunk_size):
        if query_indices is None:
            query = q[:, :, start:min(start + chunk_size, scored_queries)]
        else:
            query = q[:, :, query_indices[start:min(start + chunk_size, scored_queries)]]
        if torch.is_grad_enabled() and (query.requires_grad or k.requires_grad):
            loss = loss + checkpoint(score, query, k, use_reentrant=False)
        else:
            loss = loss + score(query, k)
    return loss * (image_queries / scored_queries)


def span_scores(spans, scores, length):
    sums, counts = torch.zeros(length), torch.zeros(length)
    for (start, end), value in zip(spans, scores):
        if end > start:
            sums[start:end] += float(value)
            counts[start:end] += 1
    return sums / counts.clamp_min(1), counts > 0


def combine_importance(components, length):
    total, count = torch.zeros(length), torch.zeros(length)
    for spans, values in components:
        chars, valid = span_scores(spans, values, length)
        chars[valid] = normalize_scores(chars[valid], 0, 1)
        total[valid] += chars[valid]
        count[valid] += 1
    valid = count > 0
    result = total / count.clamp_min(1)
    result[valid] = 0.5 + 2.5 * result[valid]
    return result, valid


def map_importance(char_scores, char_valid, spans):
    values = []
    for start, end in spans:
        valid = char_valid[start:end]
        values.append(char_scores[start:end][valid].mean().item() if valid.any() else 0.0)
    return torch.tensor(values, dtype=torch.float32)
