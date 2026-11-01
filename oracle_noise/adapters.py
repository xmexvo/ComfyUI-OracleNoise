from contextlib import contextmanager
import functools
import threading

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from .math import attention_objective


OPTIMIZATION_LOCK = threading.RLock()
_ATTENTION_PRECISION = 'fp32'


@contextmanager
def attention_precision(mode):
    global _ATTENTION_PRECISION
    if mode not in ("fp32", "bf16", "fp16"):
        raise ValueError(f"Unknown attention precision: {mode}")
    previous = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    try:
        previous_precision = _ATTENTION_PRECISION
        _ATTENTION_PRECISION = mode
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(mode in ("bf16", "fp16"))
        yield
    finally:
        _ATTENTION_PRECISION = previous_precision
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(previous)


def family_of(model):
    name = type(model.get_model_object("diffusion_model")).__name__
    if name == "Anima":
        return "anima"
    if name == "MOSSImage":
        return "moss"
    raise ValueError(f"Oracle Noise supports Anima and MOSSImage; received {name}.")


def layer_selection(count, indices, weights):
    selected = [0, count // 2, count - 1] if not indices.strip() else [int(i.strip()) for i in indices.split(",")]
    scales = [float(i.strip()) for i in weights.split(",")]
    if len(selected) != len(scales) or len(set(selected)) != len(selected):
        raise ValueError("Layer indices and weights must have equal lengths and unique indices.")
    if any(i < 0 or i >= count for i in selected) or not all(torch.isfinite(torch.tensor(scales))):
        raise ValueError(f"Layer indices must be in [0, {count - 1}] and weights must be finite.")
    return dict(zip(selected, scales))


def preset_layer_selection(count, family, preset, indices, weights):
    if preset == "Custom":
        return layer_selection(count, indices, weights)
    presets = {"Anima": ("anima", 28, "0,14,27"), "MOSS": ("moss", 42, "0,21,41")}
    if preset not in presets:
        raise ValueError(f"Unknown block preset: {preset}")
    expected_family, expected_count, selected = presets[preset]
    if family != expected_family or count != expected_count:
        raise ValueError(f"{preset} block preset requires a {expected_family} model with {expected_count} blocks; got {family} with {count} blocks. Use Custom for other models.")
    return layer_selection(count, selected, "1.0,1.5,2.0")


def torch_attention(_original, q, k, v, heads, mask=None, skip_reshape=False,
                    transformer_options=None, **kwargs):
    if not skip_reshape:
        q = q.unflatten(-1, (heads, -1)).transpose(1, 2)
        k = k.unflatten(-1, (heads, -1)).transpose(1, 2)
        v = v.unflatten(-1, (heads, -1)).transpose(1, 2)
    options = transformer_options or {}
    sink = options.get("oracle_sink")
    key_mask = options.get("oracle_key_mask")
    if sink is not None:
        sink.append(attention_objective(q, k, options["oracle_weights"], options["oracle_queries"],
                                        options.get("oracle_text_start", 0), options["oracle_cfg"],
                                        key_mask, options["oracle_chunk"], options.get("oracle_query_indices")))
    attn_mask = mask
    if key_mask is not None:
        valid = key_mask[:, None, None, :].to(q.device)
        if attn_mask is None:
            attn_mask = valid
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask & valid
        else:
            attn_mask = attn_mask.masked_fill(~valid, -torch.inf)
    # Flash backward can change the noise for a fixed seed. Use deterministic
    # math attention with bounded query chunks on the optimization path.
    outputs = []
    chunk = options.get("oracle_chunk", 128)
    result_dtype = q.dtype
    if _ATTENTION_PRECISION in ('bf16', 'fp16'):
        compute_dtype = torch.bfloat16 if _ATTENTION_PRECISION == 'bf16' else torch.float16
        q, k, v = q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype)
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, q.shape[-2], chunk):
            local_mask = attn_mask
            if local_mask is not None and local_mask.shape[-2] > 1:
                local_mask = local_mask[..., start:start + chunk, :]
            outputs.append(F.scaled_dot_product_attention(q[..., start:start + chunk, :], k, v,
                                                          attn_mask=local_mask))
    out = torch.cat(outputs, dim=-2).to(result_dtype)
    return out.transpose(1, 2).flatten(2)


def anima_qkv(module, x, context=None, rope_emb=None, transformer_options=None):
    from comfy.ldm.moss_image.model import rope_split_half

    context = x if context is None else context
    q = module.q_norm(module.q_proj(x).unflatten(-1, (module.n_heads, module.head_dim)))
    cache = (transformer_options or {}).get("oracle_static_kv") if not module.is_selfattn else None
    cached = cache.get(id(module)) if cache is not None else None
    if cached is None:
        k = module.k_norm(module.k_proj(context).unflatten(-1, (module.n_heads, module.head_dim)))
        v = module.v_norm(module.v_proj(context).unflatten(-1, (module.n_heads, module.head_dim)))
        if cache is not None:
            cache[id(module)] = (k.detach(), v.detach())
    else:
        k, v = cached
    if module.is_selfattn and rope_emb is not None:
        q, k = rope_split_half(q, rope_emb), rope_split_half(k, rope_emb)
    return q, k, v


def anima_attn(module, q, k, v, transformer_options=None, **kwargs):
    from comfy.ldm.modules.attention import AttentionTensorContainer

    if isinstance(q, AttentionTensorContainer):
        q, k, v = q.take(), k.take(), v.take()
    options = dict(transformer_options or {})
    if module.is_selfattn:
        options.pop("oracle_sink", None)
        options.pop("oracle_key_mask", None)
    # Cosmos q/k/v are [batch, positions, heads, head_dim].
    return torch_attention(None, q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                           module.n_heads, skip_reshape=True, transformer_options=options)


@contextmanager
def differentiable_model(model, family, memory_required_bytes=None, force_full_load=False,
                         native_quantized=False):
    """Use ModelPatcher object patches; never replace methods on shared modules directly."""
    import comfy.model_management as mm

    delegate = model.clone(disable_dynamic=True)
    diffusion = delegate.get_model_object("diffusion_model")
    previous_training, previous_fp8 = mm.in_training, mm.training_fp8_bwd
    try:
        supported_layouts = {"TensorCoreFP8Layout", "TensorCoreFP8E4M3Layout", "TensorCoreFP8E5M2Layout", "TensorWiseINT8Layout"}
        for module in diffusion.modules():
            layout = getattr(module, "layout_type", None)
            if layout is not None and layout not in supported_layouts:
                raise ValueError(f"Oracle Noise does not support diffusion quantization layout {layout}.")
        # Loader nodes run in inference mode. Frozen parameters still need normal
        # version counters when autograd saves them for input gradients.
        for name, parameter in diffusion.named_parameters():
            if torch.is_inference(parameter):
                delegate.add_object_patch(f"diffusion_model.{name}",
                                          torch.nn.Parameter(parameter.detach().clone(), requires_grad=False))
        for name, buffer in diffusion.named_buffers():
            if torch.is_inference(buffer):
                delegate.add_object_patch(f"diffusion_model.{name}", buffer.detach().clone())
        if family == "anima":
            for index, block in enumerate(diffusion.blocks):
                for name in ("self_attn", "cross_attn"):
                    attention = getattr(block, name)
                    prefix = f"diffusion_model.blocks.{index}.{name}"
                    delegate.add_object_patch(prefix + ".compute_qkv", functools.partial(anima_qkv, attention))
                    delegate.add_object_patch(prefix + ".attn_op", functools.partial(anima_attn, attention))
        # Frozen quantized matrices with unsupported inference-only dispatch are
        # dequantized by the existing ComfyUI cast path, preserving weight patches.
        if not native_quantized:
            for name, module in diffusion.named_modules():
                if hasattr(module, "_full_precision_mm"):
                    delegate.add_object_patch(f"diffusion_model.{name}._full_precision_mm", True)
        mm.in_training, mm.training_fp8_bwd = True, False
        mm.unload_all_models()
        mm.load_models_gpu([delegate], memory_required=(memory_required_bytes if memory_required_bytes is not None
                                                       else 2 * 1024**3), force_patch_weights=True,
                           force_full_load=force_full_load)
        # Loading a partially resident model may defer casts; its native cast
        # helpers handle them again when checkpointed blocks are recomputed.
        yield delegate, diffusion
    finally:
        try:
            mm.unload_all_models()
        finally:
            mm.in_training, mm.training_fp8_bwd = previous_training, previous_fp8


class AttentionAdapter:
    def __init__(self, delegate, diffusion, family, weights, masks, layers, cfg, chunk, interrupt,
                 cache_static=False, checkpoint_policy="all", query_stride=1):
        self.delegate, self.model, self.family = delegate, diffusion, family
        self.weights, self.masks, self.layers = weights, masks, layers
        self.cfg, self.chunk, self.interrupt = cfg, chunk, interrupt
        self._mask_cache = {}
        self.cache_static, self.checkpoint_policy, self.query_stride = cache_static, checkpoint_policy, query_stride
        self._static_cache = {}
        self._static_kv = {} if cache_static else None
        self._query_index_cache = {}

    def run_block(self, run, value, index):
        if self.checkpoint_policy == "all" or (self.checkpoint_policy == "alternate" and index % 2 == 0):
            return checkpoint(run, value, use_reentrant=False)
        if self.checkpoint_policy in ("none", "alternate"):
            return run(value)
        raise ValueError(f"Unknown checkpoint policy: {self.checkpoint_policy}")

    def options(self, sink, image_queries, text_start=0, grid=None, layer_index=0):
        if text_start not in self._mask_cache:
            self._mask_cache[text_start] = (F.pad(self.masks, (text_start, 0), value=True)
                                            if text_start else self.masks)
        masks = self._mask_cache[text_start]
        result = {"optimized_attention_override": torch_attention,
                "oracle_sink": sink, "oracle_queries": image_queries,
                "oracle_weights": self.weights, "oracle_cfg": self.cfg,
                "oracle_chunk": self.chunk, "oracle_text_start": text_start,
                "oracle_key_mask": masks}
        if self._static_kv is not None:
            result["oracle_static_kv"] = self._static_kv
        if sink is not None and self.query_stride > 1:
            if grid is None or grid[0] * grid[1] != image_queries:
                raise ValueError("Query sampling requires the image attention grid.")
            key = (grid, layer_index)
            if key not in self._query_index_cache:
                height, width = grid
                rows = torch.arange(height, device=self.weights.device)[:, None]
                columns = torch.arange(width, device=self.weights.device)[None, :]
                selected = ((rows * 131 + columns * 17 + layer_index * 7) % self.query_stride == 0)
                self._query_index_cache[key] = selected.flatten().nonzero().flatten()
            result["oracle_query_indices"] = self._query_index_cache[key]
        return result

    def objective(self, x, timestep, context):
        if self.family == "anima":
            return self.anima(x, timestep, context)
        return self.moss(x, timestep, context)

    def anima(self, x, timestep, context):
        import comfy.ldm.common_dit

        model = self.model
        if x.ndim == 4:
            x = x.unsqueeze(2)
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (model.patch_temporal, model.patch_spatial, model.patch_spatial))
        x, rope, extra = model.prepare_embedded_sequence(x, fps=None, padding_mask=None)
        emb, lora = model.t_embedder[1](model.t_embedder[0](timestep[:, None]).to(x.dtype))
        emb = model.t_embedding_norm(emb)
        if x.dtype == torch.float16:
            x = x.float()
        total = x.new_zeros((), dtype=torch.float32)
        for index, block in enumerate(model.blocks):
            self.interrupt()
            def run(value, block=block, index=index):
                sink = []
                opts = self.options(sink if index in self.layers else None,
                                    value.shape[1] * value.shape[2] * value.shape[3],
                                    grid=(value.shape[2], value.shape[3]), layer_index=index)
                out = block(value, emb, context, rope_emb_L_1_1_D=rope.unsqueeze(1).unsqueeze(0),
                            adaln_lora_B_T_3D=lora, extra_per_block_pos_emb=extra,
                            transformer_options=opts)
                loss = sink[0] if sink else value.new_zeros(())
                return out, loss
            x, loss = self.run_block(run, x, index)
            if index in self.layers:
                total = total + self.layers[index] * loss
            if index == max(self.layers):
                break
        return total

    def moss(self, x, timestep, context):
        from comfy.ldm.flux.layers import timestep_embedding

        model = self.model
        height, width = x.shape[-2:]
        image_queries = height * width
        static_key = (height, width, context.shape[1], x.dtype, x.device)
        static = self._static_cache.get(static_key) if self.cache_static else None
        if static is None:
            pe = model.rope(height, width, context.shape[1], x.device)
            vec = model.time_embedding(timestep_embedding(timestep, 256).to(x.dtype))
            mod = model.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
            # Templates are excluded from scoring but remain valid attention context.
            txt = model.text_embedder(context, pe[:, image_queries:],
                                      {"optimized_attention_override": torch_attention,
                                       "oracle_key_mask": self.masks})
            if self.cache_static:
                self._static_cache[static_key] = (pe.detach(), tuple(item.detach() for item in mod), txt.detach())
        else:
            pe, mod, txt = static
        x = torch.cat((model.video_embedder(x.flatten(2).transpose(1, 2)), txt), dim=1)
        total = x.new_zeros((), dtype=torch.float32)
        for index, block in enumerate(model.layers):
            self.interrupt()
            def run(value, block=block, index=index):
                sink = []
                opts = self.options(sink if index in self.layers else None, image_queries, image_queries,
                                    grid=(height, width), layer_index=index)
                out = block(value, mod, pe, opts)
                return out, sink[0] if sink else value.new_zeros(())
            x, loss = self.run_block(run, x, index)
            if index in self.layers:
                total = total + self.layers[index] * loss
            if index == max(self.layers):
                break
        return total
