from contextlib import contextmanager
import dataclasses
import functools
import threading

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from .math import attention_objective


OPTIMIZATION_LOCK = threading.RLock()
_ATTENTION_PRECISION = 'fp32'


def _normal_frozen_tensor(tensor):
    """Give autograd a normal tensor while sharing plain frozen storage.

    Tensor subclasses can carry layout/dispatch metadata that set_ would lose;
    retain their existing clone behavior instead.
    """
    if type(tensor) in (torch.Tensor, torch.nn.Parameter) and tensor.layout == torch.strided:
        return torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(tensor)
    return tensor.detach().clone()


def _shared_quantized_tensor(tensor):
    """Share supported Comfy quantized storage with a normal tensor wrapper."""
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return None
    layouts = {"TensorCoreFP8Layout", "TensorCoreFP8E4M3Layout", "TensorCoreFP8E5M2Layout", "TensorWiseINT8Layout"}
    if type(tensor) is not QuantizedTensor or getattr(tensor, "_layout_cls", None) not in layouts:
        return None
    params = getattr(tensor, "_params", None)
    data = getattr(tensor, "_qdata", None)
    fields = getattr(params, "_tensor_fields", None)
    if not dataclasses.is_dataclass(params) or not callable(fields) or type(data) is not torch.Tensor or data.layout != torch.strided:
        return None
    names = fields()
    values = [getattr(params, name, None) for name in names]
    if any(type(value) is not torch.Tensor or value.layout != torch.strided for value in values):
        return None
    replacements = {name: _normal_frozen_tensor(value) if torch.is_inference(value) else value
                    for name, value in zip(names, values)}
    return QuantizedTensor(_normal_frozen_tensor(data), tensor._layout_cls,
                           dataclasses.replace(params, **replacements))


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
def _non_dynamic_model(model, family, memory_required_bytes=None, force_full_load=False,
                       native_quantized=False):
    """Use ModelPatcher object patches; never replace methods on shared modules directly."""
    import comfy.model_management as mm

    is_dynamic = getattr(model, "is_dynamic", None)
    cached_delegate = getattr(model, "get_non_dynamic_delegate", None)
    if callable(is_dynamic) and is_dynamic() and callable(cached_delegate):
        delegate = cached_delegate()
    else:
        delegate = model.clone(disable_dynamic=True)
    diffusion = delegate.get_model_object("diffusion_model")
    previous_training, previous_fp8 = mm.in_training, mm.training_fp8_bwd
    completed = False
    try:
        supported_layouts = {"TensorCoreFP8Layout", "TensorCoreFP8E4M3Layout", "TensorCoreFP8E5M2Layout", "TensorWiseINT8Layout"}
        for module in diffusion.modules():
            layout = getattr(module, "layout_type", None)
            if layout is not None and layout not in supported_layouts:
                raise ValueError(f"Oracle Noise does not support diffusion quantization layout {layout}.")
        # Loader nodes run in inference mode. Autograd needs normal tensor
        # wrappers, but plain frozen weights can share their existing storage.
        share_quantized = (getattr(getattr(delegate, "load_device", None), "type", None) == "cuda"
                           and not getattr(delegate, "patches", {})
                           and not getattr(delegate, "weight_inplace_update", False))
        for name, parameter in diffusion.named_parameters():
            if torch.is_inference(parameter):
                normal = _shared_quantized_tensor(parameter) if share_quantized else None
                if normal is None:
                    normal = _normal_frozen_tensor(parameter)
                delegate.add_object_patch(f"diffusion_model.{name}",
                                          torch.nn.Parameter(normal, requires_grad=False))
        for name, buffer in diffusion.named_buffers():
            if torch.is_inference(buffer):
                delegate.add_object_patch(f"diffusion_model.{name}", _normal_frozen_tensor(buffer))
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
        completed = True
    finally:
        cleaned = False
        try:
            try:
                # Restore originals before offload to avoid a redundant host copy.
                if getattr(delegate, "object_patches_backup", None):
                    delegate.unpatch_model()
            finally:
                try:
                    mm.unload_all_models()
                finally:
                    mm.in_training, mm.training_fp8_bwd = previous_training, previous_fp8
            cleaned = True
        finally:
            if not completed or not cleaned:
                cached = getattr(model, "non_dynamic_delegate_model", None)
                if isinstance(cached, tuple) and cached and cached[0] is getattr(delegate, "model", None):
                    model.non_dynamic_delegate_model = None


class _DynamicUnavailable(Exception):
    pass


@functools.lru_cache(maxsize=8)
def _known_dynamic_runtime(functions, aimdo_version):
    # These loader internals are not a stable public API.  Only use the
    # verified implementation, particularly its owned synchronous fallback.
    import hashlib
    import inspect
    expected = (
        '91b16dd6bf1875dc1f6e6f85b8fe1614f7f4f69ecc0a02a5df7a100a711df2e3',
        'b3c9555cd6d710208d6e598258916706270ab5286a06e1023f92ec03958c74ca',
        '31fed5cc6ff1675b653f591e47d440d19888f70039e69b1db5579f6ec1de572d',
        '5c5feabea382a06756df6a9b57f5786a23e9c70a4651175fb61ebf68bef759a2',
        'a6343db79b5d13572f73a4e07dbc2527e52f017e01dca4b192bdb6e0ffa4ef61',
        'e9f73c2a374de624da14e3dce28f2d88ab779da5ee1c47600daecc2f58712384',
        '315d9bdaf8545d329c3f6744493c013271894e8661aec194fb6de436666017f2',
        '0cbb112aa2d6e2d5b38d574122f9ec9ccbbabaa2544ec6cc2db588f5bdbdc69c',
        '98b1b830217c543d6ff89ef5c4c5be50c20e83ba50447b8771591e44da0d09a7',
        '04fa0ab39f7265e6c0bb28f09a04aef2f76c5a268de0ca1a4ad9ea121e3ff7b2',
        '47d9a35d858eaee8f83b28506d444e105374eb2ab05b56349f5c0c1b631db496',
        '6f34b21a1de6c72fa9fcf89505575274614ad9d126cc36e9f588643d3184bf20',
        '96593852145546638a40afe721d0672b2e86324c44d5dc179aa56a5923a93cca',
    )
    if aimdo_version != '0.6.0':
        return False
    try:
        verified = []
        for fn in functions:
            if getattr(fn, '_torchdynamo_disable', False):
                inner = getattr(fn, '__wrapped__', None)
                if inner is not getattr(fn, '_torchdynamo_orig_callable', None):
                    return False
                fn = inner
            verified.append(hashlib.sha256(inspect.getsource(fn).encode()).hexdigest())
        return tuple(verified) == expected
    except (OSError, TypeError):
        return False


def _known_compile_wrapper(model):
    import hashlib
    import inspect
    if set(model.model_options) != {'transformer_options', 'torch_compile_kwargs'}:
        return False
    if model.model_options['transformer_options'] != {} or set(model.wrappers) != {'apply_model'}:
        return False
    wrappers = model.wrappers['apply_model']
    if set(wrappers) != {'torch.compile'} or len(wrappers['torch.compile']) != 1:
        return False
    try:
        source = inspect.getsource(wrappers['torch.compile'][0]).encode()
        return hashlib.sha256(source).hexdigest() == 'eb71dfd57789b10f928ccef6a205a3ec97bbb7f8bf9773a7ce8f45a55fc9d887'
    except (OSError, TypeError):
        return False


@functools.lru_cache(maxsize=8)
def _known_lora_runtime(functions):
    import hashlib
    import inspect
    expected = ('0445fca1a83cf542e804d76fbe19e90560ebca3256d0d641d4ebca1c8ed01362',
                '0f0e7a97eaff261d5e9d39536ef47219230f2e175ede86c598451176b97fe85f',
                'c6ba63543c0702e33f25cced28920efb0de331aefc49e042820fc9931c8c7fd2',
                '592b778b1760434ffa52e1d40f0ca121af6a6e4e8389d08dcf3f8122872adc4c')
    try:
        return tuple(hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for fn in functions) == expected
    except (OSError, TypeError):
        return False


@functools.lru_cache(maxsize=8)
def _known_quant_lora_runtime(module_type, tensor_type, kitchen_version):
    import hashlib
    import inspect
    if kitchen_version != '0.2.40':
        return False
    expected = ('930c1859094fd5c37b932ea328791ceef2e6660876d8d303e5c6df6c07e1d3a2',
                '1f59489d17eb328f589744842ed39596e346a55cf63c7c1457d4215bd0fa8058',
                '80a8de370c922eaf16dba92d5c3c1ae067c5120756c267bbfea68f66ff5d2a15',
                'd04e87b277499eda69a3b4cdfc639434d9a4f7e0b1bd4e2224d7faa2f7b109ec',
                'b27fa86ae98dd3eba6bbc443f2422494938b9545bb89b5cebe598b393832ca57',
                '7ca4cdd85ff37de0a2e90169ef0bd583bc79a2b4ec205d0539b71da29a8271af')
    functions = (module_type.forward, module_type.forward_comfy_cast_weights, module_type._forward,
                 module_type.set_weight, module_type.convert_weight, tensor_type.requantize_from_float)
    try:
        return tuple(hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for fn in functions) == expected
    except (OSError, TypeError):
        return False


def _supported_lora_patches(model, family):
    if not model.patches:
        return True
    if family not in ('anima', 'moss'):
        return False
    try:
        import re
        import importlib.metadata
        import comfy.model_management as mm
        import comfy.model_patcher as patcher
        import comfy.lora
        import comfy.utils
        from comfy.quant_ops import QuantizedTensor
        from comfy.weight_adapter.lora import LoRAAdapter
        if not _known_lora_runtime((patcher.LowVramPatch, patcher.ModelPatcher.patch_weight_to_device,
                                    mm.lora_compute_dtype, comfy.lora.calculate_weight)):
            return False
        if mm.lora_compute_dtype(model.load_device) != torch.float16:
            return False
        for key, entries in model.patches.items():
            if family == 'anima':
                match = re.fullmatch(r'diffusion_model\.blocks\.(\d+)\.(?:self_attn|cross_attn)\.'
                                     r'(?:q_proj|k_proj|v_proj|output_proj)\.weight|'
                                     r'diffusion_model\.blocks\.(\d+)\.mlp\.layer[12]\.weight', key)
                if match is None or int(match.group(1) or match.group(2)) >= len(model.model.diffusion_model.blocks):
                    return False
            else:
                match = re.fullmatch(r'diffusion_model\.layers\.(\d+)\.proj\.'
                                     r'(?:qkv|o|ffn_up_gate|ffn_down)\.weight', key)
                if match is None or int(match.group(1)) >= len(model.model.diffusion_model.layers):
                    return False
            weight, setter, converter = patcher.get_key_weight(model.model, key)
            if family == 'anima':
                if (weight is None or type(weight) not in (torch.Tensor, torch.nn.Parameter)
                        or weight.dtype != torch.float16 or weight.ndim != 2 or setter is not None or converter is not None):
                    return False
            else:
                module = comfy.utils.get_attr(model.model, key.rsplit('.', 1)[0])
                if (type(weight) is not QuantizedTensor or weight.dtype != torch.bfloat16 or weight.ndim != 2
                        or getattr(module, 'layout_type', None) != 'TensorWiseINT8Layout'
                        or getattr(module, 'bias', None) is not None or setter is None or converter is None):
                    return False
                params = weight._params
                if (not params.is_weight or not params.convrot or params.convrot_groupsize != 256 or params.transposed
                        or not _known_quant_lora_runtime(type(module), QuantizedTensor,
                                                        importlib.metadata.version('comfy-kitchen'))):
                    return False
            for entry in entries:
                if len(entry) != 5 or type(entry[1]) is not LoRAAdapter or entry[2:] != (1.0, None, None):
                    return False
                weights = entry[1].weights
                if len(weights) != 6 or any(weights[i] is not None for i in (3, 4, 5)):
                    return False
                if weights[0].ndim != 2 or weights[1].ndim != 2:
                    return False
        return True
    except (ImportError, AttributeError, KeyError, TypeError, ValueError):
        return False


class _ExactDynamicLora:
    """Use the delegate's FP32 LoRA intermediate while sharing its pin state."""
    is_lowvram_patch = True

    def __init__(self, original):
        self.__dict__ = original.__dict__

    def memory_required(self):
        from comfy.model_patcher import LowVramPatch
        return LowVramPatch.memory_required(self)

    def prepare(self, *args, **kwargs):
        from comfy.model_patcher import LowVramPatch
        return LowVramPatch.prepare(self, *args, **kwargs)

    def clear_prepared(self):
        from comfy.model_patcher import LowVramPatch
        return LowVramPatch.clear_prepared(self)

    def __call__(self, weight):
        import comfy.lora
        patches = self.prepared_patches if self.prepared_patches is not None else self.patches[self.key]
        return comfy.lora.calculate_weight(patches, weight, self.key, intermediate_dtype=torch.float32)


def _moss_quantized_lora_forward(module, delegate, key, signatures, input, compute_dtype=None,
                                  want_requant=False, weight_only_quant=False):
    import comfy.lora
    import comfy.model_management as mm
    import comfy.ops
    import comfy.utils
    from comfy.quant_ops import QuantizedTensor

    if weight_only_quant or want_requant:
        raise RuntimeError('Unsupported quantized LoRA execution mode')
    with comfy.ops.CastBiasWeightContext(module, input, offloadable=True,
                                        compute_dtype=compute_dtype, want_requant=False) as (_, bias):
        signature = getattr(module, '_v_signature', None)
        if signature is None:
            weight = delegate.patch_weight_to_device(key, device_to=delegate.load_device, return_weight=True)
        else:
            if signatures.get(key) != signature:
                dtype = mm.lora_compute_dtype(delegate.load_device)
                value = mm.cast_to_device(module._v_weight, delegate.load_device, dtype, copy=True)
                value = module.convert_weight(value, inplace=True)
                value = comfy.lora.calculate_weight(delegate.patches[key], value, key)
                weight = module.set_weight(value, inplace_update=False,
                                           seed=comfy.utils.string_to_seed(key), return_weight=True)
                with torch.no_grad():
                    module._v_weight.copy_(weight)
                signatures[key] = signature
            weight = module._v_weight
        weight = weight.to(dtype=input.dtype)
        if isinstance(weight, QuantizedTensor):
            weight = weight.dequantize()
        return module._forward(input, weight, bias)


def _unload_other_models(mm, model):
    loaded = list(getattr(mm, 'current_loaded_models', ()))
    keep = []
    other_present = False
    for item in loaded:
        patcher = item.model
        if patcher is None:
            continue
        if patcher.is_clone(model):
            keep.append(item)
        else:
            other_present = True
    if other_present:
        for device in mm.get_all_torch_devices():
            mm.free_memory(1e30, device, keep_loaded=keep)


def _dynamic_eligible(model, family, force_full_load, native_quantized):
    try:
        import importlib.metadata
        import comfy.memory_management as memory
        import comfy.model_management as mm
        import comfy.ops as ops
        from comfy.model_patcher import ModelPatcherDynamic
        import comfy_aimdo.model_vbar as vbar
        aimdo_version = importlib.metadata.version('comfy-aimdo')
    except (ImportError, importlib.metadata.PackageNotFoundError):
        return False
    required_model = ('load_device', 'object_patches', 'weight_wrapper_patches', 'weight_inplace_update',
                      'force_cast_weights', 'is_injected', 'is_multigpu_base_clone', 'model_options',
                      'wrappers', 'patches', 'model', 'clone', 'get_model_object', '_vbar_get',
                      'add_object_patch', 'unpatch_model')
    required_management = ('NUM_STREAMS', 'in_training', 'training_fp8_bwd', 'is_nvidia',
                           'get_all_torch_devices', 'free_memory', 'load_models_gpu', 'soft_empty_cache')
    if (type(model) is not ModelPatcherDynamic
            or any(not hasattr(model, name) for name in required_model)
            or any(not hasattr(mm, name) for name in required_management)
            or any(not callable(getattr(model, name, None)) for name in
                   ('clone', 'get_model_object', '_vbar_get', 'add_object_patch', 'unpatch_model'))
            or any(not callable(getattr(mm, name, None)) for name in
                   ('get_all_torch_devices', 'free_memory', 'load_models_gpu', 'soft_empty_cache'))
            or any(not callable(getattr(ops, name, None)) for name in
                   ('cast_modules_with_vbar', 'resolve_cast_module_with_vbar', 'uncast_bias_weight'))
            or not callable(getattr(mm, 'get_offload_stream', None))
            or not callable(getattr(mm, 'is_nvidia', None))):
        return False
    if (not getattr(memory, 'aimdo_enabled', False)
            or mm.in_training or _ATTENTION_PRECISION != 'fp32' or force_full_load
            or native_quantized or getattr(model.load_device, 'type', None) != 'cuda' or not mm.is_nvidia()
            or model.object_patches or model.weight_wrapper_patches
            or model.weight_inplace_update or model.force_cast_weights or model.is_injected
            or model.is_multigpu_base_clone
            or getattr(model, 'non_dynamic_delegate_model', None) is not None):
        return False
    if not _supported_lora_patches(model, family):
        return False
    if model.model_options != {'transformer_options': {}} or model.wrappers:
        if not _known_compile_wrapper(model):
            return False
    if any(getattr(model, name, None) for name in ('attachments', 'additional_models', 'callbacks',
                                                    'injections', 'current_hooks', 'forced_hooks',
                                                    'hook_patches', 'hook_patches_backup', 'hook_backup',
                                                    'cached_hook_patches')):
        return False
    if not all(callable(getattr(vbar, name, None)) for name in ('vbar_fault', 'vbar_unpin')):
        return False
    functions = (ops.cast_modules_with_vbar, ops.resolve_cast_module_with_vbar, ops.uncast_bias_weight,
                 mm.get_offload_stream, vbar.vbar_fault, vbar.vbar_unpin, mm.free_memory, mm.soft_empty_cache,
                 ModelPatcherDynamic.load, ModelPatcherDynamic._vbar_get, ModelPatcherDynamic.clone,
                 ModelPatcherDynamic.unpatch_model, ModelPatcherDynamic.add_object_patch)
    if not _known_dynamic_runtime(functions, aimdo_version):
        return False
    try:
        diffusion = model.get_model_object('diffusion_model')
    except (AttributeError, KeyError):
        return False
    if (family, type(diffusion).__module__, type(diffusion).__name__) not in {
            ('anima', 'comfy.ldm.anima.model', 'Anima'), ('moss', 'comfy.ldm.moss_image.model', 'MOSSImage')}:
        return False
    expected_dtype = torch.float16 if family == 'anima' else torch.bfloat16
    if model.model.get_dtype_inference() != expected_dtype:
        return False
    allowed = {'disable_weight_init.Linear', 'disable_weight_init.RMSNorm',
               'disable_weight_init.LayerNorm', 'disable_weight_init.Embedding',
               'manual_cast.RMSNorm', 'mixed_precision_ops.<locals>.MixedPrecisionOps.Linear'}
    found_int8 = False
    for module in diffusion.modules():
        if hasattr(module, 'comfy_cast_weights') and (type(module).__module__ != 'comfy.ops'
                                                        or type(module).__qualname__ not in allowed):
            return False
        layout = getattr(module, 'layout_type', None)
        if layout not in (None, 'TensorWiseINT8Layout'):
            return False
        found_int8 |= layout == 'TensorWiseINT8Layout'
        if family == 'anima' and layout is not None:
            return False
        if layout is None:
            for name in ('weight', 'bias'):
                value = getattr(module, name, None)
                dtype = getattr(module, name + '_comfy_model_dtype', None)
                if value is not None and dtype is not None and value.dtype != dtype:
                    # The MOSS RMSNorm loader casts BF16 checkpoint weights to
                    # FP16 in both paths, on CPU or in the dynamic allocation.
                    if not (family == 'moss' and type(module).__qualname__ == 'manual_cast.RMSNorm'
                            and value.dtype == torch.bfloat16 and dtype == torch.float16):
                        return False
    return family != 'moss' or found_int8


@contextmanager
def _dynamic_model(model, family, memory_required_bytes):
    import comfy.model_management as mm
    import comfy_aimdo.model_vbar as vbar_api

    delegate = model.clone()
    diffusion = delegate.get_model_object('diffusion_model')
    old_training, old_fp8, old_streams = mm.in_training, mm.training_fp8_bwd, mm.NUM_STREAMS
    held, backups, patch_backups = [], [], []
    quant_signatures = {}
    dynamic_vbar = None
    ready = False
    try:
        for name, parameter in diffusion.named_parameters():
            if torch.is_inference(parameter):
                normal = _shared_quantized_tensor(parameter)
                if normal is None:
                    normal = _normal_frozen_tensor(parameter)
                delegate.add_object_patch(f'diffusion_model.{name}',
                                          torch.nn.Parameter(normal, requires_grad=False))
        for name, buffer in diffusion.named_buffers():
            if torch.is_inference(buffer):
                delegate.add_object_patch(f'diffusion_model.{name}', _normal_frozen_tensor(buffer))
        if family == 'anima':
            for index, block in enumerate(diffusion.blocks):
                for name in ('self_attn', 'cross_attn'):
                    attention = getattr(block, name)
                    prefix = f'diffusion_model.blocks.{index}.{name}'
                    delegate.add_object_patch(prefix + '.compute_qkv', functools.partial(anima_qkv, attention))
                    delegate.add_object_patch(prefix + '.attn_op', functools.partial(anima_attn, attention))
        for name, module in diffusion.named_modules():
            if hasattr(module, '_full_precision_mm'):
                delegate.add_object_patch(f'diffusion_model.{name}._full_precision_mm', True)
        if family == 'moss' and delegate.patches:
            import comfy.utils
            for key in delegate.patches:
                name = key.rsplit('.', 1)[0]
                module = comfy.utils.get_attr(delegate.model, name)
                delegate.add_object_patch(name + '.forward_comfy_cast_weights',
                                          functools.partial(_moss_quantized_lora_forward, module, delegate,
                                                            key, quant_signatures))
        mm.in_training, mm.training_fp8_bwd = True, False
        # Release encoders without discarding the dynamic model's reusable
        # host state between calls.
        _unload_other_models(mm, model)
        # With no offload stream, Comfy allocates owned fallback cast buffers.
        # Reusable stream buffers can otherwise be overwritten before backward.
        mm.NUM_STREAMS = 0
        mm.load_models_gpu([delegate], memory_required=(memory_required_bytes if memory_required_bytes is not None
                                                        else 2 * 1024**3),
                           force_patch_weights=False, force_full_load=False)
        dynamic_vbar = delegate._vbar_get()
        if dynamic_vbar is None or not callable(getattr(dynamic_vbar, 'free_memory', None)):
            raise _DynamicUnavailable('Dynamic allocation unavailable')
        if delegate.patches:
            from comfy.model_patcher import LowVramPatch
            if family == 'moss':
                import comfy.utils
                for key in delegate.patches:
                    module = comfy.utils.get_attr(delegate.model, key.rsplit('.', 1)[0])
                    patch = getattr(module, 'weight_lowvram_function', None)
                    if (getattr(module, '_v', None) is None or type(patch) is not LowVramPatch
                            or patch.key != key):
                        raise _DynamicUnavailable('Quantized patch needs a managed weight allocation')
            for module in delegate.model.modules():
                for attr in ('weight_lowvram_function', 'bias_lowvram_function'):
                    original = getattr(module, attr, None)
                    if original is None:
                        continue
                    if type(original) is not LowVramPatch or original.key not in delegate.patches:
                        raise _DynamicUnavailable('Unsupported dynamic weight patch')
                    replacement = None if family == 'moss' else _ExactDynamicLora(original)
                    patch_backups.append((module, attr, original, replacement))
                    setattr(module, attr, replacement)
                    module._v_signature = None
        seen = set()
        for module in delegate.model.modules():
            allocation = getattr(module, '_v', None)
            if allocation is None:
                continue
            if id(allocation) not in seen:
                if vbar_api.vbar_fault(allocation) is None:
                    raise _DynamicUnavailable('Dynamic allocation could not be pinned')
                held.append(allocation)
                seen.add(id(allocation))
            for attr in ('_v_weight', '_v_bias'):
                value = getattr(module, attr, None)
                if value is not None and torch.is_inference(value):
                    normal = _shared_quantized_tensor(value)
                    if normal is None:
                        if type(value) not in (torch.Tensor, torch.nn.Parameter) or value.layout != torch.strided:
                            raise _DynamicUnavailable('Unsupported cached dynamic tensor')
                        normal = _normal_frozen_tensor(value)
                    backups.append((module, attr, value, normal))
                    setattr(module, attr, normal)
        if not held:
            raise _DynamicUnavailable('No dynamic allocations found')
        ready = True
        yield delegate, diffusion
    finally:
        try:
            try:
                if held:
                    torch.cuda.synchronize(delegate.load_device)
            finally:
                try:
                    for module, attr, original, normal in reversed(backups):
                        if getattr(module, attr, None) is normal:
                            setattr(module, attr, original)
                finally:
                    try:
                        for module, attr, original, replacement in reversed(patch_backups):
                            if getattr(module, attr, None) is replacement:
                                setattr(module, attr, original)
                            module._v_signature = None
                    finally:
                        try:
                            unpin_error = None
                            for allocation in reversed(held):
                                try:
                                    vbar_api.vbar_unpin(allocation)
                                except BaseException as exc:
                                    if unpin_error is None:
                                        unpin_error = exc
                            if unpin_error is not None:
                                raise unpin_error
                        finally:
                            try:
                                if getattr(delegate, 'object_patches_backup', None) or not ready:
                                    delegate.unpatch_model(unpatch_weights=not ready)
                            finally:
                                try:
                                    if dynamic_vbar is not None:
                                        dynamic_vbar.free_memory(1e30)
                                finally:
                                    try:
                                        if not ready:
                                            mm.unload_all_models()
                                    finally:
                                        mm.soft_empty_cache(force=True)
        finally:
            mm.in_training, mm.training_fp8_bwd, mm.NUM_STREAMS = old_training, old_fp8, old_streams


@contextmanager
def differentiable_model(model, family, memory_required_bytes=None, force_full_load=False,
                         native_quantized=False, optimization_shape=None):
    # Large patched MOSS workloads can repeatedly reconstruct quantized
    # LoRA weights; retain the faster delegate for that configuration.
    large_patched_moss = (family == 'moss' and bool(getattr(model, 'patches', None))
                          and (optimization_shape is None or len(optimization_shape) < 2
                               or optimization_shape[-2] * optimization_shape[-1] > 32 * 32))
    if not large_patched_moss and _dynamic_eligible(model, family, force_full_load, native_quantized):
        candidate = _dynamic_model(model, family, memory_required_bytes)
        try:
            value = candidate.__enter__()
        except _DynamicUnavailable:
            pass
        else:
            try:
                yield value
            except BaseException as exc:
                if not candidate.__exit__(type(exc), exc, exc.__traceback__):
                    raise
            else:
                candidate.__exit__(None, None, None)
            return
    with _non_dynamic_model(model, family, memory_required_bytes, force_full_load, native_quantized) as value:
        yield value


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
