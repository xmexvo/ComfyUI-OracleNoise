import logging
import math
import time

import torch
import torch.nn.functional as F

from .adapters import OPTIMIZATION_LOCK, AttentionAdapter, attention_precision, differentiable_model, family_of, preset_layer_selection
from .conditioning import native_mapping, pad_contexts, validate_conditioning
from .color import colorize_noise
from .importance import score_prompt
from .math import map_importance, spherical_step
from .prompt import parse_prompt


def validate_latent(latent):
    samples = latent["samples"]
    if samples.is_nested or samples.ndim not in (4, 5) or (samples.ndim == 5 and samples.shape[2] != 1):
        raise ValueError("Oracle Noise supports image latents only (including Anima's singleton frame axis).")
    if any(size == 0 for size in samples.shape):
        raise ValueError("Oracle Noise requires nonempty latent dimensions.")
    if "noise_mask" in latent:
        raise ValueError("Oracle Noise does not support masked/inpainting latents.")
    if torch.count_nonzero(samples):
        raise ValueError("Oracle Noise requires an empty latent for text-to-image.")
    return samples


def proxy_noise(noise, factor, alignment=2, target=None, algorithm="bilinear"):
    if factor == 1 and target is None:
        return noise
    if factor not in (2, 4) and target is None:
        raise ValueError("Proxy factor must be 1, 2, or 4.")
    if alignment not in (1, 2) or any(size == 0 for size in noise.shape[-2:]):
        raise ValueError("Proxy dimensions must be nonempty and alignment must be 1 or 2.")
    shape = noise.shape
    flat = noise.reshape(-1, shape[-3], shape[-2], shape[-1]).float()
    if algorithm not in ("bilinear", "bicubic"):
        raise ValueError(f"Unknown resize algorithm: {algorithm}")
    if target is None:
        target = tuple(min(size, alignment * ((size + alignment * factor - 1) // (alignment * factor)))
                       for size in shape[-2:])
    if tuple(shape[-2:]) == tuple(target):
        return noise
    smaller = F.interpolate(flat, size=target, mode=algorithm, align_corners=False, antialias=True)
    source_mean = flat.mean(dim=(-2, -1), keepdim=True)
    centered = smaller - smaller.mean(dim=(-2, -1), keepdim=True)
    source_norm = torch.linalg.vector_norm(flat - source_mean, dim=(-2, -1), keepdim=True)
    resized_norm = torch.linalg.vector_norm(centered, dim=(-2, -1), keepdim=True)
    normalized = source_mean + centered * (source_norm / resized_norm.clamp_min(1e-20))
    fallback = smaller * math.sqrt((shape[-2] * shape[-1]) / (target[0] * target[1]))
    smaller = torch.where(resized_norm > 1e-20, normalized, fallback)
    return smaller.reshape(*shape[:-2], *target).to(noise.dtype)


def fixed_proxy_size(shape, alignment=2, pixels=512, latent_scale=8):
    """Target the specified image area, preserving aspect ratio without upsampling."""
    height, width = shape[-2:]
    scale = min(1.0, (pixels / latent_scale) / math.sqrt(height * width))
    if scale == 1.0:
        return height, width
    target = [min(size, max(alignment, alignment * round(size * scale / alignment)))
              for size in (height, width)]
    # Very thin images can hit the minimum dimension before reaching the target area.
    # Bound the other dimension as well, so the proxy cannot grow without limit.
    budget = (pixels / latent_scale) ** 2
    if target[0] * target[1] > budget:
        axis = 0 if target[0] >= target[1] else 1
        limit = max(1, int(budget // target[1 - axis]))
        if limit >= alignment:
            limit = alignment * (limit // alignment)
        target[axis] = min(target[axis], limit)
    return tuple(target)


def scaled_step_size(step_size, full_shape, proxy_shape, enabled):
    """Scale by the geometric mean of the actual spatial resize factors."""
    if not enabled:
        return step_size
    full_area = full_shape[-2] * full_shape[-1]
    proxy_area = proxy_shape[-2] * proxy_shape[-1]
    return step_size * math.sqrt(proxy_area / full_area)


def transport_proxy_update(full_noise, small_initial, small_optimized, algorithm="bilinear"):
    shape = full_noise.shape
    delta = (small_optimized.float() - small_initial.float()).reshape(
        -1, shape[-3], small_initial.shape[-2], small_initial.shape[-1])
    if algorithm not in ("bilinear", "bicubic"):
        raise ValueError(f"Unknown resize algorithm: {algorithm}")
    direction = F.interpolate(delta, size=shape[-2:], mode=algorithm, align_corners=False).reshape(shape)
    updated = []
    for original, initial, optimized, vector in zip(full_noise.split(1), small_initial.split(1),
                                                     small_optimized.split(1), direction.split(1)):
        a, b = initial.float().flatten(), optimized.float().flatten()
        cosine = ((a @ b) / (a.norm() * b.norm()).clamp_min(1e-20)).clamp(-1, 1)
        angle = math.acos(float(cosine))
        updated.append(spherical_step(original.float(), vector, angle).to(full_noise.dtype))
    return torch.cat(updated)


class OracleNoiseProvider:
    def __init__(self, model, native_encoder, importance_encoder, positive, negative, prompt,
                 seed, sigmas, iterations=5, angular_step=0.01, objective="conditional",
                 optimization_cfg=3.5, importance_mode="auto", layer_indices="",
                 layer_weights="1.0,1.5,2.0", query_chunk_size=2048, importance_batch_size=64,
                 attention_math="fp32", block_preset="Custom", noise_color="off",
                 color_strength=0.25, color_slope=0.0, speed_preset="normal", resolution="512", relative_resolution="off", resize_algorithm="bicubic", auto_step_size_scaling=True, conditioning_prompt=None, oracle_prompt_data=None, _study_cache_static=False,
                 _study_checkpoint_policy="all", _study_memory_required_bytes=None,
                 _study_query_stride=1, _study_proxy_factor=None, _study_force_full_load=False,
                 _study_cfg_schedule="full", _study_min_sync=False, _study_adaptive_tolerance=0.0,
                 _study_native_quantized=False):
        self.model, self.native_encoder, self.importance_encoder = model, native_encoder, importance_encoder
        self.positive, self.negative, self.prompt = positive, negative, prompt
        self.conditioning_prompt = conditioning_prompt
        self.oracle_prompt_data = oracle_prompt_data
        self.seed, self.sigmas, self.iterations = seed, sigmas, iterations
        self.angular_step, self.objective, self.optimization_cfg = angular_step, objective, optimization_cfg
        self.importance_mode = importance_mode
        self.layer_indices, self.layer_weights, self.query_chunk_size = layer_indices, layer_weights, query_chunk_size
        self.importance_batch_size = importance_batch_size
        self.attention_math = attention_math
        self.block_preset = block_preset
        self.noise_color, self.color_strength, self.color_slope = noise_color, color_strength, color_slope
        self._study_cache_static = _study_cache_static
        self._study_checkpoint_policy = _study_checkpoint_policy
        self._study_memory_required_bytes = _study_memory_required_bytes
        self._study_query_stride = _study_query_stride
        self.speed_preset = speed_preset
        resolution_factors = {"256": None, "512": None, "768": None, "1024": None, "1/4": 4, "1/2": 2, "full": 1}
        if resolution not in resolution_factors:
            raise ValueError(f"Unknown Oracle Noise resolution: {resolution}")
        if relative_resolution not in ("off", "full", "1/2", "1/4"):
            raise ValueError(f"Unknown relative resolution: {relative_resolution}")
        self.resolution = resolution
        self.relative_resolution = relative_resolution
        if resize_algorithm not in ("bilinear", "bicubic"):
            raise ValueError(f"Unknown resize algorithm: {resize_algorithm}")
        self.resize_algorithm = resize_algorithm
        self.auto_step_size_scaling = auto_step_size_scaling
        effective = resolution if relative_resolution == "off" else relative_resolution
        self._study_proxy_factor = resolution_factors[effective] if _study_proxy_factor is None else _study_proxy_factor
        self._study_force_full_load = _study_force_full_load
        self._study_cfg_schedule = "first" if speed_preset == "fast" and _study_cfg_schedule == "full" else _study_cfg_schedule
        self._study_min_sync = _study_min_sync
        self._study_adaptive_tolerance = _study_adaptive_tolerance
        self._study_native_quantized = _study_native_quantized
        self.last_report = None

    def generate_noise(self, latent):
        import comfy.model_management as mm
        import comfy.sample
        import comfy.utils

        samples = validate_latent(latent)
        noise = comfy.sample.prepare_noise(samples, self.seed, latent.get("batch_index"))
        if self.iterations == 0 or self.angular_step == 0:
            return colorize_noise(noise, self.noise_color, self.color_strength, self.color_slope)
        if self.objective not in ("conditional", "cfg_logits"):
            raise ValueError(f"Unknown attention objective: {self.objective}")
        if self.iterations < 0 or not torch.isfinite(torch.tensor(self.angular_step)):
            raise ValueError("Iterations must be nonnegative and angular step must be finite.")
        if self.sigmas.ndim != 1 or self.sigmas.numel() < 2 or not torch.isfinite(self.sigmas).all() or self.sigmas[0] <= 0:
            raise ValueError("Supply the same nonempty, finite SIGMAS schedule used by the sampler.")
        if not torch.isfinite(torch.tensor(self.optimization_cfg)) or self.optimization_cfg < 0:
            raise ValueError("Optimization guidance must be finite and nonnegative.")
        family = family_of(self.model)
        transformer_options = self.model.model_options.get("transformer_options", {})
        if any(bool(value) if isinstance(value, (dict, list, tuple)) else value is not None
               for value in transformer_options.values()):
            raise ValueError("Oracle Noise v1 supports weight LoRAs and sampling patches, not additional transformer behavior patches.")
        if self.query_chunk_size < 0:
            raise ValueError("Query chunk size must be nonnegative (0 selects the model default).")
        if not 1 <= self.importance_batch_size <= 64:
            raise ValueError("Importance batch size must be between 1 and 64.")
        if self.speed_preset not in ("normal", "fast"):
            raise ValueError("Speed preset must be normal or fast.")
        if self._study_query_stride not in (1, 2, 4):
            raise ValueError("Study query stride must be 1, 2, or 4.")
        if self._study_proxy_factor not in (None, 1, 2, 4):
            raise ValueError("Study proxy factor must be 1, 2, or 4.")
        if self._study_cfg_schedule not in ("full", "last", "first"):
            raise ValueError("Study CFG schedule must be full or last.")
        if not 0 <= self._study_adaptive_tolerance <= 1:
            raise ValueError("Study adaptive tolerance must be between 0 and 1.")
        if self._study_cfg_schedule == "last" and self._study_adaptive_tolerance:
            raise ValueError("Adaptive stopping cannot skip the final CFG-logit iteration.")
        if family == "moss" and samples.ndim != 4:
            raise ValueError("MOSS requires a 4D image latent.")
        effective_chunk_size = self.query_chunk_size or 2048
        positive = validate_conditioning(self.positive, "positive")
        cfg = self.optimization_cfg if self.objective == "cfg_logits" else 1.0
        negative_entry = validate_conditioning(self.negative, "negative")
        negative = negative_entry if cfg != 1 else None
        with OPTIMIZATION_LOCK, torch.inference_mode(False), torch.enable_grad():
            mm.throw_exception_if_processing_interrupted()
            data = self.oracle_prompt_data
            if data is not None:
                from .preprocessing import OraclePromptData
                if not isinstance(data, OraclePromptData) or data.conditioning_tensor is not positive[0]:
                    raise ValueError("Oracle prompt data must come from the node that produced positive conditioning.")
                parsed = data.parsed
                semantic, spans = native_mapping(self.native_encoder, family, data.original, self.positive,
                                                 parsed, data.encoder_text, data.layout)
            else:
                parsed = parse_prompt(self.prompt.strip() if family == "moss" else self.prompt)
                semantic, spans = native_mapping(self.native_encoder, family, self.prompt, self.positive, parsed, self.conditioning_prompt)
            char_scores, char_valid = self.importance_scores(semantic)
            if not torch.any(map_importance(char_scores, char_valid, spans) > 0):
                raise ValueError("Prompt has no scorable text tokens.")
            weights = map_importance(char_scores * parsed.weights, char_valid, spans)
            if not torch.isfinite(weights).all():
                raise ValueError("Weighted prompt importance must be finite.")
            if not torch.any(weights != 0):
                return colorize_noise(noise, self.noise_color, self.color_strength, self.color_slope)
            # Detach inference tensors and keep conditioning off GPU until needed.
            entries = [positive] + ([negative] if negative is not None else [])
            entries = [(c.detach().cpu().clone(), {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                                                  for k, v in meta.items()}) for c, meta in entries]
            alignment = 2 if family == "anima" else 1
            if self._study_proxy_factor is None:
                initial = proxy_noise(noise, None, alignment=alignment,
                                      target=fixed_proxy_size(noise.shape, alignment, int(self.resolution),
                                                              8 if family == "anima" else 16), algorithm=self.resize_algorithm)
            else:
                initial = proxy_noise(noise, self._study_proxy_factor, alignment=alignment, algorithm=self.resize_algorithm)
            effective_step = scaled_step_size(self.angular_step, noise.shape, initial.shape,
                                              self.auto_step_size_scaling)
            optimized = self.optimize(initial, family, weights, entries, cfg, effective_chunk_size,
                                      effective_step_size=effective_step)
            if initial.shape[-2:] != noise.shape[-2:]:
                optimized = transport_proxy_update(noise, initial, optimized, self.resize_algorithm)
            return colorize_noise(optimized, self.noise_color, self.color_strength, self.color_slope)

    def importance_scores(self, semantic):
        import comfy.model_management as mm

        return score_prompt(self.importance_encoder, self.importance_mode, semantic,
                            mm.throw_exception_if_processing_interrupted, self.importance_batch_size)

    def optimize(self, noise, family, weights, entries, cfg, effective_chunk_size, effective_step_size=None):
        import comfy.model_management as mm
        import comfy.utils

        started = time.perf_counter()
        if effective_step_size is None:
            effective_step_size = self.angular_step
        with attention_precision(self.attention_math), differentiable_model(
                self.model, family, self._study_memory_required_bytes,
                self._study_force_full_load, self._study_native_quantized,
                optimization_shape=noise.shape) as (delegate, diffusion):
            device, dtype = delegate.load_device, delegate.model.get_dtype_inference()
            if self.attention_math == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
                raise ValueError("BF16 attention requires a CUDA GPU with BF16 support.")
            if self.attention_math == "fp16" and device.type != "cuda":
                raise ValueError("FP16 attention requires CUDA.")
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            contexts = []
            with torch.no_grad():
                for context, metadata in entries:
                    context = context.to(device=device, dtype=dtype)
                    if family == "anima":
                        if "t5xxl_ids" not in metadata or "t5xxl_weights" not in metadata:
                            raise ValueError("Anima requires native T5 IDs and weights in conditioning.")
                        context = diffusion.preprocess_text_embeds(
                            context, metadata["t5xxl_ids"].to(device)[None],
                            metadata["t5xxl_weights"].to(device=device, dtype=dtype)[None, :, None])
                    contexts.append(context.detach().clone())
            context, masks = pad_contexts(contexts)
            # Anima's native zero padding participates in attention unchanged;
            # its score is zero through the importance vector, not a new mask.
            if weights.numel() < context.shape[1]:
                weights = torch.nn.functional.pad(weights, (0, context.shape[1] - weights.numel()))
            if weights.numel() != context.shape[1]:
                raise ValueError("Native token mapping does not match denoiser context length.")
            blocks = diffusion.blocks if family == "anima" else diffusion.layers
            layers = preset_layer_selection(len(blocks), family, self.block_preset, self.layer_indices, self.layer_weights)
            adapter = AttentionAdapter(delegate, diffusion, family, weights.to(device), masks,
                                       layers, cfg, effective_chunk_size, mm.throw_exception_if_processing_interrupted,
                                       cache_static=self._study_cache_static,
                                       checkpoint_policy=self._study_checkpoint_policy,
                                       query_stride=self._study_query_stride)
            conditional_adapter = None
            if self._study_cfg_schedule in ("last", "first") and cfg != 1:
                conditional_adapter = AttentionAdapter(
                    delegate, diffusion, family, weights.to(device), masks[:1], layers, 1.0,
                    effective_chunk_size, mm.throw_exception_if_processing_interrupted,
                    cache_static=self._study_cache_static,
                    checkpoint_policy=self._study_checkpoint_policy,
                    query_stride=self._study_query_stride)
            sampling = delegate.get_model_object("model_sampling")
            sigma = self.sigmas[:1].to(device=device).detach().clone()
            timestep = sampling.timestep(sigma).float().expand(context.shape[0]).clone()
            output, objectives, iteration_seconds = [], [], []
            iteration_events = []
            progress = comfy.utils.ProgressBar(noise.shape[0] * self.iterations)
            for original in noise.split(1):
                z = original.to(device=device, dtype=torch.float32).clone()
                previous_objective = None
                previous_direction = None
                for iteration_index in range(self.iterations):
                    mm.throw_exception_if_processing_interrupted()
                    if device.type == "cuda" and not self._study_min_sync:
                        torch.cuda.synchronize(device)
                    iteration_started = time.perf_counter()
                    if self._study_min_sync and device.type == "cuda":
                        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start_event.record()
                    z = z.detach().requires_grad_(True)
                    scaled = sampling.noise_scaling(sigma, z, torch.zeros_like(z))
                    model_input = sampling.calculate_input(sigma, scaled).to(dtype)
                    use_conditional = (self._study_cfg_schedule == "last" and iteration_index < self.iterations - 1) or (self._study_cfg_schedule == "first" and iteration_index >= self.iterations // 3)
                    active = conditional_adapter if conditional_adapter is not None and use_conditional else adapter
                    active_context = context[:1] if active is conditional_adapter else context
                    active_timestep = timestep[:1] if active is conditional_adapter else timestep
                    model_input = model_input.expand(active_context.shape[0], *model_input.shape[1:])
                    loss = active.objective(model_input, active_timestep, active_context)
                    if not loss.requires_grad or (not self._study_min_sync and not torch.isfinite(loss)):
                        raise RuntimeError("Attention objective has no finite gradient; unsupported model/kernel path.")
                    gradient, = torch.autograd.grad(loss, z)
                    direction_similarity = None
                    if self._study_adaptive_tolerance:
                        zv, gv = z.detach().float().flatten(), gradient.detach().float().flatten()
                        tangent = gv - (zv @ gv) / (zv @ zv).clamp_min(1e-20) * zv
                        direction = tangent / tangent.norm().clamp_min(1e-20)
                        if previous_direction is not None:
                            direction_similarity = float(direction @ previous_direction)
                        previous_direction = direction.detach()
                    objectives.append(loss.detach() if self._study_min_sync else float(loss.detach()))
                    current_objective = float(loss.detach()) if self._study_adaptive_tolerance else None
                    z = spherical_step(z.detach(), gradient.detach(), effective_step_size,
                                       check_finite=not self._study_min_sync)
                    if device.type == "cuda" and not self._study_min_sync:
                        torch.cuda.synchronize(device)
                    if self._study_min_sync and device.type == "cuda":
                        end_event.record()
                        iteration_events.append((start_event, end_event))
                    else:
                        iteration_seconds.append(time.perf_counter() - iteration_started)
                    del loss, gradient, scaled, model_input
                    progress.update(1)
                    if previous_objective is not None and self._study_adaptive_tolerance:
                        improvement = (current_objective - previous_objective) / max(abs(previous_objective), 1e-8)
                        if improvement < self._study_adaptive_tolerance and direction_similarity >= 0.95:
                            break
                    previous_objective = current_objective
                if self._study_min_sync and not torch.isfinite(z).all():
                    raise RuntimeError("Oracle Noise produced nonfinite noise.")
                output.append(z.detach().to(device=noise.device, dtype=noise.dtype))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if iteration_events:
                iteration_seconds = [start.elapsed_time(end) / 1000 for start, end in iteration_events]
            if self._study_min_sync:
                objectives = [float(value) for value in objectives]
            self.last_report = {"family": family, "seed": self.seed, "iterations": self.iterations,
                                "step_size": self.angular_step, "effective_step_size": effective_step_size,
                                "auto_step_size_scaling": self.auto_step_size_scaling,
                                "objective": self.objective, "importance_mode": self.importance_mode,
                                "attention_math": self.attention_math, "query_chunk_size": effective_chunk_size,
                                "study_cache_static": self._study_cache_static,
                                "study_checkpoint_policy": self._study_checkpoint_policy,
                                "study_memory_required_bytes": self._study_memory_required_bytes,
                                "study_query_stride": self._study_query_stride,
                                "study_proxy_factor": self._study_proxy_factor,
                                "study_force_full_load": self._study_force_full_load,
                                "study_cfg_schedule": self._study_cfg_schedule,
                                "study_min_sync": self._study_min_sync,
                                "study_adaptive_tolerance": self._study_adaptive_tolerance,
                                "study_native_quantized": self._study_native_quantized,
                                "actual_iterations": len(objectives),
                                "objectives": objectives,
                                "iteration_seconds": iteration_seconds, "seconds": time.perf_counter() - started,
                                "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                                "loaded_model_bytes": delegate.loaded_size(),
                                "total_model_bytes": delegate.model_size(),
                                "storage_dtype": str(delegate.model.get_dtype()), "compute_dtype": str(dtype),
                                "quantization_formats": sorted({m.quant_format for m in diffusion.modules()
                                                                 if getattr(m, "quant_format", None) is not None})}
            logging.info("Oracle Noise: %s, %.2fs optimization", family, self.last_report["seconds"])
            return torch.cat(output)
