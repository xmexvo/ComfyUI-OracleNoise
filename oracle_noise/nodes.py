from .provider import OracleNoiseProvider
from .preprocessing import prepare_prompt


class OracleNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",), "native_encoder": ("CLIP",), "importance_encoder": ("CLIP",),
            "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                  "tooltip": "Share the original positive prompt. ComfyUI emphasis such as (cat:1.5) multiplies Oracle importance for all supported models. Escape literal parentheses with \\( and \\)."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "sigmas": ("SIGMAS",),
            "iterations": ("INT", {"default": 5, "min": 0, "max": 100}),
            "step_size": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001,
                                      "tooltip": "Spherical update size in radians. Lower proxy resolutions need proportionally smaller steps to avoid overshooting; Auto Step Size Scaling is recommended. 0 bypasses optimization."}),
            "variant": (["positive (conditional)", "positive + negative (cfg logits)"], {"default": "positive (conditional)", "tooltip": "Positive + negative takes longer and more closely matches CFG-guided attention, but does not guarantee better image quality."}),
            "optimization_guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1,
                                                 "tooltip": "Used by positive + negative. Generally set this to the same or a lower value than the sampler CFG."}),
            "importance_encoder_type": (["auto", "clip_l", "clip_g", "clip_l_g"], {"tooltip": "CLIP components used for semantic importance. Scoring uses a fixed batch size of 64, which can use more memory."}),
            "resolution": (["256", "512", "768", "1024"], {"default": "512", "tooltip": "Fixed target image area, adjusted for aspect ratio. Lower proxy resolutions are faster but need a proportionally lower step size to avoid overshooting. Proxy step size should scale with proxy linear size; Auto Step Size Scaling is recommended. Relative resolution overrides this setting when enabled."}),
            "relative_resolution": (["off", "full", "1/2", "1/4"], {"default": "off", "tooltip": "Overrides the fixed resolution. 1/2 and 1/4 scale each image dimension, using approximately 1/4 and 1/16 of the original area. Lower resolutions are faster but need proportionally smaller steps to avoid overshooting; Auto Step Size Scaling is recommended."}),
            "resize_algorithm": (["bilinear", "bicubic"], {"default": "bicubic", "tooltip": "Interpolation used to downscale noise and upscale the optimized update. Bicubic is the default; bilinear is also available."}),
        }, "optional": {
            "oracle_prompt_data": ("ORACLE_PROMPT_DATA",),
            "conditioning_prompt": ("STRING", {"forceInput": True, "tooltip": "Connect the encoder-facing prompt from Oracle Prompt Preprocess & Encode when using its plain-text outputs."}),
            "layer_indices": ("STRING", {"default": "", "tooltip": "Comma-separated zero-based denoiser block indices. Blank selects first, middle, and last: Anima Base 0,14,27; MOSS Base 0,21,41."}),
            "layer_weights": ("STRING", {"default": "1.0,1.5,2.0", "tooltip": "Comma-separated objective weights, one per selected block, in the same order. Change this when selecting a different number of blocks."}),
            "precision": (["fp32", "bf16", "fp16"], {"default": "fp32", "tooltip": "BF16 and FP16 use lower precision math attention on CUDA. Objective and spherical updates stay FP32."}),
            "speed_preset": (["normal", "fast"], {"default": "normal", "tooltip": "Fast schedules CFG logits for the first third of iterations when positive + negative is selected."}),
            "layer_preset": (["Custom", "Anima", "MOSS"], {"default": "Custom", "tooltip": "Anima and MOSS use the benchmark layer indices and weights. Custom uses layer_indices and layer_weights."}),
            "noise_color": (["off", "pink", "brown", "custom"], {"default": "off", "tooltip": "Apply spatial fractal color after Oracle optimization. Pink uses 1/f power; brown uses 1/f². Off preserves the current output."}),
            "color_strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Blend toward colored noise: 0 is unchanged, 1 is fully colored. Mean and norm are retained."}),
            "color_slope": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.05, "tooltip": "Custom mode only: spatial power falls as 1/f^slope. 1 is pink; 2 is brown."}),
            "auto_step_size_scaling": ("BOOLEAN", {"default": True, "tooltip": "Recommended. Scale step size by the proxy's linear size relative to the original to reduce overshooting. For example, 1/2 resolution changes 0.01 to approximately 0.005. Fixed and rounded sizes use the square root of the actual area ratio."}),
        }}

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = "sampling/noise"
    DESCRIPTION = "Experimental spherical noise optimization for Anima and MOSS. Shape comes from the downstream sampler's empty latent."

    def get_noise(self, step_size, variant, optimization_guidance, importance_encoder_type, resolution, relative_resolution="off", resize_algorithm="bicubic", **kwargs):
        variants = {"positive (conditional)": "conditional", "positive + negative (cfg logits)": "cfg_logits"}
        if variant not in variants:
            raise ValueError(f"Unknown Oracle Noise variant: {variant}")
        kwargs.update({
            "angular_step": step_size,
            "objective": variants[variant],
            "optimization_cfg": optimization_guidance,
            "importance_mode": importance_encoder_type,
            "resolution": resolution,
            "relative_resolution": relative_resolution,
            "resize_algorithm": resize_algorithm,
            "query_chunk_size": 2048,
            "importance_batch_size": 64,
        })
        kwargs["attention_math"] = kwargs.pop("precision", "fp32")
        kwargs["block_preset"] = kwargs.pop("layer_preset", "Custom")
        return (OracleNoiseProvider(**kwargs),)


class OraclePromptEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clip": ("CLIP",), "text": ("STRING", {"multiline": True, "dynamicPrompts": True})},
                "optional": {"format": (["Auto", "Plain text", "MOSS layout"],)}}

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING", "ORACLE_PROMPT_DATA")
    RETURN_NAMES = ("conditioning", "oracle_prompt", "conditioning_prompt", "oracle_prompt_data")
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip, text, format="Auto"):
        data = prepare_prompt(text, format)
        if data.layout and not hasattr(clip.tokenizer, "qwen35_9b"):
            raise ValueError("MOSS layout requires a MOSS native text encoder.")
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(data.encoder_text))
        data.conditioning_tensor = conditioning[0][0]
        return conditioning, text, data.encoder_text, data


NODE_CLASS_MAPPINGS = {"OracleNoise": OracleNoise, "OraclePromptEncode": OraclePromptEncode}
NODE_DISPLAY_NAME_MAPPINGS = {"OracleNoise": "Oracle Noise (Anima / MOSS)", "OraclePromptEncode": "Oracle Prompt Preprocess & Encode"}
