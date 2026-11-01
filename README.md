# Oracle Noise for ComfyUI

Oracle Noise shapes the starting noise for an image using your prompt. This experimental implementation supports **Anima** and **MOSS-Image** and is based on [Oracle Noise: Faster Semantic Spherical Alignment for Interpretable Latent Optimization](https://arxiv.org/abs/2604.23540). Results can vary; more optimization does not always mean a better image.

## Install

Place this repository in `ComfyUI/custom_nodes/ComfyUI-OracleNoise` and restart ComfyUI. If the tokenizer dependencies are missing, install the packages in `requirements.txt` using the Python environment that runs ComfyUI.

## Optional Long-Context CLIP Encoders

These CLIP-L encoders are used as separate importance scorers in the example and comparison workflows:

- [Anzhc SDXL Text Encoder Longer CLIP-L](https://huggingface.co/Anzhc/SDXL-Text-Encoder-Longer-CLIP-L)
- [zer0int LongCLIP KO-LITE TypoAttack ViT-L/14](https://huggingface.co/zer0int/LongCLIP-KO-LITE-TypoAttack-Attn-ViT-L-14)

Load either with `CLIPLoader` as the importance encoder; keep the model's native encoder for prompt conditioning.

## Getting started

1. Load your model, native text encoder, and positive and negative prompts. Apply any LoRAs to the model first.
2. Load a separate CLIP-L, CLIP-G, or SDXL CLIP-L/G encoder for prompt importance scoring.
3. Connect these, the positive prompt text, and your sampling schedule (`SIGMAS`) to **Oracle Noise (Anima / MOSS)**. Use the same model, conditioning, and schedule for sampling.
4. Connect its `NOISE` output to `SamplerCustomAdvanced` and supply an empty latent: `EmptySD3LatentImage` for Anima or `EmptyFlux2LatentImage` for MOSS.

The [workflow directory](workflows/) includes four UI workflows:

- `anima-example.json` and `moss-example.json` are single-image examples, set to 2 Oracle iterations and a 0.025 step size. The MOSS example uses **Oracle Prompt Preprocess & Encode**.
- `anima-compare.json` and `moss-compare.json` compare the baseline, 38-step compute-matched baseline, and Oracle variants at step sizes 0.01, 0.005, and 0.05. The 0.05 variants use one iteration; the others use five. MOSS uses its native prompt preprocessing node.

Select model files after importing. The distilled API graphs remain available as templates awaiting checkpoint validation.

## Oracle Noise inputs

The node produces starting noise for the sampler. Its main controls are:

| Input | What it does |
|---|---|
| `model`, `native_encoder` | The diffusion model and its usual text encoder. |
| `importance_encoder` | A separate CLIP encoder used to estimate which prompt words matter. |
| `positive`, `negative`, `prompt` | Your conditioning and the original positive prompt text. Keep the prompt consistent with the positive conditioning. |
| `sigmas` | The sampling schedule shared with the sampler. |
| `seed` | Controls the starting random noise. Reuse it to compare settings. |
| `iterations` | Number of optimization passes; 0 skips optimization. This is separate from sampling steps. |
| `step_size` | How much each pass changes the noise. Smaller values make gentler changes; 0 skips optimization. |
| `auto_step_size_scaling` | Adjusts step size for the working resolution. Leave it on as a starting point. |
| `variant` | Uses the positive prompt alone, or positive and negative conditioning together. The latter takes longer. |
| `optimization_guidance` | Guidance for the positive-and-negative variant. Start with the sampler's CFG value or lower. |
| `importance_encoder_type` | Selects CLIP-L, CLIP-G, both, or automatic detection. |
| `resolution`, `relative_resolution` | Set the working size for optimization. Lower sizes are generally faster; relative resolution overrides the fixed setting. |
| `resize_algorithm` | Chooses how the working image is resized. Bicubic is the default. |
| `layer_preset`, `layer_indices`, `layer_weights` | Choose which model layers guide optimization and their relative influence. Use an Anima or MOSS preset for a starting point, or Custom for manual choices. |
| `precision` | Controls calculation precision. FP32 is the default; BF16 and FP16 are lower-precision options. |
| `speed_preset` | With the positive-and-negative variant, Fast uses that guidance for fewer iterations. |
| `noise_color`, `color_strength`, `color_slope` | Optionally change the noise pattern. Color is off by default; slope applies to Custom color. |
| `oracle_prompt_data`, `conditioning_prompt` | Optional connections from the prompt preprocessing node. |

## Oracle Prompt Preprocess & Encode

This optional node prepares a prompt and encodes it for the model. It can handle plain text and supported MOSS layouts, including prompt emphasis.

- **`clip`**: Connect the model's native text encoder.
- **`text`**: Enter the positive prompt or supported MOSS layout.
- **`format`**: Choose Auto, Plain text, or MOSS layout.

Connect its `conditioning` output to both Oracle's positive input and the sampler's positive conditioning. Connect `oracle_prompt_data` to Oracle's matching input, especially for MOSS layouts. The node also provides prompt text outputs for plain-text workflows.

## Notes

Use ordinary single-prompt conditioning and an empty image latent. Video, masks, regional or scheduled conditioning, reference editing, and GGUF are not supported. Optimization adds time and memory use, and its results are not guaranteed to improve every image.
