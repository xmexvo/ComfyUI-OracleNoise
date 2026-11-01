# Oracle Noise for ComfyUI

Experimental, prompt-aware initial-noise optimization for **Anima** and **MOSS-Image**. The node returns a deferred `NOISE` object; `SamplerCustomAdvanced` supplies the actual latent shape when it calls `generate_noise(latent)`.

Supports independent **CLIP-L**, **CLIP-G**, and combined **L/G** importance scoring. No training, CLIP vision model, image-reward model, or VAE is used for optimization.

## Install

Extract the ZIP into a single folder under `ComfyUI/custom_nodes/`, with `__init__.py` directly inside that folder, then restart ComfyUI. Existing ComfyUI dependencies supply the runtime; `requirements.txt` lists the additional tokenizer dependencies if they are absent.

## Workflow

1. Load the diffusion model and its native text encoder normally. Apply any LoRAs before connecting the model.
2. Encode positive and negative conditioning with the native encoder.
3. Load the independent scorer with `CLIPLoader` for CLIP-L or CLIP-G, or `DualCLIPLoader` with type `sdxl` for L/G. An SDXL checkpoint loader's CLIP output also works.
4. Connect model, native encoder, importance encoder, conditioning, prompt, and schedule to **Oracle Noise (Anima / MOSS)**.
5. Connect its `NOISE` output to `SamplerCustomAdvanced`. Share the same model, positive/negative conditioning, and SIGMAS with the sampler/guider.
6. Supply an empty image latent to the sampler: `EmptySD3LatentImage` for Anima, `EmptyFlux2LatentImage` for MOSS.

Share a single `PrimitiveStringMultiline` output between positive encoding and the Oracle prompt socket, or use **Oracle Prompt Preprocess & Encode** below. The node checks Anima T5 IDs and native sequence lengths, but cannot recover or verify arbitrary precomputed conditioning text.

[workflows/](workflows/) contains four UI workflows (`*.json`) and matching API graphs (`*.api.json`): Anima, Anima Turbo LoRA, MOSS Base, and MOSS SFT + DMD LoRA. Select your own model filenames after importing. Workflows default to separate `clip_l.safetensors` and `clip_g.safetensors`; replace the importance loader with an existing SDXL CLIP output if preferred. Distilled workflows are templates awaiting checkpoint validation.

## Controls

| Setting | Behavior |
|---|---|
| `seed` | ComfyUI fixed/random/increment/decrement control |
| `iterations` | Optimization iterations, separate from sampling steps; default 5; 0 returns exact standard noise when `noise_color=off` |
| `step_size` | Spherical update size in radians; default 0.01; 0 bypasses optimization. Lower proxy resolutions need proportionally lower steps to avoid overshooting. |
| `auto_step_size_scaling` | On by default. Multiplies step size by the square root of the actual proxy-to-original area ratio; for example, a half-width, half-height proxy uses half the step size. |
| `variant` | `positive (conditional)` by default, or `positive + negative (cfg logits)` to optimize using both branches. The latter takes longer and more closely matches CFG-guided attention, but does not guarantee better image quality. |
| `optimization_guidance` | Guidance scale used by `positive + negative (cfg logits)`; default 3.5. Generally use the same or a lower value than the sampler CFG. |
| `importance_encoder_type` | `auto`, `clip_l`, `clip_g`, or `clip_l_g`; explicit modes require the requested components |
| `resolution` | Fixed target image area: `256`, `512` (default), `768`, or `1024` pixels square, adjusted for aspect ratio without enlarging smaller images. |
| `relative_resolution` | `off` (default), `full`, `1/2`, or `1/4`. When enabled, overrides the fixed target; fractions scale each spatial dimension. |
| `resize_algorithm` | `bicubic` (default) or `bilinear`, used for downsampling noise and upsampling the optimization update. Downsampling uses antialiasing and normalizes the proxy noise. |
| `block_preset` | `Anima` selects `0,14,27`; `MOSS` selects `0,21,41`; both use weights `1.0,1.5,2.0`. `Custom` (default) uses the two inputs below. Presets require the matching model family and block count. |
| `layer_indices` | Comma-separated zero-based denoiser block indices. Blank selects first, middle, last: Anima Base `0,14,27`; MOSS Base `0,21,41`. |
| `layer_weights` | One comma-separated weight per selected block, in the same order; default `1.0,1.5,2.0`. For example, select only block `14` with weight `1.0`. |
| Query chunk size | Fixed at 512. This was faster and used less memory than 128 in measured Anima and MOSS runs, with small numerical changes. |
| Importance scoring batch size | Fixed at 64. Larger batches can use more scoring memory. |
| `attention_math` | `fp32` default; `bf16` and `fp16` are alternatives. BF16/FP16 cast attention inputs and restore the model dtype afterward. The objective and noise updates remain FP32. |
| `noise_color` | `off` default; `pink` (1/f power), `brown` (1/f² power), or `custom`. Applied to the completed Oracle noise, or standard noise when optimization is bypassed. |
| `color_strength` | Default `0.25`; 0 leaves noise unchanged, 1 uses the full spectral tilt. |
| `color_slope` | Custom mode only; default 0; spatial power-spectrum exponent from 0 to 4. `1` matches pink and `2` matches brown. Noise color remains off by default. |

Use small angular steps first. More iterations or a larger angle can reduce quality. Equal importance scores produce uniform valid-token weights. Empty prompts without valid scorable tokens raise an error unless optimization is bypassed.

## Method and deliberate adaptations

Based on [Oracle Noise, arXiv:2604.23540](https://arxiv.org/abs/2604.23540). Each scorer measures sentence-embedding changes after padding individual tokens without moving their positions. CLIP-L uses its unprojected final normalized EOS representation; CLIP-G uses its projected EOS representation. The original EOS position is retained for pooling even when CLIP-L's padding token is also EOS.

Long prompts use overlapping context windows. Per-encoder token scores are averaged over windows, mapped through character spans, normalized separately, then averaged equally in combined mode. Automatic weights lie in `[0.5, 3.0]` before manual emphasis is applied. MOSS uses its native plain-text chat template and excludes template tokens from scoring.

### Manual importance weighting

**Oracle Prompt Preprocess & Encode** accepts a native text encoder and a weighted prompt. Connect its `conditioning` to Oracle's positive input and the sampler, and its `oracle_prompt_data` to Oracle's matching optional input. Structured data takes precedence over the prompt string inputs and is bound to the exact positive conditioning tensor returned by this node; use the same conditioning output for both connections. It strips emphasis from the encoded text while preserving escaped literal parentheses. Oracle applies the original weights to noise optimization. The original three outputs retain their order, and the internal node identifier is unchanged. Existing plain-text workflows using `oracle_prompt` and `conditioning_prompt` continue to work.

The optional `format` selector defaults to `Auto`. Plain text uses ComfyUI emphasis as described below. JSON-shaped input must match the supported MOSS layout schema: a `summary` object with a nonempty `caption`, and a nonempty `items` array whose entries have a `label` (`object`, `character`, or `text`), `caption`, and four finite `bbox` numbers. Text items also require a literal `text` string. `MOSS layout` explicitly selects that schema; `Plain text` lets you treat JSON-shaped content as ordinary text. Malformed layouts, duplicate JSON keys, and unsupported schemas raise errors.

For MOSS layouts, only `summary.caption` and item captions are parsed for emphasis and scored. Captions are cleaned, while coordinates, other fields, and exact rendered `text` values are preserved as data. JSON formatting is normalized. Use normal JSON escaping: a caption containing the ComfyUI prompt `(\(cat\):1.5)` is written as `"(\\(cat\\):1.5)"` in JSON. Literal lettering such as `"(cat:9)"` in a `text` field is never interpreted as emphasis.

The structured output tracks caption characters through JSON escapes and repeated fields to the full native conditioning. Oracle and the sampler both receive the complete layout. Keys, coordinates, literal lettering, and formatting have zero direct objective importance but remain active conditioning. Optimization is conditioned on layout context; it does not explicitly restrict each object's rewarded attention to its bounding box. Layout mode requires the MOSS native encoder and MOSS model. Use `oracle_prompt_data` for layout workflows; the legacy string outputs alone do not carry the layout mapping.

The Oracle `prompt` supports ComfyUI emphasis for every supported model: `(red cat:1.5)` multiplies that text's automatic importance by 1.5, `(red cat:0.5)` halves it, and `(red cat)` uses ComfyUI's default 1.1 emphasis. Nested groups follow the installed ComfyUI parser. Use `\(` and `\)` for literal parentheses, for example `\(cat:1.5\)` remains literal text with importance multiplier 1. Square brackets remain literal under ComfyUI's parser.

Syntax is removed before semantic scoring. Manual multipliers are applied after automatic normalization, then projected onto native token positions through character offsets; tokens spanning multiple characters use their mean weighted score. Repeated words retain their own multipliers. Formatting, chat-template, and padding positions have zero importance. Zero weights remove the corresponding text's direct objective contribution; if all scorable text has zero weight, Oracle returns standard noise (with optional noise color). Finite negative weights are accepted, following ComfyUI, and discourage attention in the Oracle objective. Nonfinite weights are rejected.

Manual importance changes Oracle's noise optimization. Native conditioning remains as supplied: Anima also preserves its native conditioning weights, while MOSS's native encoder treats emphasis syntax as text. Continue sharing the same original prompt with positive encoding and Oracle so native token alignment stays consistent. Manual importance does not add weighting support to the downstream MOSS sampler or guarantee a proportional visual change.

Anima scores its denoiser cross-attention against **T5 adapter positions**, not the Qwen source sequence. Its native zero padding remains part of attention, with zero importance. The image latent has a singleton temporal axis after ComfyUI's latent normalization; this is supported and is not treated as video.

MOSS scores image-query/text-key attention using the **full image-and-text denominator**. Both models use positive attention by default. CFG-logit mode batches positive and negative contexts at a common length for optimization only, preserves branch padding masks in forward attention, substitutes neutral zero logits for missing negative positions, and evaluates extrapolated logits over the positive key domain. This positional comparison is a heuristic, especially for unrelated negative prompts.

Spherical updates preserve each image's initial noise norm. They do **not** preserve the full Gaussian probability distribution, guarantee prompt improvement, or reproduce the paper's timing/quality claims on these models.

Optional colorization is inspired by [RES4LYF Advanced Noise](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/nodes_latents.py) and its [fractal frequency filter](https://github.com/ClownsharkBatwing/RES4LYF/blob/main/beta/noise_classes.py). It filters the already optimized sample in FP32, blends it with the original by strength, then restores each image/channel mean and centered norm. It needs no additional seed or random draw. It deliberately changes the noise spectrum and Gaussian sampling distribution. Pink with strength `0.25` is a mild starting point; use an SDE sampler if you want later Gaussian noise injection. FFT work scales with latent size and adds some CPU time after optimization.

## Gradients and memory

Optimization temporarily disables inference mode and uses a non-dynamic ModelPatcher delegate with normal frozen tensors, differentiable model operations, block checkpointing, and query-chunked PyTorch math attention. Math attention avoids the nondeterministic Flash backward encountered during testing. Ordinary sampling retains its configured backend afterward.

The default `attention_math=fp32` mode uses full-precision attention math. BF16 attention requires a CUDA GPU with BF16 support and casts attention inputs to BF16 even when Anima inference runs in FP16, so `--bf16-unet` is unnecessary. MOSS retains its native BF16 inference dtype. The attention objective, tangent projection, and spherical updates remain FP32. Temporary backend settings are restored after optimization. A separately loaded `LongerCLIP L SDXL.safetensors` can be selected as CLIP-L importance; the tested checkpoint has 77 positional embeddings, so longer text uses overlapping 75-content-token windows rather than a 248-position context.

Native FP8 and INT8 weights remain compressed in storage. Unsupported inference-only linear paths use ComfyUI's dequantized cast path during optimization, preserving weight patches. This is a **surrogate for quantized inference**, with extra compute/memory costs. CPU copies may be needed for inference-created parameters; no persistent full BF16 GPU duplicate is required. Text encoders are unloaded before optimization, and temporary models/flags are restored on completion or interruption.

Supported diffusion quantization layouts: native FP8 E4M3/E5M2 and tensorwise INT8, including the MOSS INT8 ConvRot format. Other layouts fail explicitly. Quantized native text encoders can be used because their outputs are detached; no text-encoder gradients are needed.

Only ordinary single-prompt conditioning and empty image latents are supported, including MOSS layout JSON through the preprocessing node's structured output. Regional/scheduled/concatenated conditioning, masks, nonzero starting latents, video, unsupported layout schemas, reference editing, GGUF, and additional transformer behavior patches are outside v1. Standard weight LoRAs and model-sampling patches are supported.
