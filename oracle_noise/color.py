"""Optional spatial spectral colorization of an existing noise sample."""

import math

import torch


def colorize_noise(noise, mode="off", strength=0.25, slope=1.0):
    """Tilt spatial power toward low frequencies without drawing new noise.

    Pink and brown mean power spectra proportional to 1/f and 1/f².
    The original per-image, per-channel mean and centered norm are retained.
    """
    if mode not in ("off", "pink", "brown", "custom"):
        raise ValueError(f"Unknown noise color: {mode}")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Noise color strength must be finite and between 0 and 1.")
    if not math.isfinite(slope) or not 0 <= slope <= 4:
        raise ValueError("Custom noise color slope must be finite and between 0 and 4.")
    if mode == "off" or strength == 0:
        return noise
    exponent = {"pink": 1.0, "brown": 2.0}.get(mode, slope)
    if exponent == 0:
        return noise
    if noise.ndim not in (4, 5) or (noise.ndim == 5 and noise.shape[2] != 1):
        raise ValueError("Noise colorization requires a 4D image or singleton-frame 5D latent.")

    with torch.no_grad():
        source = noise.float()
        height, width = source.shape[-2:]
        if height < 2 or width < 2:
            raise ValueError("Noise colorization requires spatial dimensions of at least 2.")
        fy = torch.fft.fftfreq(height, device=source.device) * height
        fx = torch.fft.rfftfreq(width, device=source.device) * width
        radius = torch.sqrt(fy[:, None].square() + fx[None, :].square()).clamp_min(1)
        amplitude = radius.pow(-exponent / 2)
        amplitude[0, 0] = 1  # Keep each channel's DC component unchanged.
        filtered = torch.fft.irfft2(torch.fft.rfft2(source) * amplitude, s=(height, width))
        dims = tuple(range(2, source.ndim))
        mean = source.mean(dim=dims, keepdim=True)
        centered = source - mean
        filtered_centered = filtered - filtered.mean(dim=dims, keepdim=True)
        original_norm = torch.linalg.vector_norm(centered, dim=dims, keepdim=True)
        filtered_norm = torch.linalg.vector_norm(filtered_centered, dim=dims, keepdim=True)
        filtered_centered = filtered_centered * (original_norm / filtered_norm.clamp_min(1e-20))
        mixed = centered.lerp(filtered_centered, strength)
        mixed_norm = torch.linalg.vector_norm(mixed, dim=dims, keepdim=True)
        colored = mean + mixed * (original_norm / mixed_norm.clamp_min(1e-20))
        if not torch.isfinite(colored).all():
            raise RuntimeError("Noise colorization produced nonfinite values.")
        return colored.to(dtype=noise.dtype)
