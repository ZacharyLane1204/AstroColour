"""
display_noise.py - make the noise in astronomical colour composites look good.

Everything here is for DISPLAY only. It never touches your science data; it
changes how pixels are mapped to screen brightness and colour.

Pipeline (each step is usable on its own):

    1. estimate_sky_and_noise()  sigma-clipped sky level + noise (scalar), or
                                 pass in the background_rms map from
                                 background_subtraction.determine_background
    2. denoise_faint()           (linear data, per channel) smooth only where
                                 there is no significant signal
    3. asinh_sigma_norm()        (per channel) stretch in units of noise sigma
                                 -> 0..1 array, drop-in for percent_norm
    4. chroma_denoise()          (final RGB) smooth colour noise, keep luminance
                                 detail and star colours

    display_composite() chains all four for a list of channels + colours.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from astropy.stats import sigma_clipped_stats


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _nan_gaussian(data, sigma):
    """Gaussian smooth that ignores NaNs (normalised convolution).
    NaNs in the input stay NaN in the output, so the footprint is preserved."""
    finite = np.isfinite(data)
    num = gaussian_filter(np.where(finite, data, 0.0), sigma, mode='nearest')
    den = gaussian_filter(finite.astype(float), sigma, mode='nearest')
    out = num / np.maximum(den, 1e-3)
    out[~finite] = np.nan
    return out


def _clipped_stats(values, sigma=3.0, maxiters=10, max_samples=2_000_000, seed=0):
    values = values[np.isfinite(values)]
    if values.size > max_samples:
        values = np.random.default_rng(seed).choice(values, max_samples, replace=False)
    _, median, std = sigma_clipped_stats(values, sigma=sigma, maxiters=maxiters)
    return float(median), float(std)


# --------------------------------------------------------------------------- #
# 1. sky + noise
# --------------------------------------------------------------------------- #
def estimate_sky_and_noise(data, mask=None, sigma=3.0):
    """
    Sigma-clipped sky level (median) and noise (std) of an image.

    mask : bool array, True = exclude (stars / nebula). STRONGLY recommended
           for nebula fields: plain clipping is biased when diffuse signal
           fills much of the frame (see notes in the chat reply).
    """
    good = np.isfinite(data)
    if mask is not None:
        good &= ~mask
    return _clipped_stats(data[good], sigma=sigma)


def _resolve(data, sky, noise, mask):
    if sky is None or noise is None:
        s, n = estimate_sky_and_noise(data, mask=mask)
        sky = s if sky is None else sky
        noise = n if noise is None else noise
    return sky, noise


# --------------------------------------------------------------------------- #
# 2. smooth only where there is no significant signal
# --------------------------------------------------------------------------- #
def denoise_faint(data, sky=None, noise=None, mask=None,
                  detect_px=1.5, smooth_px=2.5,
                  snr_lo=2.0, snr_hi=5.0, protect_px=1.5):
    """
    Blend the image with a smoothed copy, weighted by how much real signal
    each pixel has. Stars and bright structure keep full resolution; sky and
    very faint pixels get smoothed.

    Signal is detected on a lightly smoothed copy, and its noise is measured
    empirically (so correlated noise from PSF-matching convolution is handled
    correctly, unlike the white-noise formula).

    noise may be a scalar or a 2D rms map (e.g. from determine_background).
    """
    sky, noise = _resolve(data, sky, noise, mask)
    z = (data - sky) / noise

    light = _nan_gaussian(z, detect_px)
    good = np.isfinite(light)
    if mask is not None:
        good &= ~mask
    med, std = _clipped_stats(light[good])
    snr = (light - med) / std

    w = np.clip((snr - snr_lo) / (snr_hi - snr_lo), 0.0, 1.0)
    w = np.nan_to_num(w)
    # grow the protected region a little so star halos / edges stay sharp
    size = int(2 * round(2 * protect_px) + 1)
    w = gaussian_filter(maximum_filter(w, size=size), protect_px)

    heavy = _nan_gaussian(data, smooth_px)
    out = w * data + (1.0 - w) * heavy
    out[~np.isfinite(data)] = np.nan
    return out


# --------------------------------------------------------------------------- #
# 3. stretch in noise units
# --------------------------------------------------------------------------- #
def asinh_sigma_norm(data, sky=None, noise=None, mask=None,
                     black_sigma=2.0, soft_sigma=3.0,
                     white_percentile=99.9, gamma=1.0):
    """
    asinh stretch measured in units of the noise. Returns a 0..1 array, so it
    can replace RGB.percent_norm() directly.

    black_sigma      : black point = sky - black_sigma*noise. Larger -> less
                       negative noise is crushed to pure black -> smoother,
                       grainier-but-gentler sky. Smaller -> blacker sky but
                       speckly.
    soft_sigma       : where the stretch goes from linear to logarithmic (in
                       noise sigma). Smaller -> fainter stuff boosted more, but
                       noise boosted too. Larger -> more linear/quieter.
    white_percentile : white point taken from the data (protects against a
                       single hot pixel setting the scale).
    noise may be a scalar or a 2D rms map.
    """
    sky, noise = _resolve(data, sky, noise, mask)
    z = (data - sky) / noise

    zmax = np.nanpercentile(z, white_percentile)
    zmax = max(zmax, 2.0 * soft_sigma)          # guard against degenerate frames

    f = lambda v: np.arcsinh(v / soft_sigma)
    lo = f(-black_sigma)
    out = np.clip((f(z) - lo) / (f(zmax) - lo), 0.0, 1.0)   # NaN stays NaN
    return np.power(out, gamma)


# --------------------------------------------------------------------------- #
# 4. colour-noise reduction on the finished composite
# --------------------------------------------------------------------------- #
def chroma_denoise(rgb, chroma_px=4.0, star_lo=0.55, star_hi=0.85):
    """
    Smooth colour (chroma) far more than brightness (luma). The eye tolerates
    luminance grain, but random colour speckle looks ugly. Bright pixels
    (star cores) keep their original colour so stars don't get colour halos.
    """
    rgb = np.clip(np.nan_to_num(rgb), 0.0, 1.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb, cr = b - y, r - y

    cb_s = gaussian_filter(cb, chroma_px, mode='nearest')
    cr_s = gaussian_filter(cr, chroma_px, mode='nearest')

    keep = np.clip((y - star_lo) / (star_hi - star_lo), 0.0, 1.0)
    cb = keep * cb + (1.0 - keep) * cb_s
    cr = keep * cr + (1.0 - keep) * cr_s

    r2 = y + cr
    b2 = y + cb
    g2 = (y - 0.299 * r2 - 0.114 * b2) / 0.587
    return np.clip(np.stack([r2, g2, b2], axis=-1), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# all together
# --------------------------------------------------------------------------- #
def display_composite(channels, colours, noises=None, skies=None, masks=None,
                      intensities=1.0, denoise=True, chroma=True,
                      black_sigma=2.0, soft_sigma=3.0, white_percentile=99.9,
                      gamma=1.0, smooth_px=2.5, chroma_px=4.0):
    """
    channels : list of 2D arrays (background-subtracted, PSF-matched is best)
    colours  : list of (r, g, b) tuples, each 0..1
    noises   : optional list of scalars or 2D rms maps (one per channel)
    skies    : optional list of sky levels (0 if already background-subtracted)
    masks    : optional list of bool masks (True = star/nebula) for the noise fit
    """
    n = len(channels)
    noises = noises or [None] * n
    skies = skies or [None] * n
    masks = masks or [None] * n
    if np.isscalar(intensities):
        intensities = [intensities] * n

    out = 0.0
    for img, col, nz, sk, mk, inten in zip(channels, colours, noises, skies, masks, intensities):
        sk, nz = _resolve(img, sk, nz, mk)
        if denoise:
            img = denoise_faint(img, sky=sk, noise=nz, mask=mk, smooth_px=smooth_px)
        norm = asinh_sigma_norm(img, sky=sk, noise=nz, mask=mk, black_sigma=black_sigma,
                                soft_sigma=soft_sigma, white_percentile=white_percentile,
                                gamma=gamma)
        out = out + np.nan_to_num(norm)[..., None] * np.asarray(col, float).reshape(1, 1, 3) * inten
    out = np.clip(out, 0.0, 1.0)
    return chroma_denoise(out, chroma_px=chroma_px) if chroma else out