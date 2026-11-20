# SPDX-License-Identifier: GPL-3.0-only
"""Per-image (per-tilt) Difference-of-Gaussians picker.

Each 2D image is picked independently:

1. z-score the image;
2. DoG = G(s1) * I - G(s2) * I with s1 = d / (2 sqrt 2), s2 = 1.6 s1 (d = particle diameter in pixels);
3. keep local minima of the DoG (dark particles) in a (2 round(d/2 * window_frac) + 1) window, with DoG < 0,
   at least round(d/2) px from the border;
4. rank by score = -DoG / (1.4826 MAD of the DoG map), i.e. a robust per-image z-score, so scores are
   comparable across tilts (JOLT ranks picks by this figure of merit); keep the top_k.

With window_frac = 1 and integer coordinates this is exactly the research DoG recipe benchmarked against Topaz in
JOLT. The default window_frac = 0.5 gave ~10 % more recall at the same picks per tilt, with unchanged precision
and no split duplicates, on in-cell ribosomes.

Coordinates are 0-based pixel indices of the input array: x = column, y = row (an integer is a pixel centre),
which is the frame JOLT expects for raw-stack picks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter

K_FACTOR = 1.6  # sigma2 / sigma1


@dataclass(frozen=True)
class TiltParams:
    diameter_px: float
    top_k: int | None = 1200
    window_frac: float = 0.5
    require_negative: bool = True
    subpixel: bool = False
    bright: bool = False

    def __post_init__(self):
        if not self.diameter_px > 0:
            raise ValueError("diameter_px must be > 0")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1 (or None for all candidates)")
        if not 0 < self.window_frac <= 2:
            raise ValueError("window_frac must be in (0, 2]")

    @property
    def sigma1(self) -> float:
        return self.diameter_px / (2.0 * np.sqrt(2.0))

    @property
    def sigma2(self) -> float:
        return K_FACTOR * self.sigma1

    @property
    def nms_half(self) -> int:
        return max(1, int(round(self.diameter_px / 2.0 * self.window_frac)))

    @property
    def border(self) -> int:
        return max(1, int(round(self.diameter_px / 2.0)))

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(sigma1_px=self.sigma1, sigma2_px=self.sigma2, nms_window_px=2 * self.nms_half + 1,
                 border_px=self.border, k_factor=K_FACTOR)
        return d


def dog_map(image: np.ndarray, params: TiltParams) -> np.ndarray:
    """DoG of the z-scored image (dark particles -> negative values; bright particles if params.bright)."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {img.shape}")
    if not np.all(np.isfinite(img)):
        raise ValueError("image contains NaN or inf")
    std = img.std()
    img = (img - img.mean()) / (std if std > 0 else 1.0)
    if params.bright:
        img = -img
    return gaussian_filter(img, params.sigma1) - gaussian_filter(img, params.sigma2)


def _subpixel_offsets(m: np.ndarray, iy: np.ndarray, ix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertex of the 3-point parabola through each minimum, per axis, clipped to +-0.5 px."""
    def vertex(a, b, c):
        den = a - 2 * b + c
        with np.errstate(divide="ignore", invalid="ignore"):
            off = np.where(den != 0, 0.5 * (a - c) / den, 0.0)
        return np.clip(off, -0.5, 0.5)

    dx = vertex(m[iy, ix - 1], m[iy, ix], m[iy, ix + 1])
    dy = vertex(m[iy - 1, ix], m[iy, ix], m[iy + 1, ix])
    return dx, dy


def pick_image(image: np.ndarray, params: TiltParams) -> np.ndarray:
    """Pick one 2D image. Returns array[n, 3] of (x, y, score), sorted by descending score."""
    m = dog_map(image, params)
    h, b = params.nms_half, params.border
    mask = m == minimum_filter(m, size=2 * h + 1)  # default 'reflect' edges, as in the research recipe
    if params.require_negative:
        mask &= m < 0
    if m.shape[0] <= 2 * b or m.shape[1] <= 2 * b:
        return np.empty((0, 3))
    mask[:b, :] = mask[-b:, :] = mask[:, :b] = mask[:, -b:] = False
    iy, ix = np.nonzero(mask)
    mad = np.median(np.abs(m - np.median(m)))
    scale = 1.4826 * mad if mad > 0 else (m.std() or 1.0)
    score = -m[iy, ix] / scale
    order = np.argsort(-score, kind="stable")
    if params.top_k is not None:
        order = order[: params.top_k]
    iy, ix, score = iy[order], ix[order], score[order]
    x, y = ix.astype(float), iy.astype(float)
    if params.subpixel and len(ix):
        dx, dy = _subpixel_offsets(m, iy, ix)
        x, y = x + dx, y + dy
    return np.column_stack([x, y, score])
