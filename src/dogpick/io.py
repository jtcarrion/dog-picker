# SPDX-License-Identifier: GPL-3.0-only
"""Tilt-stack reading and JOLT-compatible 2D pick STAR writing."""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path

import mrcfile
import numpy as np


def read_stack(path: str | PathLike) -> tuple[np.ndarray, float]:
    """Return (stack[n_tilts, ny, nx] as float32, pixel size in Angstrom from the header)."""
    with mrcfile.open(path, permissive=True, mode="r") as mrc:
        data = np.asarray(mrc.data, dtype=np.float32)
        apix = float(mrc.voxel_size.x)
    if data.ndim == 2:
        data = data[None]
    return data, apix


def tilt_image_name(index0: int, n_tilts: int) -> str:
    """Same naming as ``jolt.topaz.tilt_image_name``: 1-based ``tilt_01`` ..."""
    return f"tilt_{index0 + 1:0{max(2, len(str(n_tilts)))}d}"


def write_picks_star(path: str | PathLike, rows: list[tuple], extra_columns: tuple[str, ...] = ()) -> None:
    """Write ``(image, x, y, score, tomo, *extra)`` rows as a JOLT 2D-pick STAR loop (atomically).

    Columns match ``jolt.topaz.write_picks_star``; x = column, y = row in the raw-stack frame
    (0-based pixel indices, integer = pixel center), per docs/conventions in JOLT.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    columns = ("_rlnMicrographName", "_rlnCoordinateX", "_rlnCoordinateY", "_rlnAutopickFigureOfMerit",
               "_TomoName") + tuple(extra_columns)
    with open(tmp, "w") as handle:
        handle.write("\ndata_\n\nloop_\n")
        for i, column in enumerate(columns, 1):
            handle.write(f"{column} #{i}\n")
        for image, x, y, score, tomo, *extra in rows:
            tail = "".join(f"\t{v}" for v in extra)
            handle.write(f"{image}\t{x}\t{y}\t{score:.6f}\t{tomo}{tail}\n")
    os.replace(tmp, path)
