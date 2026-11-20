# SPDX-License-Identifier: GPL-3.0-only
"""``dogpick`` command line.

    dogpick tilt TS_5.st --diameter 250 --out picks/                 # -> picks/TS_5/TS_5_picks.star
    dogpick tilt TS_5_01.mrc TS_5_02.mrc --diameter 250 --out_star picks/TS_5_picks.star

``tilt`` picks every image independently with the per-tilt DoG (dogpick.tilt) and writes 2D-pick STARs in the
format JOLT's ``import --picks_2d`` / ``detect_2d`` use.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mrcfile
import numpy as np

from . import __version__
from .io import read_stack, tilt_image_name, write_picks_star
from .tilt import TiltParams, pick_image


def _add_tilt(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "tilt", help="per-tilt DoG picker: pick each tilt/image independently, write JOLT 2D-pick STARs",
        description="Pick dark particles in every tilt of an MRC tilt stack, or in individual 2D MRC images, with a "
                    "Difference of Gaussians tuned to the particle diameter. A stack (more than one section) gives "
                    "<out>/<TS>/<TS>_picks.star with tilt_NN rows (jolt detect_2d layout); 2D images go into one "
                    "STAR keyed by file name.")
    p.add_argument("inputs", nargs="+", help="MRC tilt stack(s) (.st/.mrc) or 2D MRC image(s)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--diameter", type=float, help="particle diameter in Angstrom (needs a pixel size)")
    g.add_argument("--diameter_px", type=float, help="particle diameter in pixels")
    p.add_argument("--apix", type=float, default=None, help="Angstrom/pixel (default: MRC header)")
    p.add_argument("--top_k", type=int, default=1200,
                   help="max picks per tilt/image, best first (default 1200, as jolt detect_2d); 0 = all candidates")
    p.add_argument("--window_frac", type=float, default=0.5,
                   help="local-minimum window = 2*round(d/2*window_frac)+1 px (default 0.5; 1 = original recipe)")
    p.add_argument("--allow_positive", action="store_true",
                   help="keep minima with DoG >= 0 too (more, weaker candidates)")
    p.add_argument("--subpixel", action="store_true", help="parabolic sub-pixel refinement (default: integer px)")
    p.add_argument("--bright", action="store_true", help="particles brighter than background (default: dark)")
    o = p.add_mutually_exclusive_group(required=True)
    o.add_argument("--out", help="output directory (stacks: <out>/<TS>/<TS>_picks.star; images: <out>/picks.star)")
    o.add_argument("--out_star", help="output STAR file (one stack, or any number of 2D images)")
    p.add_argument("--ts_name", default=None, help="_TomoName (default: stack stem, or image stem for 2D images)")


def _diameter_px(args, apix_header: float, src: str) -> tuple[float, float | None]:
    if args.diameter_px is not None:
        return float(args.diameter_px), (args.apix or apix_header or None)
    apix = args.apix if args.apix is not None else apix_header
    if not apix or apix <= 0:
        raise SystemExit(f"{src}: no pixel size in the MRC header; pass --apix or use --diameter_px")
    if args.apix is None and abs(apix - 1.0) < 1e-6:
        print(f"warning: {src}: header pixel size is 1.0 A (often unset); pass --apix if that is wrong",
              file=sys.stderr)
    return args.diameter / apix, apix


def _fmt(v: float, subpixel: bool):
    return f"{v:.2f}" if subpixel else int(round(v))


def _run_tilt(args: argparse.Namespace) -> int:
    stacks, images = [], []
    for f in args.inputs:
        with mrcfile.open(f, permissive=True, header_only=True) as m:
            (stacks if int(m.header.nz) > 1 else images).append(f)
    if args.out_star and len(stacks) + (1 if images else 0) > 1:
        raise SystemExit("--out_star takes one stack, or 2D images only; use --out for several stacks")
    top_k = None if args.top_k == 0 else args.top_k
    written = []

    def params_for(src, apix_header):
        d, apix = _diameter_px(args, apix_header, src)
        return TiltParams(diameter_px=d, top_k=top_k, window_frac=args.window_frac,
                          require_negative=not args.allow_positive, subpixel=args.subpixel, bright=args.bright), apix

    def finish(out, rows, meta, t_start):
        write_picks_star(out, rows)
        meta.update(dogpick_version=__version__, method="tilt", n_picks=len(rows),
                    seconds=round(time.perf_counter() - t_start, 3))
        Path(str(out) + ".json").write_text(json.dumps(meta, indent=2))
        written.append(out)

    for f in stacks:
        t_item = time.perf_counter()
        stack, apix_header = read_stack(f)
        params, apix = params_for(f, apix_header)
        ts = args.ts_name or Path(f).stem
        out = Path(args.out_star) if args.out_star else Path(args.out) / ts / f"{ts}_picks.star"
        n = stack.shape[0]
        rows, per_tilt = [], []
        for i in range(n):
            pk = pick_image(stack[i], params)
            per_tilt.append(len(pk))
            name = tilt_image_name(i, n)
            rows += [(name, _fmt(x, params.subpixel), _fmt(y, params.subpixel), float(s), ts) for x, y, s in pk]
        finish(out, rows, {"input": str(f), "kind": "stack", "ts_name": ts, "n_tilts": n, "apix": apix,
                           "picks_per_tilt": per_tilt, "params": params.as_dict()}, t_item)
        print(f"{ts}: {len(rows)} picks on {n} tilts (mean {np.mean(per_tilt):.0f}/tilt) -> {out}")

    if images:
        t_item = time.perf_counter()
        out = Path(args.out_star) if args.out_star else Path(args.out) / "picks.star"
        rows, per_image, used = [], {}, None
        for f in images:
            data, apix_header = read_stack(f)
            params, apix = params_for(f, apix_header)
            used = params
            pk = pick_image(data[0], params)
            per_image[Path(f).name] = len(pk)
            tomo = args.ts_name or Path(f).stem
            rows += [(Path(f).name, _fmt(x, params.subpixel), _fmt(y, params.subpixel), float(s), tomo)
                     for x, y, s in pk]
        finish(out, rows, {"inputs": [str(f) for f in images], "kind": "images", "picks_per_image": per_image,
                           "params": used.as_dict()}, t_item)
        print(f"{len(images)} image(s): {len(rows)} picks -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dogpick", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"dogpick {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_tilt(sub)
    args = parser.parse_args(argv)
    if args.command == "tilt":
        return _run_tilt(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
