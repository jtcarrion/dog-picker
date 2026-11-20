# dogpick

Lightweight Difference-of-Gaussians (DoG) particle picking for cryo-ET tilt series. It writes per-tilt 2D picks
in the STAR format that [JOLT](https://github.com/jtcarrion/jolt) reads (`jolt import --picks_2d`).

Status: **0.0.1.dev0**. CPU only; depends on numpy, scipy and mrcfile.

## `dogpick tilt`: per-tilt DoG

Every tilt (or every 2D image) is picked **independently**:

1. Z-score the image.
2. Compute the DoG $G_{\sigma_1}*I - G_{\sigma_2}*I$ with $\sigma_1 = d/(2\sqrt2)$ and $\sigma_2 = 1.6\,\sigma_1$,
   where $d$ is the particle diameter in pixels.
3. Keep dark local minima (DoG < 0) in a $2\,\mathrm{round}(d/2 \cdot w)+1$ px window ($w$ = `--window_frac`,
   default 0.5), at least $d/2$ from the border.
4. Rank by a robust per-tilt score, $-\mathrm{DoG} / (1.4826\,\mathrm{MAD})$, so scores are comparable across
   tilts. Keep the top `--top_k` per tilt (default 1200, as in `jolt detect_2d`).

```bash
# tilt stack -> picks/TS_5/TS_5_picks.star (one row per pick, tilt_NN = 1-based stack position)
python -m dogpick tilt TS_5.st --diameter 250 --out picks/

# several stacks at once
python -m dogpick tilt tiltstack/TS_*/TS_*.st --diameter 250 --out picks/

# individual 2D images -> one STAR keyed by file name
python -m dogpick tilt TS_5_01.mrc TS_5_02.mrc --diameter_px 25 --out_star picks/TS_5_picks.star --ts_name TS_5

# then, in JOLT
jolt import ... --picks_2d "picks/{ts}/{ts}_picks.star"
```

| Option | Default | Meaning |
|---|---|---|
| `--diameter` / `--diameter_px` | required | particle diameter in Å (pixel size from the MRC header or `--apix`) or in px |
| `--top_k` | 1200 | max picks per tilt/image, best first; `0` = all candidates |
| `--window_frac` | 0.5 | local-minimum window as a fraction of $d$; `1` = the original research recipe exactly |
| `--allow_positive` | off | also keep minima with DoG ≥ 0 (more, weaker candidates) |
| `--subpixel` | off | parabolic sub-pixel refinement (otherwise integer pixel centres) |
| `--bright` | off | particles brighter than background |

**Output** (identical columns to `jolt detect_2d`):
- `_rlnMicrographName` (`tilt_NN` for stacks, file name for images);
- `_rlnCoordinateX/Y`: 0-based pixels of the input image, x = column, y = row, the raw-stack frame;
- `_rlnAutopickFigureOfMerit`: the score;
- `_TomoName`;
- plus `<star>.json` with all parameters, picks per tilt and run time.

**Expectations:**
- About 1 s per 31-tilt 630² series on one CPU core.
- A particle-scale DoG has a natural candidate ceiling: for ~25 px ribosomes it is ~250–350 picks/tilt, so
  fewer than `--top_k` may be returned.
- `--window_frac 1` reproduces the research DoG used in the JOLT benchmarks pick-for-pick.
- End to end through JOLT on CZI DS-10441 (27 TS): F-β4 0.683 with the defaults (0.658 with
  `--window_frac 1`).

## Test

```bash
PYTHONPATH=src python -m pytest -q
```

## License

GPL-3.0-only.
