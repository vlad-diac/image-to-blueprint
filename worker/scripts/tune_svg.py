#!/usr/bin/env python3
"""Sweep png_to_svg pipeline flags/params and render a labelled comparison grid.

This is a tuning harness, not part of the production path. It reuses the exact
stage functions from ``png_to_svg.py`` so every tile is what production would
produce for that flag combination — only the orchestration is repeated here so
we can hold everything at a baseline and vary one knob per tile.

Rotation (``orient``) is intentionally EXCLUDED and forced off, so every tile
shares the source orientation and is directly comparable.

Each tile varies exactly one thing from the baseline (all defaults, orient off):
  - range params are shown at their min and max (the baseline tile covers default)
  - boolean flags are shown toggled (only the ones that change geometry)

Usage:
    python3 tune_svg.py <image> [--input-dir DIR] [--output-dir DIR]
                        [--cols N] [--tile W]

    # image resolved against --input-dir (default: repo input/)
    python3 tune_svg.py Qwen_Edit_2511_Q4_00043_.png
    # or an explicit path
    python3 tune_svg.py ../../input/Qwen_Edit_2511_Q4_00043_.png

Writes:
    <output-dir>/svg/tune_<stem>_<timestamp>/grid.png     labelled montage
    <output-dir>/svg/tune_<stem>_<timestamp>/<label>.svg  one SVG per tile
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))  # so `import png_to_svg` works from anywhere

import png_to_svg as p  # noqa: E402

_REPO_ROOT = _SCRIPT_DIR.parent.parent
_DEFAULT_INPUT_DIR = _REPO_ROOT / "input"


# ---------------------------------------------------------------------------
# Baseline: every stage at its png_to_svg default, EXCEPT orient (forced off so
# tiles stay comparable). Vary one entry per tile.
# ---------------------------------------------------------------------------
BASELINE_FLAGS: dict = dict(
    grayscale=True,
    denoise=True,
    threshold=True,
    cc_cleanup=True,
    morph=False,
    skeletonize=True,
    graph=True,
    traversal=True,
    simplify=True,
    geo_postproc=False,
    svg_opt=True,
    section_detect=True,
    orient=False,  # EXCLUDED from tuning
)

BASELINE_PARAMS: dict = dict(
    threshold=0,  # 0 = Otsu
    min_area=6,
    blur_ksize=3,
    morph_ksize=2,
    epsilon=1.0,
    geo_tolerance=0.5,
    stroke_width=1.0,
    stroke_color="#000000",
    svg_precision=2,
    section_bow_threshold=0.33,
)

# Range params → (min, max). The baseline tile already shows the default.
# Some params only bite when their stage flag is on; enable it for those tiles.
RANGE_SWEEP: list[dict] = [
    dict(param="threshold", lo=50, hi=200, note="0=Otsu is baseline"),
    dict(param="min_area", lo=1, hi=40),
    dict(param="blur_ksize", lo=1, hi=9),
    dict(param="epsilon", lo=0.5, hi=4.0),
    dict(param="morph_ksize", lo=2, hi=7, flags=dict(morph=True)),
    dict(param="geo_tolerance", lo=0.2, hi=3.0, flags=dict(geo_postproc=True)),
    dict(param="stroke_width", lo=0.5, hi=3.0),
]

# Boolean flags worth toggling (only ones that change the traced geometry).
TOGGLE_FLAGS: list[str] = [
    "denoise",
    "threshold",
    "cc_cleanup",
    "morph",
    "skeletonize",
    "simplify",
    "geo_postproc",
]


# ---------------------------------------------------------------------------
# Pipeline (mirrors png_to_svg.bitmap_to_svg_centerline, minus file writes and
# with orient skipped). Reuses png_to_svg's stage functions for all real work.
# ---------------------------------------------------------------------------
def run_variant(
    bgr: np.ndarray, flags: dict, params: dict
) -> tuple[str, list[np.ndarray], dict[int, str], int, int]:
    F = p.PipelineFlags(**flags)
    h, w = bgr.shape[:2]

    work, _ = p.stage_grayscale(bgr, F.grayscale)
    work = p.stage_denoise(work, F.denoise, params["blur_ksize"])
    work = p.stage_threshold(work, F.threshold, params["threshold"])
    work = p.stage_cc_cleanup(work, F.cc_cleanup, params["min_area"])
    work = p.stage_morph(work, F.morph, params["morph_ksize"])

    skeleton = p.stage_skeletonize(work, F.skeletonize)
    graph = p.stage_graph(skeleton, F.graph)

    if F.traversal:
        polylines = p.stage_traversal(graph, enabled=True)
    else:
        polylines = p.stage_traversal_passthrough(graph)

    polylines = p.stage_simplify(polylines, F.simplify, params["epsilon"])
    polylines = p.stage_geo_postproc(polylines, F.geo_postproc, params["geo_tolerance"])
    # orient intentionally skipped — rotation is not being tuned here.

    if F.section_detect:
        section_map = p.stage_section_detect(
            polylines, width=w, height=h, bow_threshold=params["section_bow_threshold"]
        )
    else:
        section_map = {i: "default" for i in range(len(polylines))}

    svg = p.stage_svg_strings_grouped(
        polylines,
        section_map,
        width=w,
        height=h,
        stroke_width=params["stroke_width"],
        stroke_color=params["stroke_color"],
        svg_opt=F.svg_opt,
        svg_precision=params["svg_precision"],
    )
    return svg, polylines, section_map, w, h


# ---------------------------------------------------------------------------
# Variant list
# ---------------------------------------------------------------------------
def build_variants() -> list[dict]:
    """Each variant: {label, filename, flags, params}."""
    variants: list[dict] = [
        dict(label="BASELINE (defaults)", flags={}, params={})
    ]

    for spec in RANGE_SWEEP:
        name = spec["param"]
        extra_flags = spec.get("flags", {})
        suffix = ""
        if extra_flags:
            suffix = " [" + ",".join(f"{k}=on" for k in extra_flags) + "]"
        for kind, val in (("min", spec["lo"]), ("max", spec["hi"])):
            variants.append(
                dict(
                    label=f"{name}={val} ({kind}){suffix}",
                    flags=dict(extra_flags),
                    params={name: val},
                )
            )

    for name in TOGGLE_FLAGS:
        new_val = not BASELINE_FLAGS[name]
        variants.append(
            dict(
                label=f"{name}={'ON' if new_val else 'OFF'}",
                flags={name: new_val},
                params={},
            )
        )

    # Number and slugify for filenames.
    for i, v in enumerate(variants):
        slug = (
            v["label"]
            .replace(" ", "_")
            .replace("=", "-")
            .replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
            .replace(",", "-")
            .replace("/", "-")
        )
        v["filename"] = f"{i:02d}_{slug}.svg"
    return variants


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_polylines(
    polylines: list[np.ndarray],
    section_map: dict[int, str],
    w: int,
    h: int,
    out_w: int,
    stroke_width: float,
    color_by_section: bool,
) -> Image.Image:
    scale = out_w / max(w, 1)
    out_h = max(1, round(h * scale))
    img = Image.new("RGB", (out_w, out_h), "white")
    draw = ImageDraw.Draw(img)
    line_w = max(1, round(stroke_width * scale))
    for i, xy in enumerate(polylines):
        if len(xy) < 2:
            continue
        pts = [(float(x) * scale, float(y) * scale) for x, y in xy]
        if color_by_section:
            import hashlib

            d = hashlib.md5(section_map.get(i, "default").encode()).digest()
            color = (int(d[0]), int(d[1]), int(d[2]))
        else:
            color = (0, 0, 0)
        draw.line(pts, fill=color, width=line_w)
    return img


def compose_grid(
    tiles: list[dict], source: Image.Image, title: str, cols: int, tile_w: int
) -> Image.Image:
    """tiles: list of {image: PIL.Image, label: str, stats: str}."""
    font_lbl = _load_font(15)
    font_sub = _load_font(13)
    font_title = _load_font(22)

    pad = 12
    cap_h = 46  # caption band height
    title_h = 44

    # Prepend a SOURCE reference tile.
    src_resized = source.copy()
    src_resized.thumbnail((tile_w, tile_w * 4))  # keep aspect, cap height
    cells = [dict(image=src_resized, label="SOURCE", stats="original PNG")] + tiles

    cell_h = max(c["image"].height for c in cells) + cap_h
    cell_w = tile_w
    rows = (len(cells) + cols - 1) // cols

    grid_w = cols * cell_w + (cols + 1) * pad
    grid_h = title_h + rows * cell_h + (rows + 1) * pad
    canvas = Image.new("RGB", (grid_w, grid_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    draw.text((pad, 12), title, fill=(20, 20, 20), font=font_title)

    for idx, cell in enumerate(cells):
        r, c = divmod(idx, cols)
        x0 = pad + c * (cell_w + pad)
        y0 = title_h + pad + r * (cell_h + pad)

        # White card background for the whole cell.
        draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill=(255, 255, 255))

        im = cell["image"]
        ix = x0 + (cell_w - im.width) // 2
        iy = y0 + (max(c["image"].height for c in cells) - im.height) // 2
        canvas.paste(im, (ix, iy))

        # Caption band.
        cap_y = y0 + cell_h - cap_h
        draw.rectangle(
            [x0, cap_y, x0 + cell_w, y0 + cell_h], fill=(30, 34, 40)
        )
        draw.text(
            (x0 + 6, cap_y + 4), cell["label"], fill=(255, 255, 255), font=font_lbl
        )
        draw.text(
            (x0 + 6, cap_y + 24),
            cell["stats"],
            fill=(150, 200, 255),
            font=font_sub,
        )

    return canvas


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="Image filename (resolved under --input-dir) or a path")
    ap.add_argument("--input-dir", type=Path, default=_DEFAULT_INPUT_DIR)
    ap.add_argument("--output-dir", type=Path, default=None, help="Default: --input-dir")
    ap.add_argument("--cols", type=int, default=5, help="Grid columns (default 5)")
    ap.add_argument("--tile", type=int, default=380, help="Tile width px (default 380)")
    ap.add_argument("--no-svg", action="store_true", help="Skip writing per-tile SVGs")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        img_path = args.input_dir / args.image
    if not img_path.is_file():
        print(f"Error: image not found: {args.image} (also tried {img_path})", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or args.input_dir
    stem = img_path.stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "svg" / f"tune_{stem}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    bgr = p._load_image_bgr(img_path)
    source_rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    variants = build_variants()
    print(f"Sweeping {len(variants)} variants of {img_path.name} (rotation excluded)\n")

    tiles: list[dict] = []
    for v in variants:
        flags = {**BASELINE_FLAGS, **v["flags"]}
        params = {**BASELINE_PARAMS, **v["params"]}
        svg, polylines, section_map, w, h = run_variant(bgr, flags, params)

        n_paths = sum(1 for xy in polylines if len(xy) >= 2)
        n_pts = sum(len(xy) for xy in polylines if len(xy) >= 2)
        stats = f"paths:{n_paths}  pts:{n_pts}"
        print(f"  {v['label']:<40} {stats}")

        if not args.no_svg:
            (run_dir / v["filename"]).write_text(svg, encoding="utf-8")

        tile_img = render_polylines(
            polylines,
            section_map,
            w,
            h,
            out_w=args.tile,
            stroke_width=params["stroke_width"],
            color_by_section=False,
        )
        tiles.append(dict(image=tile_img, label=v["label"], stats=stats))

    title = f"tune_svg — {img_path.name}   (rotation excluded; one knob varied per tile)"
    grid = compose_grid(tiles, source_rgb, title, cols=args.cols, tile_w=args.tile)
    grid_path = run_dir / "grid.png"
    grid.save(grid_path)

    print(f"\nGrid:      {grid_path}")
    if not args.no_svg:
        print(f"Per-tile SVGs in: {run_dir}")
    print("\nBaseline flags:", {k: v for k, v in BASELINE_FLAGS.items()})
    print("Baseline params:", BASELINE_PARAMS)


if __name__ == "__main__":
    main()
