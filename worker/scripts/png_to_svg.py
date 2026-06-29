#!/usr/bin/env python3
"""
Custom centerline tracing: raster → skeleton graph → stroke SVG (no Potrace).

Pipeline stages (fixed order; each can be toggled except SVG generation):
grayscale → denoise → threshold → CC cleanup → morph (optional) → skeletonize
→ graph → edge traversal → Douglas–Peucker → geo post-proc (optional)
→ section detect (optional) → SVG paths (grouped by section) → SVG optimisation.

Install: pip install -r requirements-centerline.txt

Each run writes previews under output/svg/<name>_<timestamp>/.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import sknw  # type: ignore[import-untyped]
from shapely.geometry import LineString  # type: ignore[import-untyped]
from skimage.morphology import skeletonize

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_OUTPUT_DIR = _SCRIPT_DIR / "output"


@dataclass
class PipelineFlags:
    grayscale: bool
    denoise: bool
    threshold: bool
    cc_cleanup: bool
    morph: bool
    skeletonize: bool
    graph: bool
    traversal: bool
    simplify: bool
    geo_postproc: bool
    svg_opt: bool
    section_detect: bool


def _load_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def stage_grayscale(
    bgr: np.ndarray, enabled: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return (work_gray, preview_same_as_output)."""
    if enabled:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr[:, :, 0].copy()
    return gray, gray


def stage_denoise(gray: np.ndarray, enabled: bool, blur_ksize: int) -> np.ndarray:
    if not enabled:
        return gray
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    k = max(1, k)
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    # Light bilateral to reduce speckle without destroying edges
    return cv2.bilateralFilter(blurred, d=3, sigmaColor=25, sigmaSpace=3)


def _lines_white_from_threshold(work: np.ndarray, threshold: int) -> np.ndarray:
    """Binary image: line pixels = 255, background = 0 (same convention as Potrace inv)."""
    if threshold == 0:
        _, mask = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(mask) > 127:
            mask = 255 - mask
        lines_white = mask
    else:
        _, mask = cv2.threshold(work, threshold, 255, cv2.THRESH_BINARY)
        lines_white = 255 - mask
    return lines_white


def stage_threshold(work: np.ndarray, enabled: bool, threshold: int) -> np.ndarray:
    if not enabled:
        return work
    return _lines_white_from_threshold(work, threshold)


def _is_binary_lines(img: np.ndarray) -> bool:
    if img.size == 0:
        return True
    u = np.unique(img)
    if len(u) > 3:
        return False
    return set(int(x) for x in u.tolist()) <= {0, 255}


def _to_lines_white_binary(img: np.ndarray) -> np.ndarray:
    if _is_binary_lines(img):
        b = img.astype(np.uint8)
        if np.median(b) > 127:
            return 255 - b
        return b
    _, otsu = cv2.threshold(img.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(otsu) > 127:
        otsu = 255 - otsu
    return otsu


def stage_cc_cleanup(lines_white: np.ndarray, enabled: bool, min_area: int) -> np.ndarray:
    if not enabled:
        return lines_white
    bin_img = _to_lines_white_binary(lines_white)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    clean = np.zeros_like(bin_img)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def stage_morph(lines_white: np.ndarray, enabled: bool, ksize: int) -> np.ndarray:
    if not enabled:
        return lines_white
    bin_img = _to_lines_white_binary(lines_white)
    k = max(1, int(ksize))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    return closed


def stage_skeletonize(lines_white: np.ndarray, enabled: bool) -> np.ndarray:
    bin_img = _to_lines_white_binary(lines_white)
    fg = bin_img > 0
    if not enabled:
        return (fg.astype(np.uint8) * 255)
    skel = skeletonize(fg)
    return (skel.astype(np.uint8) * 255)


def stage_graph(skeleton: np.ndarray, enabled: bool) -> nx.Graph | None:
    if not enabled:
        return None
    ske = (skeleton > 0).astype(np.uint16)
    if not np.any(ske):
        return nx.Graph()
    return sknw.build_sknw(ske, multi=False, iso=True, ring=True, full=True)


def stage_traversal(
    graph: nx.Graph | None, enabled: bool
) -> list[np.ndarray]:
    """Return list of Nx2 float64 arrays of (x, y) pixel coordinates."""
    if graph is None or not enabled:
        return []
    polylines: list[np.ndarray] = []
    for _u, _v, edata in graph.edges(data=True):
        pts = edata.get("pts")
        if pts is None or len(pts) < 2:
            continue
        yx = np.asarray(pts, dtype=np.float64)
        xy = np.column_stack((yx[:, 1], yx[:, 0]))
        # Remove consecutive duplicates (traversal normalisation)
        out = [xy[0]]
        for i in range(1, len(xy)):
            if np.max(np.abs(xy[i] - out[-1])) > 1e-6:
                out.append(xy[i])
        if len(out) >= 2:
            polylines.append(np.asarray(out, dtype=np.float64))
    return polylines


def stage_traversal_passthrough(graph: nx.Graph | None) -> list[np.ndarray]:
    """When traversal is disabled: raw edge points from sknw without deduplication."""
    if graph is None:
        return []
    polylines: list[np.ndarray] = []
    for _u, _v, edata in graph.edges(data=True):
        pts = edata.get("pts")
        if pts is None or len(pts) < 2:
            continue
        yx = np.asarray(pts, dtype=np.float64)
        xy = np.column_stack((yx[:, 1], yx[:, 0]))
        polylines.append(xy)
    return polylines


def stage_simplify(polylines: list[np.ndarray], enabled: bool, epsilon: float) -> list[np.ndarray]:
    if not enabled:
        return polylines
    out: list[np.ndarray] = []
    eps = float(epsilon)
    for xy in polylines:
        if len(xy) < 3:
            out.append(xy)
            continue
        contour = xy.reshape(-1, 1, 2).astype(np.float32)
        approx = cv2.approxPolyDP(contour, eps, closed=False)
        pts = approx.reshape(-1, 2).astype(np.float64)
        if len(pts) >= 2:
            out.append(pts)
    return out


def stage_geo_postproc(
    polylines: list[np.ndarray], enabled: bool, tolerance: float
) -> list[np.ndarray]:
    if not enabled:
        return polylines
    tol = float(tolerance)
    out: list[np.ndarray] = []
    for xy in polylines:
        if len(xy) < 2:
            continue
        line = LineString(xy)
        simp = line.simplify(tolerance=tol, preserve_topology=False)
        c = np.asarray(simp.coords, dtype=np.float64)
        if len(c) >= 2:
            out.append(c)
    return out


def stage_section_detect(
    polylines: list[np.ndarray],
    width: int,
    height: int,
    bow_threshold: float,
) -> dict[int, str]:
    """Map each polyline to a section using a 3×2 grid on the hull bounding box.

    Bow is left (low X), stern/crows-nest is right (high X). Port is above the
    horizontal midline (lower image Y), starboard below.
    """
    del width, height  # reserved for future use
    n = len(polylines)
    nonempty = [p for p in polylines if len(p) > 0]
    if not nonempty:
        return {} if n == 0 else {i: "mid-port" for i in range(n)}

    allp = np.vstack(nonempty)
    x_min = float(allp[:, 0].min())
    x_max = float(allp[:, 0].max())
    y_min = float(allp[:, 1].min())
    y_max = float(allp[:, 1].max())
    centerline_y = (y_min + y_max) / 2.0
    xr = max(x_max - x_min, 1e-6)

    section_map: dict[int, str] = {}
    for i, xy in enumerate(polylines):
        if len(xy) == 0:
            section_map[i] = "mid-port"
            continue
        x_norm = (float(xy[:, 0].mean()) - x_min) / xr
        cy = float(xy[:, 1].mean())
        if x_norm < bow_threshold:
            zone = "bow"
        elif x_norm > (1.0 - bow_threshold):
            zone = "crows-nest"
        else:
            zone = "mid"
        side = "port" if cy < centerline_y else "starboard"
        section_map[i] = f"{zone}-{side}"
    return section_map


def _format_path_d(xy: np.ndarray, precision: int) -> str:
    fmt = f"{{:.{precision}f}}"
    parts = [f"M {fmt.format(xy[0,0])} {fmt.format(xy[0,1])}"]
    row = " L ".join(f"{fmt.format(x)} {fmt.format(y)}" for x, y in xy[1:])
    parts.append(" L " + row)
    return "".join(parts)


def _save_section_debug(
    polylines: list[np.ndarray],
    section_map: dict[int, str],
    width: int,
    height: int,
    path: Path,
) -> None:
    """Colour-code polylines by section; write PNG preview."""
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for idx, xy in enumerate(polylines):
        if len(xy) < 2:
            continue
        sec = section_map.get(idx, "uncategorized")
        digest = hashlib.md5(sec.encode("utf-8")).digest()
        color = (int(digest[0]), int(digest[1]), int(digest[2]))
        pts = np.round(xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=1)
    cv2.imwrite(str(path), canvas)


def stage_svg_strings_grouped(
    polylines: list[np.ndarray],
    section_map: dict[int, str],
    width: int,
    height: int,
    stroke_width: float,
    stroke_color: str,
    svg_opt: bool,
    svg_precision: int,
) -> str:
    prec = max(0, int(svg_precision))
    by_section: dict[str, list[str]] = defaultdict(list)
    seen_global: set[str] = set()
    for idx, xy in enumerate(polylines):
        if len(xy) < 2:
            continue
        d = _format_path_d(xy, prec)
        if svg_opt:
            h = hashlib.sha256(d.encode()).hexdigest()[:16]
            if h in seen_global:
                continue
            seen_global.add(h)
        label = section_map.get(idx, "uncategorized")
        by_section[label].append(d)

    g_blocks: list[str] = []
    sw = stroke_width
    sc = stroke_color
    for label in sorted(by_section.keys()):
        d_strings = by_section[label]
        paths = "\n    ".join(f'<path d="{ds}" />' for ds in d_strings)
        g_blocks.append(
            f'  <g id="{label}" fill="none"\n'
            f'     stroke="{sc}"\n'
            f'     stroke-width="{sw}"\n'
            f'     stroke-linecap="round"\n'
            f'     stroke-linejoin="round">\n'
            f"    {paths}\n"
            f"  </g>"
        )
    inner = "\n".join(g_blocks)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width}"
     height="{height}"
     viewBox="0 0 {width} {height}">
{inner}
</svg>
"""


def bitmap_to_svg_centerline(
    image_name: str,
    output_root: Path,
    flags: PipelineFlags,
    threshold: int,
    min_area: int,
    blur_ksize: int,
    morph_ksize: int,
    epsilon: float,
    geo_tolerance: float,
    stroke_width: float,
    stroke_color: str,
    svg_precision: int,
    section_bow_threshold: float,
) -> Path:
    input_path = output_root / image_name
    if not input_path.is_file():
        print(f"Error: image not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    stem = input_path.stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "svg" / f"{stem}_{stamp}_centerline"
    run_dir.mkdir(parents=True, exist_ok=True)

    bgr = _load_image_bgr(input_path)

    # 1 — Grayscale
    work, preview = stage_grayscale(bgr, flags.grayscale)
    cv2.imwrite(str(run_dir / "01_grayscale.png"), preview)

    # 2 — Denoise
    work = stage_denoise(work, flags.denoise, blur_ksize)
    cv2.imwrite(str(run_dir / "02_denoised.png"), work)

    # 3 — Threshold
    work = stage_threshold(work, flags.threshold, threshold)
    cv2.imwrite(str(run_dir / "03_threshold.png"), work)

    # 4 — CC cleanup
    work = stage_cc_cleanup(work, flags.cc_cleanup, min_area)
    cv2.imwrite(str(run_dir / "04_clean.png"), work)

    # 5 — Morph (optional)
    work = stage_morph(work, flags.morph, morph_ksize)
    if flags.morph:
        cv2.imwrite(str(run_dir / "04b_morph.png"), work)

    # 6 — Skeleton
    skeleton = stage_skeletonize(work, flags.skeletonize)
    cv2.imwrite(str(run_dir / "05_skeleton.png"), skeleton)

    # 7 — Graph
    graph = stage_graph(skeleton, flags.graph)

    # 8 — Traversal
    if flags.traversal:
        polylines = stage_traversal(graph, enabled=True)
    else:
        polylines = stage_traversal_passthrough(graph)

    # 9 — Simplify
    polylines = stage_simplify(polylines, flags.simplify, epsilon)

    # 10 — Geo
    polylines = stage_geo_postproc(
        polylines, flags.geo_postproc, geo_tolerance
    )

    # 11 — Section detection + debug preview
    h, w = bgr.shape[:2]
    if flags.section_detect:
        section_map = stage_section_detect(
            polylines,
            width=w,
            height=h,
            bow_threshold=section_bow_threshold,
        )
    else:
        section_map = {i: "default" for i in range(len(polylines))}
    _save_section_debug(polylines, section_map, w, h, run_dir / "06_sections.png")

    # 12 — SVG (grouped by section)
    svg_text = stage_svg_strings_grouped(
        polylines,
        section_map,
        width=w,
        height=h,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        svg_opt=flags.svg_opt,
        svg_precision=svg_precision,
    )
    svg_path = run_dir / "output.svg"
    svg_path.write_text(svg_text, encoding="utf-8")

    print(f"Done. Output: {svg_path}")
    return svg_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert a PNG to stroke-only SVG via centerline skeleton graph "
            "(scikit-image + sknw)."
        ),
    )
    p.add_argument("image_name", help="Basename under output/, e.g. blueprint2.png")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Folder containing input images (default: {_DEFAULT_OUTPUT_DIR})",
    )

    # Per-stage disable flags (defaults: stages on except morph & geo_postproc)
    p.add_argument(
        "--no-grayscale",
        action="store_true",
        help="Skip BGR→gray conversion (use blue channel)",
    )
    p.add_argument("--no-denoise", action="store_true", help="Skip denoise filters")
    p.add_argument("--no-threshold", action="store_true", help="Skip binarisation")
    p.add_argument(
        "--no-cc-cleanup",
        action="store_true",
        help="Skip connected-component filtering",
    )
    p.add_argument(
        "--morph",
        action="store_true",
        help="Enable morphological closing repair (off by default)",
    )
    p.add_argument(
        "--no-skeletonize",
        action="store_true",
        help="Skip thinning (pass binary foreground through)",
    )
    p.add_argument(
        "--no-graph",
        action="store_true",
        help="Skip sknw graph build (no vector paths)",
    )
    p.add_argument(
        "--no-traversal",
        action="store_true",
        help="Skip traversal normalisation (raw sknw edge points)",
    )
    p.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip Douglas–Peucker simplification",
    )
    p.add_argument(
        "--geo-postproc",
        action="store_true",
        help="Enable Shapely geometry simplify (off by default)",
    )
    p.add_argument(
        "--no-svg-opt",
        action="store_true",
        help="Skip SVG coordinate rounding / duplicate path culling",
    )
    p.add_argument(
        "--no-section-detect",
        action="store_true",
        help="Skip ship-section grid split; emit a single group id=\"default\"",
    )
    p.add_argument(
        "--section-bow-threshold",
        type=float,
        default=0.33,
        help="Normalised X cutoff for bow/stern zones (default 0.33 = thirds)",
    )

    p.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Fixed threshold 1–255, or 0 for Otsu (default 0)",
    )
    p.add_argument(
        "--min-area",
        type=int,
        default=6,
        help="Min CC area to keep (default 6)",
    )
    p.add_argument(
        "--blur-ksize",
        type=int,
        default=3,
        help="Gaussian blur kernel size (odd; default 3)",
    )
    p.add_argument(
        "--morph-ksize",
        type=int,
        default=2,
        help="Morph close kernel size (default 2)",
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Douglas–Peucker epsilon in pixels (default 1.0)",
    )
    p.add_argument(
        "--geo-tolerance",
        type=float,
        default=0.5,
        help="Shapely simplify tolerance when --geo-postproc (default 0.5)",
    )
    p.add_argument("--stroke-width", type=float, default=1.0)
    p.add_argument("--stroke-color", type=str, default="#000000")
    p.add_argument(
        "--svg-precision",
        type=int,
        default=2,
        help="Decimal places in path coordinates when optimising (default 2)",
    )
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    flags = PipelineFlags(
        grayscale=not args.no_grayscale,
        denoise=not args.no_denoise,
        threshold=not args.no_threshold,
        cc_cleanup=not args.no_cc_cleanup,
        morph=args.morph,
        skeletonize=not args.no_skeletonize,
        graph=not args.no_graph,
        traversal=not args.no_traversal,
        simplify=not args.no_simplify,
        geo_postproc=args.geo_postproc,
        svg_opt=not args.no_svg_opt,
        section_detect=not args.no_section_detect,
    )

    bitmap_to_svg_centerline(
        args.image_name,
        args.output_dir.resolve(),
        flags,
        threshold=args.threshold,
        min_area=args.min_area,
        blur_ksize=args.blur_ksize,
        morph_ksize=args.morph_ksize,
        epsilon=args.epsilon,
        geo_tolerance=args.geo_tolerance,
        stroke_width=args.stroke_width,
        stroke_color=args.stroke_color,
        svg_precision=args.svg_precision,
        section_bow_threshold=args.section_bow_threshold,
    )


if __name__ == "__main__":
    main()
