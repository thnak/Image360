"""Draws a synthetic textured scene and slices it into overlapping tiles,
simulating a sequence of photos taken across a panorama with enough overlap
for feature-based stitching to find correspondences between neighbors.

Usage: uv run gen_scene.py <output_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCENE_W, SCENE_H = 736, 240
TILE_W, TILE_H = 320, 240
NUM_TILES = 3
STRIDE = 208  # (SCENE_W - TILE_W) / (NUM_TILES - 1) = (736-320)/2 = 208
# Kept small deliberately - these tiles are wrapped into synthetic DNGs and
# checked into tests/pipeline_e2e/fixtures/, so file size matters (see
# scripts/panorama_gen/README - regenerate via generate_all.py if you need
# a bigger/different scene).


def draw_scene(rng: np.random.Generator) -> Image.Image:
    # Background: smooth horizontal gradient so every tile shares some
    # low-frequency content, plus per-pixel noise for texture.
    xs = np.linspace(0, 1, SCENE_W, dtype=np.float32)
    grad = (60 + 120 * xs).astype(np.uint8)
    base = np.tile(grad, (SCENE_H, 1))
    rgb = np.stack([base, np.roll(base, 40), np.roll(base, -40)], axis=-1).astype(np.int16)

    noise = rng.integers(-12, 13, size=(SCENE_H, SCENE_W, 3), dtype=np.int16)
    rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)

    # Scatter a variety of distinct shapes across the whole scene so that
    # every overlap band between neighboring tiles contains several sharp
    # corners/edges - real feature detectors (FAST/ORB-style) need this,
    # a flat gradient alone gives them nothing to lock onto.
    shapes = int((SCENE_W * SCENE_H) / 6000)
    for _ in range(shapes):
        x = int(rng.integers(0, SCENE_W))
        y = int(rng.integers(0, SCENE_H))
        size = int(rng.integers(14, 46))
        color = tuple(int(c) for c in rng.integers(20, 235, size=3))
        kind = rng.integers(0, 3)
        if kind == 0:
            draw.rectangle([x, y, x + size, y + size], outline=color, width=3)
        elif kind == 1:
            draw.ellipse([x, y, x + size, y + size], outline=color, width=3)
        else:
            x2 = int(np.clip(x + rng.integers(-size, size), 0, SCENE_W - 1))
            y2 = int(np.clip(y + rng.integers(-size, size), 0, SCENE_H - 1))
            draw.line([x, y, x2, y2], fill=color, width=3)

    # A few small clusters of irregular speckle noise - dense, strong
    # corners for a FAST/BRIEF-style detector to key on, but NOT periodic:
    # a checkerboard (tried first) gave every corner an identical local
    # descriptor, so the matcher confused cells within a pattern with
    # each other and RANSAC converged on a homography biased by ~tens of
    # pixels (the classic feature-matching "aperture problem" on
    # repetitive textures) - irregular speckles keep every local patch
    # visually unique.
    for _ in range(6):
        cx = int(rng.integers(0, SCENE_W - 96))
        cy = int(rng.integers(0, SCENE_H - 96))
        for _ in range(140):
            dx = int(rng.integers(0, 96))
            dy = int(rng.integers(0, 96))
            r = int(rng.integers(2, 5))
            shade = int(rng.integers(10, 60))
            draw.ellipse([cx + dx, cy + dy, cx + dx + r, cy + dy + r], fill=(shade, shade, shade))

    return img


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = out_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed=20260709)
    scene = draw_scene(rng)
    scene.save(out_dir / "scene.png")
    # Also saved as JPEG - this is the ground-truth reference the C++
    # pipeline_e2e test's quality-comparison scenario decodes and compares
    # the assembled panorama export against (see
    # tests/pipeline_e2e/fixtures/scene_reference.jpg). Quality matches
    # what that test uses for its own PanoramaExporter::ExportPreviewJpeg
    # call, so JPEG-quantization noise applies symmetrically to both
    # sides of the comparison rather than biasing it.
    scene.save(out_dir / "scene_reference.jpg", quality=95)

    tiles_meta = []
    for i in range(NUM_TILES):
        x0 = i * STRIDE
        crop = scene.crop((x0, 0, x0 + TILE_W, TILE_H))
        tile_path = tiles_dir / f"tile_{i}.png"
        crop.save(tile_path)
        tiles_meta.append({"index": i, "file": str(tile_path.name), "x_offset": x0, "y_offset": 0,
                            "width": TILE_W, "height": TILE_H})
        print(f"wrote {tile_path} at x_offset={x0}")

    meta = {
        "scene_width": SCENE_W,
        "scene_height": SCENE_H,
        "tile_width": TILE_W,
        "tile_height": TILE_H,
        "stride": STRIDE,
        "overlap_px": TILE_W - STRIDE,
        "tiles": tiles_meta,
    }
    with open(out_dir / "tiles_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"overlap between neighboring tiles: {TILE_W - STRIDE}px ({(TILE_W - STRIDE) / TILE_W:.0%} of tile width)")


if __name__ == "__main__":
    main()
