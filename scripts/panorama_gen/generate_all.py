"""Orchestrates the full synthetic test-fixture generation: draws the scene,
crops overlapping tiles, and wraps each tile as a synthetic DNG the
Image360 pipeline can actually ingest.

Usage: uv run generate_all.py <output_dir>
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import gen_scene
import make_dng


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out")

    sys.argv = [sys.argv[0], str(out_dir)]
    gen_scene.main()

    meta_path = out_dir / "tiles_meta.json"
    meta = json.loads(meta_path.read_text())

    for tile in meta["tiles"]:
        png_path = out_dir / "tiles" / tile["file"]
        dng_path = png_path.with_suffix(".dng")

        img = Image.open(png_path).convert("RGB")
        rgb16 = (np.array(img, dtype=np.uint8).astype(np.uint16)) << 8
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        make_dng.write_dng(rgb16, buf.getvalue(), dng_path)

        tile["dng_file"] = dng_path.name
        print(f"tile {tile['index']}: {dng_path.name} (x_offset={tile['x_offset']})")

    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"updated {meta_path}")


if __name__ == "__main__":
    main()
