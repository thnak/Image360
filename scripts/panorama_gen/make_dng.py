"""Wraps an RGB image into a minimal single-file synthetic DNG that the
vendored LibRaw (WindowsApp.Core/libraw) will accept:

- IFD0: the real raw plane - Bayer-mosaiced (RGGB) 16-bit CFA data,
  PhotometricInterpretation=32803, tagged with DNGVersion so LibRaw's
  `dng_version` DNG code path engages (WindowsApp.Core/libraw/src/metadata/
  tiff.cpp ~line 917-922). CFARepeatPatternDim/CFAPattern give LibRaw a
  nonzero `filters` value (required - RawIngestExecutor/AlignExecutor both
  reject CfaType::UNKNOWN, which is what an absent/zero `filters` maps to).
- IFD1: NewSubfileType=1 (preview), holding a real embedded JPEG via the
  classic JPEGInterchangeFormat/Length tags (0x0201/0x0202) - LibRaw
  auto-detects the real JPEG SOI/SOF header there (tiff.cpp's "collide w/
  Panasonic" tag switch, ~line 91-137) regardless of this IFD's own
  Compression/PhotometricInterpretation tags, which is what makes
  GetEmbeddedPreviewJpeg()/unpack_thumb() succeed. Kept at the SAME
  resolution as the raw plane (not a real camera's usual downscaled
  thumbnail) so Align's feature coordinates map 1:1 onto the full-res
  color-corrected buffer Render later warps - avoiding a preview-vs-fullres
  scale mismatch that isn't otherwise handled by this pipeline and is out
  of scope for this synthetic test to fix.

  Critically, that JPEG auto-detect only fires when `tiff_ifd[ifd].bps==0`
  at the moment tag 0x0201 is parsed (tiff.cpp:105). Tags are read in
  ascending numeric order within an IFD, and BitsPerSample (0x0102=258, <
  0x0201=513) unconditionally sets `.bps` in a second, generic tag switch
  further down the same function (tiff.cpp:440-446) - so if the thumbnail
  IFD carries its own BitsPerSample tag at all, `.bps` is already nonzero
  by the time 0x0201 is reached and the JPEG auto-detect is silently
  skipped (confirmed empirically: LibRaw returned LIBRAW_NO_THUMBNAIL when
  writing this IFD via tifffile's normal tif.write(), which always emits a
  BitsPerSample tag for whatever array you hand it). tifffile's high-level
  writer offers no way to omit that tag, so this IFD is instead built by
  hand: only NewSubfileType/Compression/PhotometricInterpretation/
  JPEGInterchangeFormat/JPEGInterchangeFormatLength, appended as raw bytes
  after the main IFD (whose own tifffile-written "next IFD offset" field
  is then patched to point at it).
"""
import struct
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

DNG_VERSION_TAG = 0xC612
CFA_REPEAT_PATTERN_DIM_TAG = 0x828D
CFA_PATTERN_TAG = 0x828E
CFA_PLANE_COLOR_TAG = 0xC616
NEW_SUBFILE_TYPE_TAG = 0xFE
AS_SHOT_NEUTRAL_TAG = 0xC628
JPEG_INTERCHANGE_FORMAT_TAG = 0x0201
JPEG_INTERCHANGE_FORMAT_LENGTH_TAG = 0x0202

# LibRaw CFAPattern byte order is [row0col0, row0col1, row1col0, row1col1]
# with color indices 0=R,1=G,2=B,3=G2 - this is RGGB.
CFA_PATTERN_RGGB = bytes([0, 1, 1, 2])


def mosaic_to_bayer_rggb(rgb: np.ndarray) -> np.ndarray:
    """rgb: HxWx3 uint16 -> HxW uint16 CFA plane, RGGB pattern
    (row0: R,G,R,G...; row1: G,B,G,B...)."""
    h, w, _ = rgb.shape
    cfa = np.empty((h, w), dtype=np.uint16)
    cfa[0::2, 0::2] = rgb[0::2, 0::2, 0]  # R
    cfa[0::2, 1::2] = rgb[0::2, 1::2, 1]  # G
    cfa[1::2, 0::2] = rgb[1::2, 0::2, 1]  # G
    cfa[1::2, 1::2] = rgb[1::2, 1::2, 2]  # B
    return cfa


def write_dng(rgb: np.ndarray, jpeg_bytes: bytes, out_path: Path) -> None:
    h, w, _ = rgb.shape
    cfa = mosaic_to_bayer_rggb(rgb)

    main_extratags = [
        (DNG_VERSION_TAG, "B", 4, (1, 4, 0, 0), False),
        (NEW_SUBFILE_TYPE_TAG, "I", 1, 0, False),
        (CFA_REPEAT_PATTERN_DIM_TAG, "H", 2, (2, 2), False),
        (CFA_PATTERN_TAG, "B", 4, tuple(CFA_PATTERN_RGGB), False),
        (CFA_PLANE_COLOR_TAG, "B", 3, (0, 1, 2), False),
        # Deliberately NO ColorMatrix1/CalibrationIlluminant1 tag - two
        # earlier attempts to set an explicit ColorMatrix1 (first identity,
        # then the standard sRGB->XYZ matrix) both made LibRaw's resulting
        # rgb_cam WORSE, not better: DNG's ColorMatrix1 maps camera RGB ->
        # XYZ, and LibRaw's cam_xyz_coeff() (identify.cpp) composes that
        # with its OWN fixed XYZ->sRGB step and then inverts the whole
        # thing (traced via WindowsApp.Core/libraw/src/utils/
        # utils_dcraw.cpp's cam_xyz_coeff), so neither "identity" nor "the
        # forward sRGB matrix" cancels out correctly - only the exact
        # inverse of LibRaw's own XYZ->sRGB matrix does, and that's just
        # extra fragility for no benefit here. Simpler: without ANY
        # ColorMatrix tag, `use_cm` never gets set and cam_xyz_coeff is
        # never called at all, so `imgdata.color.rgb_cam` stays the
        # literal identity LibRaw initializes it to (identify.cpp
        # ~476-482) - exactly what's wanted, since this synthetic
        # fixture's CFA data is mosaiced directly from sRGB PNG/JPEG
        # pixels (not real sensor data), so "camera RGB" already IS sRGB.
        # (Safe as long as Make/Model - neither set here - don't happen to
        # match a real camera's adobe_coeff() table entry.)
        #
        # AsShotNeutral is still needed though (a SEPARATE code path from
        # ColorMatrix1/cam_xyz_coeff) - without it, cam_mul defaults to
        # {0,1,0,0} (identify.cpp's initial default), zeroing the red
        # channel entirely during DemosaicBayer's white-balance step.
        (AS_SHOT_NEUTRAL_TAG, 5, 3, [(1_000_000, 1_000_000)] * 3, False),
    ]

    with tifffile.TiffWriter(out_path, byteorder=">") as tif:
        tif.write(
            cfa,
            photometric=32803,  # CFA
            planarconfig=1,
            compression=None,  # Compression=1 (uncompressed) -> packed_dng_load_raw
            extratags=main_extratags,
            software=None,
        )

    with tifffile.TiffFile(out_path) as tif:
        page0 = tif.pages[0]
        ifd0_offset = page0.offset
        num_tags = len(page0.tags)
    next_ifd_field_pos = ifd0_offset + 2 + num_tags * 12

    with open(out_path, "r+b") as f:
        f.seek(0, 2)  # end of file
        jpeg_offset = f.tell()
        f.write(jpeg_bytes)
        ifd1_offset = f.tell()
        f.write(_build_thumbnail_ifd(jpeg_offset, len(jpeg_bytes)))

        f.seek(next_ifd_field_pos)
        (existing_next,) = struct.unpack(">I", f.read(4))
        assert existing_next == 0, f"expected no existing next-IFD, got {existing_next}"
        f.seek(next_ifd_field_pos)
        f.write(struct.pack(">I", ifd1_offset))

    # Sanity re-check: re-parse the file and confirm IFD1 is visible with
    # exactly the tags we intended, at the offset we computed.
    def _scalar(v):
        return v[0] if isinstance(v, tuple) else v

    with tifffile.TiffFile(out_path) as tif:
        assert len(tif.pages) == 2, f"expected 2 IFDs, tifffile sees {len(tif.pages)}"
        thumb_tags = {t.code: t.value for t in tif.pages[1].tags}
        assert _scalar(thumb_tags[JPEG_INTERCHANGE_FORMAT_TAG]) == jpeg_offset
        assert _scalar(thumb_tags[JPEG_INTERCHANGE_FORMAT_LENGTH_TAG]) == len(jpeg_bytes)
        assert 258 not in thumb_tags, "thumbnail IFD must not carry a BitsPerSample tag"


def _build_thumbnail_ifd(jpeg_offset: int, jpeg_length: int) -> bytes:
    """Hand-rolled minimal IFD (no BitsPerSample!) - see module docstring."""
    SHORT, LONG = 3, 4

    def entry(tag: int, dtype: int, count: int, value: int) -> bytes:
        head = struct.pack(">HHI", tag, dtype, count)
        if dtype == SHORT:
            return head + struct.pack(">HH", value, 0)
        return head + struct.pack(">I", value)

    entries = [
        entry(NEW_SUBFILE_TYPE_TAG, LONG, 1, 1),
        entry(0x0103, SHORT, 1, 6),  # Compression = old-style JPEG
        entry(0x0106, SHORT, 1, 6),  # PhotometricInterpretation = YCbCr
        entry(JPEG_INTERCHANGE_FORMAT_TAG, LONG, 1, jpeg_offset),
        entry(JPEG_INTERCHANGE_FORMAT_LENGTH_TAG, LONG, 1, jpeg_length),
    ]
    body = struct.pack(">H", len(entries)) + b"".join(entries) + struct.pack(">I", 0)
    return body


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: make_dng.py <input.png> <output.dng>", file=sys.stderr)
        sys.exit(2)

    src_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    img = Image.open(src_path).convert("RGB")
    rgb8 = np.array(img, dtype=np.uint8)
    rgb16 = (rgb8.astype(np.uint16)) << 8  # 8-bit source -> 16-bit raw range

    # Embedded preview: encode the ORIGINAL (non-mosaiced) RGB at full
    # resolution as a real baseline JPEG - this is what
    # GetEmbeddedPreviewJpeg()/Align's feature detector actually sees.
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    jpeg_bytes = buf.getvalue()

    write_dng(rgb16, jpeg_bytes, out_path)
    print(f"wrote {out_path} ({rgb16.shape[1]}x{rgb16.shape[0]} raw, {len(jpeg_bytes)}-byte embedded JPEG)")


if __name__ == "__main__":
    main()
