#pragma once
#include <string>

namespace WindowsApp::Core
{
    // Minimal, self-contained, uncompressed baseline TIFF writer (docs/
    // superpowers/plans/2026-07-22-cli-front-end.md) - no libtiff or other
    // third-party dependency. Writes a single-strip, little-endian ("II"),
    // 3 x 16-bit-sample-per-pixel RGB TIFF: the lossless counterpart to
    // JpegCodec's 8-bit lossy path, for the CLI's burst-mode outputs (a
    // BURST_FINISH PixelBuffer has no existing on-disk writer at all).
    // `data` is interleaved RGB, width*height*3 unsigned shorts, same
    // layout as PixelBuffer::data. Returns false on any file I/O failure.
    bool WriteTiff16RGB(const std::wstring& path, const unsigned short* data, int width, int height);
}
