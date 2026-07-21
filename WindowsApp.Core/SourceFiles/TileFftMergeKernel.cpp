#include "pch.h"
#include "HeaderFiles/TileFftMergeKernel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

using namespace WindowsApp::Compute;

namespace WindowsApp::Core::Kernels
{
    namespace
    {
        using Complex = std::complex<float>;
        constexpr float kPi = 3.14159265358979323846f;

        bool IsPowerOfTwo(int n) { return n > 0 && (n & (n - 1)) == 0; }

        // Iterative radix-2 Cooley-Tukey, in place. inverse=true also
        // applies the 1/n normalization (so callers never need a separate
        // normalization pass).
        void Fft1D(Complex* a, int n, bool inverse)
        {
            for (int i = 1, j = 0; i < n; ++i)
            {
                int bit = n >> 1;
                for (; j & bit; bit >>= 1) j ^= bit;
                j ^= bit;
                if (i < j) std::swap(a[i], a[j]);
            }

            for (int len = 2; len <= n; len <<= 1)
            {
                float ang = (inverse ? 1.0f : -1.0f) * 2.0f * kPi / static_cast<float>(len);
                Complex wlen(std::cos(ang), std::sin(ang));
                for (int i = 0; i < n; i += len)
                {
                    Complex w(1.0f, 0.0f);
                    int half = len / 2;
                    for (int k = 0; k < half; ++k)
                    {
                        Complex u = a[i + k];
                        Complex v = a[i + k + half] * w;
                        a[i + k] = u + v;
                        a[i + k + half] = u - v;
                        w *= wlen;
                    }
                }
            }

            if (inverse)
            {
                for (int i = 0; i < n; ++i) a[i] /= static_cast<float>(n);
            }
        }

        // In-place 2D FFT on an n*n row-major buffer: row passes, then
        // column passes (gather/transform/scatter - tiles are tiny, cache
        // locality doesn't matter at this scale).
        void Fft2D(std::vector<Complex>& tile, int n, bool inverse)
        {
            for (int y = 0; y < n; ++y) Fft1D(&tile[static_cast<size_t>(y) * n], n, inverse);

            std::vector<Complex> col(n);
            for (int x = 0; x < n; ++x)
            {
                for (int y = 0; y < n; ++y) col[y] = tile[static_cast<size_t>(y) * n + x];
                Fft1D(col.data(), n, inverse);
                for (int y = 0; y < n; ++y) tile[static_cast<size_t>(y) * n + x] = col[y];
            }
        }

        // Clamp-to-edge sample - used both for the reference tile (right/
        // bottom border tiles narrower than tileSize) and for alternate
        // frames' offset-shifted sampling window (an offset can push the
        // window partly outside [0,width)x[0,height) even for an interior
        // tile). Never zero-pads: a hard edge-to-zero transition would
        // ring badly through the FFT.
        inline unsigned short SampleClamped(const unsigned short* frame, int width, int height,
                                             int px, int py, int channel)
        {
            int cx = (std::max)(0, (std::min)(width - 1, px));
            int cy = (std::max)(0, (std::min)(height - 1, py));
            return frame[(static_cast<size_t>(cy) * width + cx) * 3 + channel];
        }
    }

    bool TileFftMerge(
        const unsigned short* const* frames, int numFrames,
        const TileOffset* const* perFrameOffsets,
        int width, int height, int tileSize, int tilesX, int tilesY,
        float noiseVariance, unsigned short* output)
    {
        if (!IsPowerOfTwo(tileSize)) return false;
        if (numFrames < 1) return false;

        const int n = tileSize;
        const size_t n2 = static_cast<size_t>(n) * n;
        std::vector<Complex> refSpectrum(n2), altSpectrum(n2), merged(n2);
        const float invAltCount = 1.0f / static_cast<float>((std::max)(1, numFrames - 1));

        for (int ty = 0; ty < tilesY; ++ty)
        {
            for (int tx = 0; tx < tilesX; ++tx)
            {
                int baseX = tx * tileSize;
                int baseY = ty * tileSize;
                int validW = (std::min)(tileSize, width - baseX);
                int validH = (std::min)(tileSize, height - baseY);

                for (int c = 0; c < 3; ++c)
                {
                    for (int ly = 0; ly < n; ++ly)
                        for (int lx = 0; lx < n; ++lx)
                            refSpectrum[static_cast<size_t>(ly) * n + lx] = Complex(
                                static_cast<float>(SampleClamped(frames[0], width, height, baseX + lx, baseY + ly, c)),
                                0.0f);
                    Fft2D(refSpectrum, n, false);

                    merged = refSpectrum;

                    for (int k = 1; k < numFrames; ++k)
                    {
                        const TileOffset& off = perFrameOffsets[k - 1][static_cast<size_t>(ty) * tilesX + tx];
                        for (int ly = 0; ly < n; ++ly)
                            for (int lx = 0; lx < n; ++lx)
                                altSpectrum[static_cast<size_t>(ly) * n + lx] = Complex(
                                    static_cast<float>(SampleClamped(frames[k], width, height,
                                        baseX + lx + off.dx, baseY + ly + off.dy, c)),
                                    0.0f);
                        Fft2D(altSpectrum, n, false);

                        for (size_t p = 0; p < n2; ++p)
                        {
                            Complex diff = altSpectrum[p] - refSpectrum[p];
                            float power = std::norm(diff); // |diff|^2
                            float denom = power + noiseVariance;
                            float weight = (denom > 0.0f) ? (power / denom) : 0.0f;
                            merged[p] += (weight * invAltCount) * diff;
                        }
                    }

                    Fft2D(merged, n, true);

                    for (int ly = 0; ly < validH; ++ly)
                    {
                        for (int lx = 0; lx < validW; ++lx)
                        {
                            float v = merged[static_cast<size_t>(ly) * n + lx].real();
                            v = (std::max)(0.0f, (std::min)(65535.0f, v));
                            size_t pixel = (static_cast<size_t>(baseY + ly) * width + (baseX + lx)) * 3 + c;
                            output[pixel] = static_cast<unsigned short>(v + 0.5f);
                        }
                    }
                }
            }
        }

        return true;
    }
}
