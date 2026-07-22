#include "pch.h"
#include "HeaderFiles/Tiff16Writer.h"
#include "HeaderFiles/PlatformFile.h"

#include <cstdint>
#include <vector>

namespace WindowsApp::Core
{
    namespace
    {
        void AppendU16(std::vector<unsigned char>& buf, uint16_t v)
        {
            buf.push_back(static_cast<unsigned char>(v & 0xFF));
            buf.push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
        }

        void AppendU32(std::vector<unsigned char>& buf, uint32_t v)
        {
            buf.push_back(static_cast<unsigned char>(v & 0xFF));
            buf.push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
            buf.push_back(static_cast<unsigned char>((v >> 16) & 0xFF));
            buf.push_back(static_cast<unsigned char>((v >> 24) & 0xFF));
        }

        // type: 3=SHORT, 4=LONG, 5=RATIONAL. value: for SHORT/LONG with
        // count==1, the value itself (left-justified into the 4-byte
        // field per TIFF6's "Value/Offset" convention - a lone SHORT is
        // stored in the low 2 bytes); otherwise an offset into the file.
        void AppendIfdEntry(std::vector<unsigned char>& buf, uint16_t tag, uint16_t type, uint32_t count,
                             uint32_t value)
        {
            AppendU16(buf, tag);
            AppendU16(buf, type);
            AppendU32(buf, count);
            AppendU32(buf, value);
        }
    }

    bool WriteTiff16RGB(const std::wstring& path, const unsigned short* data, int width, int height)
    {
        if (!data || width <= 0 || height <= 0) return false;

        constexpr uint32_t kHeaderSize = 8;
        const size_t pixelDataSize = static_cast<size_t>(width) * height * 3 * sizeof(unsigned short);
        const uint32_t stripOffset = kHeaderSize;

        // External value blocks (values too wide for a 4-byte IFD entry)
        // - written immediately after the pixel data.
        const uint32_t bitsPerSampleOffset = static_cast<uint32_t>(kHeaderSize + pixelDataSize);
        const uint32_t xResolutionOffset = bitsPerSampleOffset + 6; // 3 x SHORT
        const uint32_t yResolutionOffset = xResolutionOffset + 8;   // RATIONAL = 2 x LONG
        const uint32_t ifdOffset = yResolutionOffset + 8;

        std::vector<unsigned char> out;
        out.reserve(kHeaderSize + pixelDataSize + 64);

        // Header: byte order "II" (little-endian), magic 42, IFD offset.
        out.push_back('I'); out.push_back('I');
        AppendU16(out, 42);
        AppendU32(out, ifdOffset);

        // Pixel data - native little-endian uint16 samples (this project
        // only targets x86/x64 Windows/Linux, both little-endian, so no
        // byte-swap is needed).
        out.insert(out.end(), reinterpret_cast<const unsigned char*>(data),
                   reinterpret_cast<const unsigned char*>(data) + pixelDataSize);

        // BitsPerSample external block: {16, 16, 16}.
        AppendU16(out, 16); AppendU16(out, 16); AppendU16(out, 16);
        // XResolution / YResolution external blocks: 72/1 (RATIONAL).
        AppendU32(out, 72); AppendU32(out, 1);
        AppendU32(out, 72); AppendU32(out, 1);

        // IFD - tags MUST be in ascending order per the TIFF6 spec.
        constexpr uint16_t kEntryCount = 12;
        AppendU16(out, kEntryCount);
        AppendIfdEntry(out, 256, 4, 1, static_cast<uint32_t>(width));                 // ImageWidth
        AppendIfdEntry(out, 257, 4, 1, static_cast<uint32_t>(height));                // ImageLength
        AppendIfdEntry(out, 258, 3, 3, bitsPerSampleOffset);                          // BitsPerSample
        AppendIfdEntry(out, 259, 3, 1, 1);                                            // Compression = none
        AppendIfdEntry(out, 262, 3, 1, 2);                                            // PhotometricInterpretation = RGB
        AppendIfdEntry(out, 273, 4, 1, stripOffset);                                  // StripOffsets
        AppendIfdEntry(out, 277, 3, 1, 3);                                            // SamplesPerPixel
        AppendIfdEntry(out, 278, 4, 1, static_cast<uint32_t>(height));                // RowsPerStrip (single strip)
        AppendIfdEntry(out, 279, 4, 1, static_cast<uint32_t>(pixelDataSize));         // StripByteCounts
        AppendIfdEntry(out, 282, 5, 1, xResolutionOffset);                            // XResolution
        AppendIfdEntry(out, 283, 5, 1, yResolutionOffset);                            // YResolution
        AppendIfdEntry(out, 296, 3, 1, 2);                                            // ResolutionUnit = inches
        AppendU32(out, 0); // next IFD offset - none

        PlatformFile file;
        if (!file.Open(path, FileOpenMode::CreateAlways)) return false;
        bool ok = file.Write(out.data(), out.size());
        file.Close();
        return ok;
    }
}
