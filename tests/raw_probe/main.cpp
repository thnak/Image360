// Dev utility: exercises the exact ImageLoader calls RawIngestExecutor/
// AlignExecutor make (Open -> UnpackRaw / GetEmbeddedPreviewJpeg) against a
// real file path, so synthetic/hand-rolled RAW files can be validated
// against the vendored LibRaw directly instead of debugging failures
// through the whole pipeline. Not a ctest target - a manual diagnostic tool.

#include "HeaderFiles/ImageLoader.h"
#include "HeaderFiles/TextEncoding.h"
#include <iostream>

using namespace WindowsApp::Core;

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: raw_probe <path-to-raw-file>" << std::endl;
        return 2;
    }

    std::wstring path = Utf8ToWide(argv[1]);

    ImageLoader loader;
    if (!loader.Open(path))
    {
        std::wstring err = loader.GetLastError();
        std::wcerr << L"Open() FAILED: " << err << std::endl;
        return 1;
    }
    std::cout << "Open() OK" << std::endl;

    ImageMetadata meta;
    if (loader.GetMetadata(meta))
    {
        std::cout << "metadata: " << meta.width << "x" << meta.height
                  << " colors=" << meta.colors << std::endl;
    }

    RawPlane plane;
    if (!loader.UnpackRaw(plane))
    {
        std::wcerr << L"UnpackRaw() FAILED: " << loader.GetLastError() << std::endl;
    }
    else
    {
        const char* cfaName = "UNKNOWN";
        switch (plane.cfaType)
        {
        case CfaType::BAYER: cfaName = "BAYER"; break;
        case CfaType::X_TRANS: cfaName = "X_TRANS"; break;
        case CfaType::FOVEON: cfaName = "FOVEON"; break;
        default: break;
        }
        std::cout << "UnpackRaw() OK: " << plane.width << "x" << plane.height
                  << " cfaType=" << cfaName << " filters=0x" << std::hex << plane.filters << std::dec
                  << " blackLevel=" << plane.blackLevel
                  << " camMul=[" << plane.camMul[0] << "," << plane.camMul[1] << ","
                  << plane.camMul[2] << "," << plane.camMul[3] << "]" << std::endl;
        std::cout << "rgbCam=[";
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                std::cout << plane.rgbCam[i][j] << (i == 2 && j == 3 ? "" : ",");
        std::cout << "]" << std::endl;
    }

    std::vector<unsigned char> jpegBytes;
    if (!loader.GetEmbeddedPreviewJpeg(jpegBytes))
    {
        std::wcerr << L"GetEmbeddedPreviewJpeg() FAILED: " << loader.GetLastError() << std::endl;
    }
    else
    {
        std::cout << "GetEmbeddedPreviewJpeg() OK: " << jpegBytes.size() << " bytes, starts with "
                  << std::hex << (int)jpegBytes[0] << " " << (int)jpegBytes[1] << std::dec << std::endl;
    }

    return 0;
}
