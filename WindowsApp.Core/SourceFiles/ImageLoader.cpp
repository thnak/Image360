#include "pch.h"
#include "HeaderFiles/ImageLoader.h"
#include "libraw/libraw.h"
#include <algorithm>

namespace WindowsApp::Core
{
    struct ImageLoader::Impl
    {
        LibRaw processor;
        bool isOpen = false;
        bool isDecoded = false;
        std::wstring lastError;

        void SetError(const std::wstring& msg)
        {
            lastError = msg;
        }

        void SetErrorFromCode(int ret)
        {
            switch (ret)
            {
            case LIBRAW_FILE_UNSUPPORTED:
                lastError = L"File format not supported by LibRaw.";
                break;
            case LIBRAW_REQUEST_FOR_NONEXISTENT_IMAGE:
                lastError = L"Requested image index does not exist.";
                break;
            case LIBRAW_UNSUFFICIENT_MEMORY:
                lastError = L"Insufficient memory for decoding.";
                break;
            case LIBRAW_DATA_ERROR:
                lastError = L"Data error during RAW processing.";
                break;
            case LIBRAW_IO_ERROR:
                lastError = L"I/O error reading RAW file.";
                break;
            case LIBRAW_CANCELLED_BY_CALLBACK:
                lastError = L"Processing cancelled by callback.";
                break;
            default:
                lastError = L"LibRaw error code: " + std::to_wstring(ret);
                break;
            }
        }
    };

    ImageLoader::ImageLoader()
        : m_impl(std::make_unique<Impl>())
    {
        // Configure LibRaw for 16-bit output
        m_impl->processor.imgdata.params.output_bps = 16;
        m_impl->processor.imgdata.params.output_color = 1; // sRGB
        m_impl->processor.imgdata.params.gamm[0] = 1.0;    // Linear gamma
        m_impl->processor.imgdata.params.gamm[1] = 1.0;
        m_impl->processor.imgdata.params.no_auto_bright = 1; // No auto brightness
        m_impl->processor.imgdata.params.highlight = 0;      // No highlight recovery
        m_impl->processor.imgdata.params.use_camera_wb = 1;  // Use camera white balance
    }

    ImageLoader::~ImageLoader()
    {
        Close();
    }

    bool ImageLoader::Open(const std::wstring& filePath)
    {
        Close();

        // Convert wide path to UTF-8
        int len = WideCharToMultiByte(CP_UTF8, 0, filePath.c_str(), -1, nullptr, 0, nullptr, nullptr);
        std::string utf8Path(len - 1, '\0');
        WideCharToMultiByte(CP_UTF8, 0, filePath.c_str(), -1, utf8Path.data(), len, nullptr, nullptr);

        int ret = m_impl->processor.open_file(utf8Path.c_str());
        if (ret != LIBRAW_SUCCESS)
        {
            m_impl->SetErrorFromCode(ret);
            return false;
        }

        // Unpack the RAW data
        ret = m_impl->processor.unpack();
        if (ret != LIBRAW_SUCCESS)
        {
            m_impl->SetErrorFromCode(ret);
            m_impl->processor.recycle();
            return false;
        }

        m_impl->isOpen = true;
        m_impl->isDecoded = false;
        return true;
    }

    void ImageLoader::Close()
    {
        if (m_impl->isOpen)
        {
            m_impl->processor.recycle();
            m_impl->isOpen = false;
            m_impl->isDecoded = false;
        }
    }

    bool ImageLoader::IsOpen() const
    {
        return m_impl->isOpen;
    }

    bool ImageLoader::GetMetadata(ImageMetadata& metadata) const
    {
        if (!m_impl->isOpen)
        {
            m_impl->SetError(L"No file is open.");
            return false;
        }

        const auto& sizes = m_impl->processor.imgdata.sizes;
        const auto& idata = m_impl->processor.imgdata.idata;
        const auto& other = m_impl->processor.imgdata.other;

        metadata.width = sizes.width;
        metadata.height = sizes.height;
        metadata.bitsPerSample = 16;
        metadata.colors = idata.colors;

        // Camera info
        if (idata.make[0])
        {
            int wlen = MultiByteToWideChar(CP_UTF8, 0, idata.make, -1, nullptr, 0);
            metadata.cameraMake.resize(wlen - 1);
            MultiByteToWideChar(CP_UTF8, 0, idata.make, -1, metadata.cameraMake.data(), wlen);
        }
        if (idata.model[0])
        {
            int wlen = MultiByteToWideChar(CP_UTF8, 0, idata.model, -1, nullptr, 0);
            metadata.cameraModel.resize(wlen - 1);
            MultiByteToWideChar(CP_UTF8, 0, idata.model, -1, metadata.cameraModel.data(), wlen);
        }

        metadata.isoSpeed = other.iso_speed;
        metadata.shutterSpeed = other.shutter;
        metadata.aperture = other.aperture;
        metadata.focalLength = other.focal_len;
        metadata.orientation = sizes.flip;

        return true;
    }

    bool ImageLoader::DecodeFull(PixelBuffer& output)
    {
        return DecodeInternal(output, 100, 0, 0, 0, 0);
    }

    bool ImageLoader::DecodeThumbnail(PixelBuffer& output, int scalePercent)
    {
        if (scalePercent < 1 || scalePercent > 100)
        {
            m_impl->SetError(L"Scale percent must be 1-100.");
            return false;
        }
        return DecodeInternal(output, scalePercent, 0, 0, 0, 0);
    }

    bool ImageLoader::DecodeROI(int x, int y, int width, int height, PixelBuffer& output)
    {
        if (x < 0 || y < 0 || width <= 0 || height <= 0)
        {
            m_impl->SetError(L"Invalid ROI parameters.");
            return false;
        }
        return DecodeInternal(output, 100, x, y, width, height);
    }

    bool ImageLoader::DecodeInternal(PixelBuffer& output, int scalePercent, int roiX, int roiY, int roiW, int roiH)
    {
        if (!m_impl->isOpen)
        {
            m_impl->SetError(L"No file is open.");
            return false;
        }

        // Set output parameters
        m_impl->processor.imgdata.params.half_size = (scalePercent < 100) ? 1 : 0;

        // Process the RAW data
        int ret = m_impl->processor.dcraw_process();
        if (ret != LIBRAW_SUCCESS)
        {
            m_impl->SetErrorFromCode(ret);
            return false;
        }

        m_impl->isDecoded = true;

        // Get processed image
        libraw_processed_image_t* image = m_impl->processor.dcraw_make_mem_image(&ret);
        if (!image)
        {
            m_impl->SetErrorFromCode(ret);
            return false;
        }

        // Determine output dimensions
        int srcW = image->width;
        int srcH = image->height;
        int colors = image->colors;
        int bps = image->bits;

        // If ROI is specified, validate and crop
        int outX = 0, outY = 0, outW = srcW, outH = srcH;
        if (roiW > 0 && roiH > 0)
        {
            // For half_size mode, scale ROI coordinates
            if (scalePercent < 100)
            {
                float scale = static_cast<float>(scalePercent) / 100.0f;
                roiX = static_cast<int>(roiX * scale);
                roiY = static_cast<int>(roiY * scale);
                roiW = static_cast<int>(roiW * scale);
                roiH = static_cast<int>(roiH * scale);
            }

            outX = std::clamp(roiX, 0, srcW - 1);
            outY = std::clamp(roiY, 0, srcH - 1);
            outW = std::clamp(roiW, 1, srcW - outX);
            outH = std::clamp(roiH, 1, srcH - outY);
        }

        // Allocate output buffer (always 16-bit RGB48)
        output.width = outW;
        output.height = outH;
        output.data.resize(outW * outH * 3);

        // Copy and convert to unsigned short RGB48
        const unsigned char* srcData = image->data;

        for (int row = 0; row < outH; row++)
        {
            for (int col = 0; col < outW; col++)
            {
                int srcIdx = ((row + outY) * srcW + (col + outX)) * colors * (bps / 8);
                int dstIdx = (row * outW + col) * 3;

                if (bps == 16 && colors >= 3)
                {
                    // 16-bit RGB
                    const unsigned short* src16 = reinterpret_cast<const unsigned short*>(srcData + srcIdx);
                    output.data[dstIdx]     = src16[0]; // R
                    output.data[dstIdx + 1] = src16[1]; // G
                    output.data[dstIdx + 2] = src16[2]; // B
                }
                else if (bps == 8 && colors >= 3)
                {
                    // 8-bit RGB -> expand to 16-bit
                    output.data[dstIdx]     = static_cast<unsigned short>(srcData[srcIdx]) << 8;
                    output.data[dstIdx + 1] = static_cast<unsigned short>(srcData[srcIdx + 1]) << 8;
                    output.data[dstIdx + 2] = static_cast<unsigned short>(srcData[srcIdx + 2]) << 8;
                }
                else if (bps == 16 && colors == 1)
                {
                    // 16-bit Grayscale -> replicate to RGB
                    const unsigned short* src16 = reinterpret_cast<const unsigned short*>(srcData + srcIdx);
                    output.data[dstIdx] = output.data[dstIdx + 1] = output.data[dstIdx + 2] = src16[0];
                }
                else
                {
                    // Fallback: zero fill
                    output.data[dstIdx] = output.data[dstIdx + 1] = output.data[dstIdx + 2] = 0;
                }
            }
        }

        // Free LibRaw's image buffer
        LibRaw::dcraw_clear_mem(image);

        return true;
    }

    bool ImageLoader::UnpackRaw(RawPlane& output)
    {
        if (!m_impl->isOpen)
        {
            m_impl->SetError(L"No file is open.");
            return false;
        }

        const auto& sizes = m_impl->processor.imgdata.sizes;
        const auto& idata = m_impl->processor.imgdata.idata;
        const auto& color = m_impl->processor.imgdata.color;
        const auto& rawdata = m_impl->processor.imgdata.rawdata;

        if (!rawdata.raw_image)
        {
            m_impl->SetError(L"No unpacked raw CFA plane available for this file.");
            return false;
        }

        output.width = sizes.raw_width;
        output.height = sizes.raw_height;
        size_t numPixels = static_cast<size_t>(output.width) * static_cast<size_t>(output.height);
        output.cfaData.assign(rawdata.raw_image, rawdata.raw_image + numPixels);

        output.blackLevel = color.black;
        for (int i = 0; i < 4; ++i)
        {
            output.camMul[i] = color.cam_mul[i];
        }

        // cam_xyz_coeff converts the camera's XYZ color matrix into a
        // camera-RGB -> sRGB matrix (rgb_cam). Numeric correctness needs
        // real-hardware verification, same standing caveat as every other
        // GPU/RAW-numeric path in this plan.
        double camXyz[4][3];
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                camXyz[i][j] = color.cam_xyz[i][j];
            }
        }
        m_impl->processor.cam_xyz_coeff(output.rgbCam, camXyz);

        if (idata.is_foveon)
        {
            output.cfaType = CfaType::FOVEON;
        }
        else if (idata.filters == 9)
        {
            output.cfaType = CfaType::X_TRANS;
        }
        else if (idata.filters != 0)
        {
            output.cfaType = CfaType::BAYER;
        }
        else
        {
            output.cfaType = CfaType::UNKNOWN;
        }

        return true;
    }

    std::wstring ImageLoader::GetLastError() const
    {
        return m_impl->lastError;
    }
}
