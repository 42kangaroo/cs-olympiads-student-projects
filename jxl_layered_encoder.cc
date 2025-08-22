#include "../include/jxl/encode.h"
#include <vector>
#include <fstream>
#include <iostream>

using namespace std;

vector<float> loadImage(const string &path,
                        size_t num_channels,
                        size_t &width,
                        size_t &height)
{
    ifstream file(path);
    if (!file)
    {
        throw runtime_error("Failed to open file: " + path);
    }

    file >> width >> height;
    cout << "Image size (" << path << "): " << width << "x" << height << endl;

    const size_t frame_bytes = width * height * num_channels;
    vector<float> frame(frame_bytes);

    for (size_t i = 0; i < width * height; ++i)
    {
        for (size_t c = 0; c < num_channels; ++c)
        {
            file >> frame[i * num_channels + c];
        }
    }

    file.close();
    return frame;
}

int main(int argc, char *argv[])
{
    const size_t num_channels = 3;
    const size_t bytes_per_pixel = 4;

    if (argc < 4)
    {
        cerr << "Usage: " << argv[0] << " <high_res_image> <low_res_image> <output>" << endl;
        return 1;
    }

    size_t width_high, height_high;
    vector<float> frame_high = loadImage(argv[1], num_channels, width_high, height_high);

    size_t width_low, height_low;
    vector<float> frame_low = loadImage(argv[2], num_channels, width_low, height_low);

    if (width_high != 2 * width_low || height_high != 2 * height_low)
    {
        throw runtime_error("High resolution image must be double the size of low resolution image");
    }

    // Create encoder
    JxlEncoder *enc = JxlEncoderCreate(nullptr);

    // Basic image info
    JxlBasicInfo basic_info;
    JxlEncoderInitBasicInfo(&basic_info);
    basic_info.xsize = width_high;
    basic_info.ysize = height_high;
    basic_info.bits_per_sample = 8;
    basic_info.uses_original_profile = JXL_FALSE;
    basic_info.have_animation = false;
    basic_info.animation.tps_numerator = 1;
    basic_info.animation.tps_denominator = 1;
    basic_info.animation.num_loops = 0;
    JxlEncoderSetBasicInfo(enc, &basic_info);

    // Color encoding
    JxlColorEncoding color_encoding;
    JxlColorEncodingSetToSRGB(&color_encoding, JXL_FALSE);
    JxlEncoderSetColorEncoding(enc, &color_encoding);

    // Pixel format
    JxlPixelFormat pixel_format = {num_channels, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};

    // Add first frame
    {
        JxlEncoderFrameSettings *frame_settings = JxlEncoderFrameSettingsCreate(enc, nullptr);
        JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_RESAMPLING, 2);
        JxlEncoderFrameSettingsSetOption(frame_settings, JXL_ENC_FRAME_SETTING_ALREADY_DOWNSAMPLED, 1);

        if (argv[5])
        {
            JxlEncoderSetFrameDistance(frame_settings, atof(argv[5]));
        }

        JxlEncoderAddImageFrame(frame_settings, &pixel_format,
                                frame_low.data(), frame_low.size());
    }

    // Add second frame
    {
        JxlEncoderFrameSettings *frame_settings = JxlEncoderFrameSettingsCreate(enc, nullptr);
        // JxlBlendInfo *blend_info = new JxlBlendInfo;
        // JxlEncoderInitBlendInfo(blend_info);
        // std::cout << "JxlBlendInfo initialized." << std::endl;

        // blend_info->blendmode = JxlBlendMode::JXL_BLEND_ADD;

        // init JxlFrameHeader
        JxlFrameHeader *frame_header = new JxlFrameHeader;
        JxlEncoderInitFrameHeader(frame_header);
        std::cout << "JxlFrameHeader initialized." << std::endl;

        if (argv[4])
        {
            JxlEncoderSetFrameDistance(frame_settings, atof(argv[4]));
        }

        // frame_header->layer_info.blend_info = *blend_info;
        /* set frame header */

        // set frame_header as the header to use for the current frame
        JxlEncoderSetFrameHeader(frame_settings, frame_header);
        JxlEncoderAddImageFrame(frame_settings, &pixel_format,
                                frame_high.data(), frame_high.size());
    }

    // Close input so encoding can finish
    JxlEncoderCloseInput(enc);

    // Prepare output buffer
    std::vector<uint8_t> compressed(1024);
    uint8_t *next_out = compressed.data();
    size_t avail_out = compressed.size();

    // Process output
    JxlEncoderStatus status;
    do
    {
        status = JxlEncoderProcessOutput(enc, &next_out, &avail_out);
        if (status == JXL_ENC_NEED_MORE_OUTPUT)
        {
            size_t offset = next_out - compressed.data();
            compressed.resize(compressed.size() * 2);
            next_out = compressed.data() + offset;
            avail_out = compressed.size() - offset;
        }
    } while (status == JXL_ENC_NEED_MORE_OUTPUT);

    if (status != JXL_ENC_SUCCESS)
    {
        std::cerr << "Encoding failed!" << std::endl;
        JxlEncoderDestroy(enc);
        return 1;
    }

    // Save file
    size_t compressed_size = next_out - compressed.data();
    std::ofstream out(argv[3], std::ios::binary);
    out.write(reinterpret_cast<const char *>(compressed.data()), compressed_size);
    out.close();

    std::cout << "Saved " << argv[3] << " with size " << compressed_size << " bytes\n";

    // Clean up
    JxlEncoderDestroy(enc);

    return 0;
}
