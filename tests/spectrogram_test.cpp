#include <fstream>
#include <iostream>
#include <vector>

#include <sndfile.h>

#include "fft_utils.h"

int main()
{
    constexpr const char* file_name = "./tests/chirp.wav";
    SF_INFO info;
    SNDFILE* file = sf_open(file_name, SFM_READ, &info);

    if (file == nullptr)
    {
        std::cerr << "Failed to open file: " << file_name << std::endl;
        return 1;
    }

    std::cout << "Opened file: " << file_name << std::endl;
    std::cout << "Channels: " << info.channels << std::endl;
    std::cout << "Sample rate: " << info.samplerate << std::endl;
    std::cout << "Frames: " << info.frames << std::endl;
    std::cout << "Format:" << info.format << std::endl;

    std::vector<float> buffer(info.frames * info.channels);

    int read = sf_readf_float(file, buffer.data(), info.frames);

    if (read != info.frames)
    {
        std::cerr << "Failed to read file" << std::endl;
        return 1;
    }

    sf_close(file);

    spectrogram_info spec_info;
    spec_info.fft_size = 1024;
    spec_info.overlap = 512;
    spec_info.samplerate = info.samplerate;
    spec_info.window_type = FFTWindowType::Hann;

    std::vector<float> spec = Spectrogram(buffer.data(), info.frames, spec_info);

    // size of spec should be a multiple of spec_info.fft_size / 2
    const size_t num_freqs = spec_info.fft_size / 2 + 1;
    if (spec.size() % num_freqs != 0)
    {
        std::cerr << "Invalid size of spectrogram" << std::endl;
        return 1;
    }

    std::ofstream out_file("spectrogram.txt");
    if (!out_file.is_open())
    {
        std::cerr << "Failed to open output file" << std::endl;
        return 1;
    }

    for (size_t i = 0; i < spec.size(); i += num_freqs)
    {
        for (size_t j = 0; j < num_freqs; ++j)
        {
            out_file << spec[i + j] << " ";
        }
        out_file << std::endl;
    }
}