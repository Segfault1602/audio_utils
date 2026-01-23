#pragma once
#include <catch2/catch_test_macros.hpp>
#include <sndfile.h>

#include <complex>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace test_utils
{
constexpr uint32_t kSampleRate = 48000;

constexpr std::string_view kTestSignalFilename = "tests/test_signal.wav";
constexpr std::string_view kImpulseResponseFilename = "tests/ir_1.wav";

constexpr std::string_view kTestSignalSpectrumFilename = "tests/test_signal_spectrum_4096.txt";
constexpr std::string_view kTestSignalOversampledSpectrumFilename = "tests/test_signal_spectrum_8192.txt";

constexpr std::string_view kTestSignalAbsSpectrum = "tests/test_signal_mag_spectrum.txt";
constexpr std::string_view kTestSignalDbSpectrum = "tests/test_signal_db_spectrum.txt";
constexpr std::string_view kTestSignalCepstrum = "tests/test_signal_cepstrum.txt";
constexpr std::string_view kTestSignalAutocorr = "tests/test_signal_autocorr.txt";

constexpr std::string_view kTestSignalSpectrogram = "tests/test_signal_spectrogram.txt";

inline std::vector<float> LoadTestSignal(const std::string_view filename)
{
    SF_INFO sfinfo;
    sfinfo.format = 0;

    SNDFILE* infile = sf_open(filename.data(), SFM_READ, &sfinfo);
    if (!infile)
    {
        throw std::runtime_error("Failed to open test signal file");
    }

    REQUIRE(sfinfo.channels == 1);
    REQUIRE(sfinfo.samplerate == kSampleRate);

    std::vector<float> signal(sfinfo.frames);
    auto read_count = sf_readf_float(infile, signal.data(), sfinfo.frames);
    REQUIRE(read_count == sfinfo.frames);
    sf_close(infile);

    return signal;
}

inline std::vector<std::complex<float>> LoadTestSignalSpectrum(const std::string_view filename)
{
    std::ifstream infile(filename.data());
    REQUIRE(infile.is_open());

    std::vector<std::complex<float>> result;
    result.reserve(4096);

    std::string line;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        float real{0.f};
        float imag{0.f};
        char comma{0};
        if (!(iss >> real >> comma >> imag))
        {
            throw std::runtime_error("Failed to parse line: " + line);
        }
        REQUIRE(comma == ',');
        result.emplace_back(real, imag);
    }

    return result;
}

inline std::vector<float> LoadTestSignalMetric(const std::string_view filename)
{
    std::ifstream infile(filename.data());
    REQUIRE(infile.is_open());

    std::vector<float> result;
    result.reserve(4096);

    std::string line;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        float value{0.f};

        if (!(iss >> value))
        {
            throw std::runtime_error("Failed to parse line: " + line);
        }
        result.emplace_back(value);
    }

    return result;
}

inline std::vector<float> LoadTestSignal2DMetric(const std::string_view filename, uint32_t& out_rows,
                                                 uint32_t& out_cols)
{
    std::ifstream infile(filename.data());
    REQUIRE(infile.is_open());

    out_rows = 0;
    out_cols = 0;

    std::vector<float> result;
    result.reserve(4096);
    std::string line;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        float value{0.f};

        size_t col_count = 0;
        while (iss >> value)
        {
            result.emplace_back(value);
            ++col_count;
            if (iss.peek() == ',' || iss.peek() == ' ')
            {
                iss.ignore();
            }
        }

        if (out_cols == 0)
        {
            out_cols = col_count;
        }
        else
        {
            REQUIRE(col_count == out_cols);
        }
        ++out_rows;
    }

    return result;
}

inline std::vector<std::complex<float>> ToComplex(std::span<const float> real_signal)
{
    std::vector<std::complex<float>> complex_signal(real_signal.size());
    for (size_t i = 0; i < real_signal.size(); ++i)
    {
        complex_signal[i] = std::complex<float>(real_signal[i], 0.f);
    }
    return complex_signal;
}

} // namespace test_utils