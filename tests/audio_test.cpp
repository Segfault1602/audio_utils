#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <iostream>
#include <mdspan>
#include <random>
#include <vector>

#include <sndfile.h>

#include "audio_utils/audio_analysis.h"
#include "audio_utils/fft.h"
#include "audio_utils/fft_utils.h"
#include "test_utils.h"

TEST_CASE("FFTReal")
{
    auto signal = test_utils::LoadTestSignal(test_utils::kTestSignalFilename);

    const uint32_t nfft = signal.size();

    audio_utils::FFT fft(nfft);
    std::vector<std::complex<float>> signal_spectrum(fft.GetSpectrumSize(), {0.f, 0.f});
    fft.Forward(signal, signal_spectrum);

    auto test_signal_spectrum = test_utils::LoadTestSignalSpectrum(test_utils::kTestSignalSpectrumFilename);
    REQUIRE(test_signal_spectrum.size() >= signal_spectrum.size());

    for (size_t i = 0; i < signal_spectrum.size(); ++i)
    {
        REQUIRE_THAT(signal_spectrum[i].real(), Catch::Matchers::WithinAbs(test_signal_spectrum[i].real(), 1e-4f) ||
                                                    Catch::Matchers::WithinRel(test_signal_spectrum[i].real(), 0.001f));
        REQUIRE_THAT(signal_spectrum[i].imag(), Catch::Matchers::WithinAbs(test_signal_spectrum[i].imag(), 1e-4f) ||
                                                    Catch::Matchers::WithinRel(test_signal_spectrum[i].imag(), 0.001f));
    }

    std::vector<float> reconstructed_signal(nfft);
    fft.Inverse(signal_spectrum, reconstructed_signal);

    for (size_t i = 0; i < signal.size(); ++i)
    {
        REQUIRE_THAT(reconstructed_signal[i], Catch::Matchers::WithinAbs(signal[i], 1e-6f));
    }

    // Oversampling

    const uint32_t nfft2 = audio_utils::FFT::NextSupportedFFTSize(nfft * 2);
    REQUIRE(nfft2 == 8192);

    audio_utils::FFT fft2(nfft2);
    signal_spectrum.resize(fft2.GetSpectrumSize(), 0.f);
    fft2.Forward(signal, signal_spectrum);

    test_signal_spectrum = test_utils::LoadTestSignalSpectrum(test_utils::kTestSignalOversampledSpectrumFilename);
    REQUIRE(test_signal_spectrum.size() >= signal_spectrum.size());

    for (size_t i = 0; i < signal_spectrum.size(); ++i)
    {
        REQUIRE_THAT(signal_spectrum[i].real(), Catch::Matchers::WithinAbs(test_signal_spectrum[i].real(), 1e-3f) ||
                                                    Catch::Matchers::WithinRel(test_signal_spectrum[i].real(), 0.001f));
        REQUIRE_THAT(signal_spectrum[i].imag(), Catch::Matchers::WithinAbs(test_signal_spectrum[i].imag(), 1e-3f) ||
                                                    Catch::Matchers::WithinRel(test_signal_spectrum[i].imag(), 0.001f));
    }

    reconstructed_signal.clear();
    reconstructed_signal.resize(nfft2, 0.f);
    fft2.Inverse(signal_spectrum, reconstructed_signal);

    for (size_t i = 0; i < signal.size(); ++i)
    {
        REQUIRE_THAT(reconstructed_signal[i], Catch::Matchers::WithinAbs(signal[i], 1e-6f));
    }

    for (size_t i = signal.size(); i < reconstructed_signal.size(); ++i)
    {
        REQUIRE_THAT(reconstructed_signal[i], Catch::Matchers::WithinAbs(0.f, 1e-6f));
    }
}

TEST_CASE("FFTMagnitude")
{
    auto signal = test_utils::LoadTestSignal(test_utils::kTestSignalFilename);

    const uint32_t nfft = signal.size();

    audio_utils::FFT fft(nfft);
    std::vector<float> signal_spectrum(fft.GetSpectrumSize());
    fft.ForwardMag(signal, signal_spectrum, {audio_utils::FFTOutputType::Magnitude, false});

    REQUIRE(signal_spectrum.size() == fft.GetSpectrumSize());

    auto test_signal_spectrum = test_utils::LoadTestSignalMetric(test_utils::kTestSignalAbsSpectrum);
    REQUIRE(test_signal_spectrum.size() >= signal_spectrum.size());

    for (size_t i = 1; i < signal_spectrum.size(); ++i)
    {
        REQUIRE_THAT(signal_spectrum[i], Catch::Matchers::WithinRel(test_signal_spectrum[i], 0.001f));
    }

    fft.ForwardMag(signal, signal_spectrum, {audio_utils::FFTOutputType::Magnitude, true});
    auto test_signal_db_spectrum = test_utils::LoadTestSignalMetric(test_utils::kTestSignalDbSpectrum);
    REQUIRE(test_signal_db_spectrum.size() >= signal_spectrum.size());

    for (size_t i = 0; i < signal_spectrum.size(); ++i)
    {
        REQUIRE_THAT(signal_spectrum[i], Catch::Matchers::WithinRel(test_signal_db_spectrum[i], 0.01f));
    }
}

TEST_CASE("RealCepstrum")
{
    auto signal = test_utils::LoadTestSignal(test_utils::kTestSignalFilename);

    const uint32_t nfft = signal.size();
    std::vector<float> cepstrum(signal.size());

    audio_utils::FFT fft(nfft);
    fft.RealCepstrum(signal, cepstrum);

    auto test_signal_cepstrum = test_utils::LoadTestSignalMetric(test_utils::kTestSignalCepstrum);
    REQUIRE(test_signal_cepstrum.size() == cepstrum.size());

    for (auto i = 0; i < 10; ++i)
    {
        REQUIRE_THAT(cepstrum[i], Catch::Matchers::WithinAbs(test_signal_cepstrum[i], 1e-5f));
    }
}

TEST_CASE("Convolution")
{
    constexpr uint32_t kFFTSize = 1024;
    constexpr uint32_t kFilterSize = 128;
    constexpr uint32_t kSignalSize = 512;

    static_assert(kFilterSize + kSignalSize - 1 <= kFFTSize, "Filter and signal size exceed FFT size");

    audio_utils::FFT fft(kFFTSize);

    // Impulse signal

    std::vector<float> signal(kSignalSize, 0.f);
    signal[0] = 1.f;

    std::vector<float> filter(kFilterSize, 0.f);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (uint32_t i = 0; i < kFilterSize; ++i)
    {
        filter[i] = dist(gen);
    }

    std::vector<float> result(kSignalSize + kFilterSize - 1, 0.f);

    fft.Convolve(signal, filter, result);

    for (uint32_t i = 0; i < kFilterSize; ++i)
    {
        REQUIRE_THAT(result[i], Catch::Matchers::WithinAbs(filter[i], 1e-6f));
    }
}

TEST_CASE("Autocorrelation")
{
    auto signal = test_utils::LoadTestSignal(test_utils::kTestSignalFilename);
    // std::vector<float> signal = {1, 2, 3, 0, 0, 0, 0, 0};

    auto autocorr = audio_utils::analysis::Autocorrelation(signal, true);

    auto test_signal_autocorr = test_utils::LoadTestSignalMetric(test_utils::kTestSignalAutocorr);
    REQUIRE(test_signal_autocorr.size() == autocorr.size());

    for (auto i = 0; i < autocorr.size(); ++i)
    {
        // std::cout << autocorr[i] << " ";
        REQUIRE_THAT(autocorr[i], Catch::Matchers::WithinAbs(test_signal_autocorr[i], 1e-5f));
    }
}

TEST_CASE("Spectrogram")
{
    auto signal = test_utils::LoadTestSignal(test_utils::kTestSignalFilename);

    audio_utils::analysis::SpectrogramInfo info{};
    info.fft_size = 256;
    info.window_size = 128;
    info.overlap = 127;
    info.samplerate = test_utils::kSampleRate;
    info.window_type = audio_utils::FFTWindowType::Hann;

    auto spectrogram = audio_utils::analysis::STFT(signal, info);

    std::mdspan<float, std::dextents<size_t, 2>, std::layout_left> spec_mdspan{
        spectrogram.data.data(), spectrogram.num_bins, spectrogram.num_frames};

    uint32_t row_count = 0;
    uint32_t col_count = 0;
    auto test_signal_spectrogram =
        test_utils::LoadTestSignal2DMetric(test_utils::kTestSignalSpectrogram, row_count, col_count);

    std::cout << "Spectrogram size: " << spectrogram.num_bins << " x " << spectrogram.num_frames << std::endl;
    std::cout << "Test spectrogram size: " << row_count << " x " << col_count << std::endl;
    REQUIRE(spectrogram.num_bins == row_count);
    REQUIRE(spectrogram.num_frames == col_count);

    auto ref_mdspan = std::mdspan(test_signal_spectrogram.data(), row_count, col_count);

    for (auto i = 0; i < spectrogram.num_bins; ++i)
    {
        for (auto j = 0; j < spectrogram.num_frames; ++j)
        {
            REQUIRE_THAT((spec_mdspan[i, j]), Catch::Matchers::WithinAbs((ref_mdspan[i, j]), 1e-5f));
        }
    }
}

TEST_CASE("SpectralFlatness")
{
    constexpr uint32_t kSize = 8192;
    std::vector<float> noise(kSize, 0.f);

    // Fill the noise vector with random values
    std::mt19937 rng(0); // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.f, 1.f);
    for (auto& sample : noise)
    {
        sample = dist(rng);
    }

    audio_utils::FFT fft(kSize);
    std::vector<float> spectrum(fft.GetSpectrumSize(), 0.f);

    fft.ForwardMag(noise, spectrum, {audio_utils::FFTOutputType::Power, false});

    float flatness = audio_utils::analysis::SpectralFlatness(spectrum);

    // Not a perfect test, but flatness of white noise should be around 0.56 based on Matlab implementation
    REQUIRE_THAT(flatness, Catch::Matchers::WithinAbs(0.56f, 0.005f));
}