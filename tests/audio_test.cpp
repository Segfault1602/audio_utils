#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <iostream>
#include <mdspan>
#include <sys/types.h>
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
    std::vector<std::complex<float>> signal_spectrum((nfft / 2) + 1);

    audio_utils::FFT fft(nfft);
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
    signal_spectrum.resize((nfft2 / 2) + 1, 0.f);
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

TEST_CASE("AbsFFT")
{
    auto signal = test_utils::LoadTestSignal(test_utils::kTestSignalFilename);

    const uint32_t nfft = signal.size();
    std::vector<float> signal_spectrum((nfft / 2) + 1);

    audio_utils::FFT fft(nfft);
    fft.ForwardAbs(signal, signal_spectrum, false, false);

    REQUIRE(signal_spectrum.size() == (nfft / 2) + 1);

    auto test_signal_spectrum = test_utils::LoadTestSignalMetric(test_utils::kTestSignalAbsSpectrum);
    REQUIRE(test_signal_spectrum.size() >= signal_spectrum.size());

    for (size_t i = 1; i < signal_spectrum.size(); ++i)
    {
        REQUIRE_THAT(signal_spectrum[i], Catch::Matchers::WithinRel(test_signal_spectrum[i], 0.001f));
    }

    fft.ForwardAbs(signal, signal_spectrum, true, false);
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

TEST_CASE("Autocorrelation")
{
    auto signal = test_utils::LoadTestSignal(test_utils::kTestSignalFilename);

    auto autocorr = audio_utils::analysis::Autocorrelation(signal, true);

    auto test_signal_autocorr = test_utils::LoadTestSignalMetric(test_utils::kTestSignalAutocorr);
    REQUIRE(test_signal_autocorr.size() == autocorr.size());

    for (auto i = 0; i < autocorr.size(); ++i)
    {
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

    std::mdspan<float, std::dextents<size_t, 2>, std::layout_left> spec_mdspan{spectrogram.data(), info.num_freqs,
                                                                               info.num_frames};

    uint32_t row_count = 0;
    uint32_t col_count = 0;
    auto test_signal_spectrogram =
        test_utils::LoadTestSignal2DMetric(test_utils::kTestSignalSpectrogram, row_count, col_count);

    std::cout << "Spectrogram size: " << info.num_freqs << " x " << info.num_frames << std::endl;
    std::cout << "Test spectrogram size: " << row_count << " x " << col_count << std::endl;
    REQUIRE(info.num_freqs == row_count);
    REQUIRE(info.num_frames == col_count);

    auto ref_mdspan = std::mdspan(test_signal_spectrogram.data(), row_count, col_count);

    for (auto i = 0; i < info.num_freqs; ++i)
    {
        for (auto j = 0; j < info.num_frames; ++j)
        {
            REQUIRE_THAT((spec_mdspan[i, j]), Catch::Matchers::WithinAbs((ref_mdspan[i, j]), 1e-5f));
        }
    }
}