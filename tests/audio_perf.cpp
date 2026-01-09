#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include "audio_utils/audio_analysis.h"
#include "audio_utils/fft.h"
#include "test_utils.h"

#include <array>
#include <cstdint>
#include <format>
#include <random>

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("FFT")
{
    nanobench::Bench bench;
    bench.title("FFT Perf");
    bench.timeUnit(1ns, "ns");

    constexpr std::array kNFFTSizes = {256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u, 32768u, 44100u, 48000u};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto nfft : kNFFTSizes)
    {
        std::vector<float> test_signal;
        test_signal.reserve(nfft);

        const uint32_t supported_fft_size = audio_utils::FFT::NextSupportedFFTSize(nfft);
        if (supported_fft_size != nfft)
            continue;

        audio_utils::FFT fft(nfft);
        for (uint32_t i = 0; i < nfft; ++i)
        {
            test_signal.push_back(dist(gen));
        }

        bench.minEpochIterations(10000000 / nfft);
        bench.batch(nfft);
        bench.unit("samples");

        std::vector<std::complex<float>> test_spectrum((nfft / 2) + 1);
        std::string title = std::format("FFT (NFFT={})", nfft);
        bench.run(title, [&] {
            fft.Forward(test_signal, test_spectrum);
            nanobench::doNotOptimizeAway(test_spectrum);
        });
    }
}

TEST_CASE("ForwardAbs")
{
    nanobench::Bench bench;
    bench.title("FFT Perf - ForwardAbs");
    bench.timeUnit(1ns, "ns");

    constexpr std::array kNFFTSizes = {256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u, 32768u, 44100u, 48000u};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto nfft : kNFFTSizes)
    {
        std::vector<float> test_signal;
        test_signal.reserve(nfft);

        const uint32_t supported_fft_size = audio_utils::FFT::NextSupportedFFTSize(nfft);
        if (supported_fft_size != nfft)
            continue;

        audio_utils::FFT fft(nfft);
        for (uint32_t i = 0; i < nfft; ++i)
        {
            test_signal.push_back(dist(gen));
        }

        bench.minEpochIterations(5000000 / nfft);
        bench.batch(nfft);
        bench.unit("samples");

        std::vector<float> test_spectrum((nfft / 2) + 1);
        std::string title = std::format("FFT (NFFT={})", nfft);
        bench.run(title, [&] {
            fft.ForwardAbs(test_signal, test_spectrum, true);
            nanobench::doNotOptimizeAway(test_spectrum);
        });
    }
}

TEST_CASE("Cepstrum")
{
    auto signal = test_utils::LoadTestSignal(test_utils::kTestSignalFilename);

    uint32_t nfft = audio_utils::FFT::NextSupportedFFTSize(signal.size());
    audio_utils::FFT fft(nfft);
    std::vector<float> cepstrum(nfft);

    nanobench::Bench bench;
    bench.title("Cepstrum Perf");
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(10000);

    std::string title = std::format("Cepstrum (NFFT={})", nfft);

    bench.run(title, [&] {
        fft.RealCepstrum(signal, cepstrum);
        nanobench::doNotOptimizeAway(cepstrum);
    });

    signal = test_utils::LoadTestSignal(test_utils::kImpulseResponseFilename);
    nfft = audio_utils::FFT::NextSupportedFFTSize(signal.size());
    audio_utils::FFT fft2 = audio_utils::FFT(nfft);

    cepstrum.resize(nfft);
    title = std::format("Cepstrum  (NFFT={})", signal.size());
    bench.minEpochIterations(100);
    bench.run(title, [&] {
        fft2.RealCepstrum(signal, cepstrum);
        nanobench::doNotOptimizeAway(cepstrum);
    });
}

TEST_CASE("Convolution")
{
    constexpr uint32_t kFilterSize = 4096;
    constexpr uint32_t kSignalSize = 48000;

    const uint32_t kFFTSize = audio_utils::FFT::NextSupportedFFTSize(kFilterSize + kSignalSize - 1);

    audio_utils::FFT fft(kFFTSize);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> signal(kSignalSize, 0.f);
    for (uint32_t i = 0; i < kSignalSize; ++i)
    {
        signal[i] = dist(gen);
    }

    std::vector<float> filter(kFilterSize, 0.f);
    for (uint32_t i = 0; i < kFilterSize; ++i)
    {
        filter[i] = dist(gen);
    }

    std::vector<float> result(kSignalSize + kFilterSize - 1, 0.f);

    nanobench::Bench bench;
    bench.title("Convolution Perf");
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(500);

    bench.run("Convolution", [&] {
        fft.Convolve(signal, filter, result);
        nanobench::doNotOptimizeAway(result);
    });
}

TEST_CASE("Spectrogram")
{
    constexpr audio_utils::analysis::SpectrogramInfo info1{.fft_size = 1024,
                                                           .overlap = 256,
                                                           .window_size = 1024,
                                                           .samplerate = test_utils::kSampleRate,
                                                           .window_type = audio_utils::FFTWindowType::Hann};

    constexpr audio_utils::analysis::SpectrogramInfo info2{.fft_size = 2048,
                                                           .overlap = 512,
                                                           .window_size = 2048,
                                                           .samplerate = test_utils::kSampleRate,
                                                           .window_type = audio_utils::FFTWindowType::Hann};

    constexpr audio_utils::analysis::SpectrogramInfo info3{.fft_size = 8192,
                                                           .overlap = 2048,
                                                           .window_size = 8192,
                                                           .samplerate = test_utils::kSampleRate,
                                                           .window_type = audio_utils::FFTWindowType::Hann};

    constexpr std::array kSpectrogramInfos = {info1, info2, info3};

    auto signal = test_utils::LoadTestSignal(test_utils::kImpulseResponseFilename);

    nanobench::Bench bench;
    bench.title("Spectrogram Perf");
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(1000);

    for (const auto& info : kSpectrogramInfos)
    {
        std::string title = std::format("Spectrogram (FFTSize={}, WindowSize={}, Overlap={})", info.fft_size,
                                        info.window_size, info.overlap);
        auto info_copy = info;
        bench.run(title, [&] {
            auto spectrogram = audio_utils::analysis::STFT(signal, info_copy);
            nanobench::doNotOptimizeAway(spectrogram);
        });
    }
}