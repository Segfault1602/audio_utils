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

namespace
{
std::vector<float> GenerateRandomSignal(uint32_t size)
{
    std::vector<float> signal;
    signal.reserve(size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (uint32_t i = 0; i < size; ++i)
    {
        signal.push_back(dist(gen));
    }

    return signal;
}
} // namespace

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
        std::vector<float> test_signal = GenerateRandomSignal(nfft);

        const uint32_t supported_fft_size = audio_utils::FFT::NextSupportedFFTSize(nfft);
        if (supported_fft_size != nfft)
            continue;

        audio_utils::FFT fft(nfft);

        bench.minEpochIterations(10000000 / nfft);
        bench.batch(nfft);
        bench.unit("samples");

        std::vector<std::complex<float>> test_spectrum(fft.GetSpectrumSize());
        std::string title = std::format("FFT (NFFT={})", nfft);
        bench.run(title, [&] {
            fft.Forward(test_signal, test_spectrum);
            nanobench::doNotOptimizeAway(test_spectrum);
        });
    }
}

TEST_CASE("ForwardMag")
{
    nanobench::Bench bench;
    bench.title("FFT Perf - ForwardMag");
    bench.timeUnit(1ns, "ns");

    constexpr std::array kNFFTSizes = {256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u, 32768u, 44100u, 48000u};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto nfft : kNFFTSizes)
    {
        const uint32_t supported_fft_size = audio_utils::FFT::NextSupportedFFTSize(nfft);
        if (supported_fft_size != nfft)
            continue;

        audio_utils::FFT fft(nfft);
        std::vector<float> test_signal = GenerateRandomSignal(nfft);

        bench.minEpochIterations(5000000 / nfft);
        bench.batch(nfft);
        bench.unit("samples");

        std::vector<float> test_spectrum(fft.GetSpectrumSize());
        std::string title = std::format("FFT (NFFT={})", nfft);
        bench.run(title, [&] {
            fft.ForwardMag(test_signal, test_spectrum, {audio_utils::FFTOutputType::Magnitude, false});
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
    constexpr uint32_t kSignalSize = 32768;

    nanobench::Bench bench;
    bench.title("Convolution Perf");
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(1000);

    for (uint32_t filter_size = 32; filter_size <= 8192; filter_size *= 2)
    {
        const uint32_t kFFTSize = audio_utils::FFT::NextSupportedFFTSize(filter_size + kSignalSize - 1);

        audio_utils::FFT fft(kFFTSize);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> signal = GenerateRandomSignal(kSignalSize);

        std::vector<float> filter = GenerateRandomSignal(filter_size);

        std::vector<float> result(kSignalSize + filter_size - 1, 0.f);

        auto title = std::format("Convolution (FilterSize={})", filter_size);
        bench.run(title, [&] {
            fft.Convolve(signal, filter, result);
            nanobench::doNotOptimizeAway(result);
        });
    }
}

TEST_CASE("Spectrogram")
{
    constexpr audio_utils::analysis::STFTOptions info1{.fft_size = 1024,
                                                       .overlap = 256,
                                                       .window_size = 1024,
                                                       .window_type = audio_utils::FFTWindowType::Hann,
                                                       .samplerate = test_utils::kSampleRate};

    constexpr audio_utils::analysis::STFTOptions info2{.fft_size = 2048,
                                                       .overlap = 512,
                                                       .window_size = 2048,
                                                       .window_type = audio_utils::FFTWindowType::Hann,
                                                       .samplerate = test_utils::kSampleRate};

    constexpr audio_utils::analysis::STFTOptions info3{.fft_size = 8192,
                                                       .overlap = 2048,
                                                       .window_size = 8192,
                                                       .window_type = audio_utils::FFTWindowType::Hann,
                                                       .samplerate = test_utils::kSampleRate};

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

TEST_CASE("MelSpectrogram")
{
    auto signal = test_utils::LoadTestSignal(test_utils::kImpulseResponseFilename);

    audio_utils::analysis::STFTOptions spec_info{.fft_size = 2048,
                                                 .overlap = 512,
                                                 .window_size = 2048,
                                                 .window_type = audio_utils::FFTWindowType::Hann,
                                                 .samplerate = test_utils::kSampleRate};

    constexpr uint32_t kNumMelBands = 32;

    nanobench::Bench bench;
    bench.title("MelSpectrogram Perf");
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(1000);

    std::string title = std::format("MelSpectrogram (FFTSize={}, WindowSize={}, Overlap={}, MelBands={})",
                                    spec_info.fft_size, spec_info.window_size, spec_info.overlap, kNumMelBands);

    bench.run(title, [&] {
        auto mel_spectrogram = audio_utils::analysis::MelSpectrogram(signal, spec_info, kNumMelBands);
        nanobench::doNotOptimizeAway(mel_spectrogram);
    });
}

TEST_CASE("SpectralFlatness")
{
    auto signal = GenerateRandomSignal(48000);

    uint32_t nfft = audio_utils::FFT::NextSupportedFFTSize(signal.size());
    audio_utils::FFT fft(nfft);
    std::vector<float> spectrum(fft.GetSpectrumSize());

    fft.ForwardMag(signal, spectrum, {audio_utils::FFTOutputType::Power, false});

    nanobench::Bench bench;
    bench.title("Spectral Flatness Perf");
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(100);

    std::string title = std::format("Spectral Flatness (NFFT={})", nfft);

    bench.run(title, [&] {
        float flatness = audio_utils::analysis::SpectralFlatness(spectrum);
        nanobench::doNotOptimizeAway(flatness);
    });
}

TEST_CASE("Autocorrelation")
{
    auto signal = test_utils::LoadTestSignal(test_utils::kTestSignalFilename);

    nanobench::Bench bench;
    bench.title("Autocorrelation Perf");
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(1000);

    bench.run("Autocorrelation", [&] {
        auto autocorr = audio_utils::analysis::Autocorrelation(signal, false);
        nanobench::doNotOptimizeAway(autocorr);
    });

    bench.run("Autocorrelation (normalized)", [&] {
        auto autocorr = audio_utils::analysis::Autocorrelation(signal, true);
        nanobench::doNotOptimizeAway(autocorr);
    });
}

TEST_CASE("EnergyDecayCurve")
{
    auto signal = test_utils::LoadTestSignal(test_utils::kImpulseResponseFilename);

    nanobench::Bench bench;
    bench.title("Energy Decay Curve Perf");
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(1000);
    bench.run("Energy Decay Curve", [&] {
        auto edc = audio_utils::analysis::EnergyDecayCurve(signal, false);
        nanobench::doNotOptimizeAway(edc);
    });

    bench.run("TrimSilence", [&] {
        auto trimmed_signal = audio_utils::analysis::TrimSilence(signal, 0.5f);
        nanobench::doNotOptimizeAway(trimmed_signal);
    });
}