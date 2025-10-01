#include "nanobench.h"
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <format>

#include "audio_utils/audio_analysis.h"
#include "audio_utils/fft.h"
#include "test_utils.h"

using namespace ankerl;
using namespace std::chrono_literals;

TEST_CASE("FFT")
{
    auto signal = test_utils::LoadTestSignal(test_utils::kTestSignalFilename);
    std::vector<std::complex<float>> signal_spectrum((signal.size() / 2) + 1);

    uint32_t nfft = signal.size();
    audio_utils::FFT fft(nfft);

    nanobench::Bench bench;
    bench.title("FFT Perf");
    bench.timeUnit(1us, "us");
    bench.minEpochIterations(10000);

    std::string title = std::format("FFT (NFFT={})", nfft);

    bench.run(title, [&] {
        fft.Forward(signal, signal_spectrum);
        nanobench::doNotOptimizeAway(signal_spectrum);
    });

    signal = test_utils::LoadTestSignal(test_utils::kImpulseResponseFilename);
    nfft = signal.size();
    nfft = audio_utils::FFT::NextSupportedFFTSize(nfft);
    audio_utils::FFT fft2 = audio_utils::FFT(nfft);

    signal_spectrum.resize((nfft / 2) + 1);
    title = std::format("FFT  (NFFT={})", nfft);
    bench.minEpochIterations(100);
    bench.run(title, [&] {
        fft2.Forward(signal, signal_spectrum);
        nanobench::doNotOptimizeAway(signal_spectrum);
    });

    std::vector<float> abs_spectrum((nfft / 2) + 1);
    bench.run(title, [&] {
        fft2.ForwardAbs(signal, abs_spectrum, true, false);
        nanobench::doNotOptimizeAway(abs_spectrum);
    });
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

TEST_CASE("Spectrogram")
{
    audio_utils::analysis::SpectrogramInfo info{};
    info.fft_size = 4096;
    info.window_size = 512;
    info.overlap = 400;
    info.samplerate = test_utils::kSampleRate;
    info.window_type = audio_utils::FFTWindowType::Hann;

    auto signal = test_utils::LoadTestSignal(test_utils::kImpulseResponseFilename);

    nanobench::Bench bench;
    bench.title("Spectrogram Perf");
    bench.timeUnit(1ms, "ms");
    bench.minEpochIterations(100);

    bench.run("Spectrogram", [&] {
        auto spectrogram = audio_utils::analysis::STFT(signal, info);
        nanobench::doNotOptimizeAway(spectrogram);
    });
}