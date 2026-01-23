#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <span>

using complex_t = std::complex<float>;

namespace audio_utils
{
enum class FFTTransformType
{
    Real,
    Complex
};

enum class FFTOutputType
{
    Magnitude,
    Power
};

struct ForwardFFTOptions
{
    FFTOutputType output_type = FFTOutputType::Magnitude;
    bool to_db = false;
};

class FFT
{
  public:
    FFT(uint32_t fft_size);
    ~FFT();

    void Forward(std::span<const float> signal, std::span<complex_t> spectrum);

    void ForwardMag(std::span<const float> signal, std::span<float> spectrum, const ForwardFFTOptions& options);

    void Inverse(std::span<const complex_t> spectrum, std::span<float> signal);

    void RealCepstrum(std::span<const float> signal, std::span<float> cepstrum);

    void Convolve(std::span<const float> signal, std::span<const float> filter, std::span<float> result);

    uint32_t GetFFTSize() const;

    uint32_t GetSpectrumSize() const;

    static uint32_t NextSupportedFFTSize(uint32_t min_size);

  private:
    struct FFTState;
    std::unique_ptr<FFTState> state_;
};
} // namespace audio_utils
