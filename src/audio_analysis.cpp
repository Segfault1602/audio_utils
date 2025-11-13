#include "audio_utils/audio_analysis.h"

#include <vector>

#include "audio_utils/fft.h"
#include "audio_utils/fft_utils.h"

#include <Eigen/Core>

namespace
{
float HzToMel(float hz)
{
    constexpr float lin_step = 200.f / 3.f;
    constexpr float change_point = 1000.f;
    const float logstep = std::log(6.4f) / 27.f;

    if (hz <= change_point)
    {
        return hz / lin_step;
    }

    float mel = (change_point / lin_step) + (std::log(hz / change_point) / logstep);

    return mel;
}

float MelToHz(float mel)
{
    constexpr float lin_step = 200.f / 3.f;
    constexpr float change_point = 1000.f;
    const float logstep = std::log(6.4f) / 27.f;
    const float change_point_mel = change_point / lin_step;

    if (mel <= change_point_mel)
    {
        return mel * lin_step;
    }

    float hz = change_point * std::exp(logstep * (mel - change_point_mel));
    return hz;
}

Eigen::MatrixXf GetMelFilter(size_t n_mels, size_t nfft, size_t sample_rate)
{
    Eigen::Vector2f range;
    range << HzToMel(0.0f), HzToMel(sample_rate / 2.0f);

    Eigen::VectorXf band_edges = Eigen::VectorXf::LinSpaced(n_mels + 2, range[0], range[1]);
    band_edges = band_edges.unaryExpr([](float mel) { return MelToHz(mel); });

    Eigen::MatrixXf filter_bank = Eigen::MatrixXf::Zero(nfft / 2 + 1, n_mels);

    Eigen::VectorXf linear_frequencies =
        Eigen::VectorXf::LinSpaced(nfft / 2 + 1, 0.0f, nfft / 2) / static_cast<float>(nfft) * sample_rate;

    Eigen::VectorXf p = Eigen::VectorXf::Zero(band_edges.size());

    for (size_t i = 0; i < band_edges.size(); ++i)
    {
        for (size_t j = 0; j < linear_frequencies.size(); ++j)
        {
            if (linear_frequencies[j] >= band_edges[i])
            {
                p[i] = j;
                break;
            }
        }
    }

    Eigen::VectorXf bw = band_edges.tail(band_edges.size() - 1) - band_edges.head(band_edges.size() - 1);

    for (size_t k = 0; k < n_mels; ++k)
    {
        // Rising side of triangle
        for (size_t j = p[k]; j < p[k + 1]; ++j)
        {
            filter_bank(j, k) = (linear_frequencies[j] - band_edges[k]) / bw[k];
        }

        // Falling side of triangle
        for (size_t j = p[k + 1]; j < p[k + 2]; ++j)
        {
            filter_bank(j, k) = (band_edges[k + 2] - linear_frequencies[j]) / bw[k + 1];
        }
    }

    Eigen::VectorXf weight = filter_bank.colwise().sum();
    for (size_t i = 0; i < filter_bank.cols(); ++i)
    {
        if (weight[i] > 0.0f)
        {
            filter_bank.col(i) /= weight[i];
        }
    }

    return filter_bank.transpose();
}
} // namespace

namespace audio_utils::analysis
{

std::vector<float> Autocorrelation(std::span<const float> signal, bool normalize)
{
    const uint32_t kNFFT = FFT::NextSupportedFFTSize(2 * signal.size());
    std::vector<float> out(kNFFT, 0.0f);

    FFT fft(kNFFT);

    std::vector<std::complex<float>> spectrum((kNFFT / 2) + 1, 0);

    fft.Forward(signal, spectrum);

    for (size_t i = 0; i < spectrum.size(); ++i)
    {
        spectrum[i] = std::pow(std::abs(spectrum[i]), 2.f);
    }

    fft.Inverse(spectrum, out);

    // Only keep the first half (positive lags)
    out.resize(signal.size());

    if (normalize)
    {
        float zero_lag = out[0];

        for (auto& val : out)
        {
            val /= zero_lag;
        }
    }

    return out;
}

float SpectralFlatness(std::span<const float> spectrum)
{
    float geo_mean = 1.0f;
    float arith_mean = 0.0f;

    for (const auto& mag : spectrum)
    {
        const float power = mag * mag;
        geo_mean += std::log(power + std::numeric_limits<float>::epsilon());
        arith_mean += power;
    }

    geo_mean = std::exp(geo_mean / static_cast<float>(spectrum.size()));
    arith_mean /= static_cast<float>(spectrum.size());

    if (arith_mean == 0.0f)
    {
        return 0.0f;
    }

    return geo_mean / arith_mean;
}

SpectrogramResult STFT(std::span<const float> signal, SpectrogramInfo& info, bool flip)
{
    if (info.overlap >= info.fft_size)
    {
        throw std::invalid_argument("Overlap must be less than FFT size");
    }

    if (info.overlap >= info.window_size)
    {
        throw std::invalid_argument("Overlap must be less than window size");
    }

    const uint32_t hop = info.window_size - info.overlap;
    const uint32_t num_frames = (signal.size() - info.overlap) / hop;
    const uint32_t num_bins = info.fft_size / 2 + 1;

    std::vector<float> result;
    result.resize(num_frames * num_bins, -9999.999f);

    std::vector<float> window(info.window_size);
    GetWindow(info.window_type, window);

    FFT fft(info.fft_size);

    auto spec_span = std::span(result);

    std::vector<float> frame(info.window_size);

    for (auto i = 0; i < num_frames; ++i)
    {
        auto signal_span = signal.subspan(i * hop, info.window_size);

        for (size_t j = 0; j < info.window_size; ++j)
        {
            frame[j] = signal_span[j] * window[j];
        }

        auto spectrum_subspan = spec_span.subspan(i * num_bins, num_bins);
        fft.ForwardAbs(frame, spectrum_subspan, false, false);
    }

    if (flip)
    {
        Eigen::Map<Eigen::MatrixXf> spec_map(result.data(), num_bins, num_frames);
        spec_map.colwise().reverseInPlace();
    }

    SpectrogramResult result_struct;
    result_struct.data = std::move(result);
    result_struct.num_bins = num_bins;
    result_struct.num_frames = num_frames;

    return result_struct;
}

SpectrogramResult MelSpectrogram(std::span<const float> signal, SpectrogramInfo& info, size_t n_mels, bool flip)
{
    SpectrogramResult result = STFT(signal, info, false);

    Eigen::MatrixXf mel_weights = GetMelFilter(n_mels, info.fft_size, info.samplerate);

    std::vector<float> mel_data(result.num_frames * n_mels, -50.0f); // Initialize with -50 dB

    Eigen::Map<Eigen::MatrixXf> result_map(mel_data.data(), n_mels, result.num_frames);
    Eigen::Map<Eigen::MatrixXf> stft_map(result.data.data(), result.num_bins, result.num_frames);

    for (auto i = 0; i < stft_map.cols(); ++i)
    {
        Eigen::VectorXf mel_spectrum = mel_weights * stft_map.col(i);
        result_map.col(i) = mel_spectrum;
    }

    if (flip)
    {
        result_map.colwise().reverseInPlace();
    }

    result.data = std::move(mel_data);
    result.num_bins = n_mels;

    return result;
}

} // namespace audio_utils::analysis