#include "audio_utils/audio_analysis.h"

#include "audio_utils/array_math.h"
#include "audio_utils/fft.h"
#include "audio_utils/fft_utils.h"

#include <Eigen/Core>

#ifdef AUDIO_UTILS_USE_IPP
#include <ipp.h>
#endif

#include <format>
#include <iterator>
#include <numeric>
#include <ranges>
#include <vector>

namespace
{

void ValidateSTFTOptions(const audio_utils::analysis::STFTOptions& options)
{
    const uint32_t fft_size = audio_utils::FFT::NextSupportedFFTSize(options.fft_size);
    if (options.fft_size != fft_size)
    {
        throw std::invalid_argument(
            std::format("FFT size {} is not supported. Next supported size is {}", options.fft_size, fft_size));
    }

    if (options.window_size == 0 || options.window_size > options.fft_size)
    {
        throw std::invalid_argument(
            std::format("Window size ({}) must be greater than zero and less than or equal to FFT size ({})",
                        options.window_size, options.fft_size));
    }

    if (options.overlap >= options.window_size)
    {
        throw std::invalid_argument(
            std::format("Overlap ({}) must be less than window size ({})", options.overlap, options.window_size));
    }
}

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
#ifndef AUDIO_UTILS_USE_IPP
    const uint32_t kNFFT = FFT::NextSupportedFFTSize(2 * signal.size());
    std::vector<float> out(kNFFT, 0.0f);

    FFT fft(kNFFT);

    std::vector<std::complex<float>> spectrum(fft.GetSpectrumSize(), 0);

    fft.Forward(signal, spectrum);

    array_math::PowerSpectrum(spectrum, spectrum);

    fft.Inverse(spectrum, out);

    // Only keep the first half (positive lags)
    out.resize(signal.size());

#else
    IppEnum func_cfg = static_cast<IppEnum>(ippAlgAuto);
    Ipp8u* work_buffer = nullptr;
    int buffer_size = 0;
    IppStatus status = ippsAutoCorrNormGetBufferSize(signal.size(), signal.size(), ipp32f, func_cfg, &buffer_size);

    if (status != ippStsNoErr)
    {
        throw std::runtime_error("ippsAutoCorrNormGetBufferSize failed with error code " + std::to_string(status));
    }

    work_buffer = static_cast<Ipp8u*>(ippMalloc(buffer_size));
    if (work_buffer == nullptr)
    {
        throw std::runtime_error("Failed to allocate IPP work buffer");
    }

    std::vector<float> out(signal.size(), 0.0f);
    status = ippsAutoCorrNorm_32f(signal.data(), signal.size(), out.data(), out.size(), func_cfg, work_buffer);
    ippFree(work_buffer);

    if (status != ippStsNoErr)
    {
        throw std::runtime_error("ippsAutoCorrNorm_32f failed with error code " + std::to_string(status));
    }
#endif

    if (normalize)
    {
        float zero_lag = out[0];
        array_math::Divide(out, zero_lag, out);
    }

    return out;
}

std::span<float> TrimSilence(std::span<float> signal, float threshold)
{
    if (signal.empty())
    {
        return {};
    }

    // Discard silence at the beginning of impulse response
    const float max_val = array_math::MaxAbs(signal);

    const float target = threshold * std::abs(max_val);

    for (auto i = 0u; i < signal.size(); ++i)
    {
        if (std::abs(signal[i]) >= target)
        {
            return signal.subspan(i);
        }
    }
    return signal;
    // auto it_start = std::ranges::find_if(
    //     signal, [threshold = threshold * std::abs(max_val)](float sample) { return std::abs(sample) >= threshold; });

    // if (it_start != signal.end())
    // {
    //     return signal.subspan(std::distance(signal.begin(), it_start));
    // }
    // else
    // {
    //     return signal;
    // }
}

std::vector<float> EnergyDecayCurve(std::span<const float> signal, bool to_db)
{
    if (signal.empty())
    {
        return {};
    }

    std::ranges::reverse_view trimmed_signal_reversed{signal};
    auto s = trimmed_signal_reversed | std::views::transform([](float x) { return x * x; });

    // Calculate the energy decay curve
    std::vector<float> decay_curve(signal.size(), 0.0f);

    std::ranges::reverse_view decay_curve_reversed{decay_curve};

    std::partial_sum(s.begin(), s.end(), decay_curve_reversed.begin());

    if (to_db)
    {
        array_math::ToDb(decay_curve, 10.0f);
    }

    return decay_curve;
}

std::vector<float> Convolve(std::span<const float> signal, std::span<const float> kernel)
{
    const uint32_t conv_size = signal.size() + kernel.size() - 1;
    const uint32_t fft_size = FFT::NextSupportedFFTSize(static_cast<uint32_t>(conv_size));

    FFT fft(fft_size);
    std::vector<float> result(fft_size, 0.0f);
    fft.Convolve(signal, kernel, result);

    // Depending on the FFT implementation, the result may be larger than what would usually be expected from
    // a convolution. Trim to the expected size.
    result.resize(conv_size);
    return result;
}

float SpectralFlatness(std::span<const float> power_spectrum)
{
    float geo_mean = 1.0f;
    float arith_mean = 0.0f;

#ifndef AUDIO_UTILS_USE_IPP
    for (const auto& power : power_spectrum)
    {
        geo_mean += std::log(power + std::numeric_limits<float>::epsilon());
        arith_mean += power;
    }

    geo_mean = std::exp(geo_mean / static_cast<float>(power_spectrum.size()));
    arith_mean /= static_cast<float>(power_spectrum.size());
#else
    std::vector<float> log_spectrum(power_spectrum.size(), 0.0f);
    array_math::Ln(power_spectrum, log_spectrum);
    geo_mean = std::exp(array_math::Mean(log_spectrum));
    arith_mean = array_math::Mean(power_spectrum);
#endif

    if (arith_mean == 0.0f)
    {
        return 0.0f;
    }

    return geo_mean / arith_mean;
}

STFTResult STFT(std::span<const float> signal, STFTOptions& options, bool flip)
{
    ValidateSTFTOptions(options);

    const uint32_t hop = options.window_size - options.overlap;
    const uint32_t num_frames = (signal.size() - options.overlap) / hop;
    const uint32_t num_bins = options.fft_size / 2 + 1;

    std::vector<float> result;
    result.resize(num_frames * num_bins, -9999.999f);

    std::vector<float> window(options.window_size);
    GetWindow(options.window_type, window);

    FFT fft(options.fft_size);

    auto spec_span = std::span(result);

    std::vector<float> frame(options.window_size);

    for (auto i = 0; i < num_frames; ++i)
    {
        auto signal_span = signal.subspan(i * hop, options.window_size);

        array_math::Multiply(signal_span, window, frame);

        auto spectrum_subspan = spec_span.subspan(i * num_bins, num_bins);
        fft.ForwardMag(frame, spectrum_subspan, {FFTOutputType::Magnitude, false});
    }

    if (flip)
    {
        Eigen::Map<Eigen::MatrixXf> spec_map(result.data(), num_bins, num_frames);
        spec_map.colwise().reverseInPlace();
    }

    STFTResult result_struct;
    result_struct.data = std::move(result);
    result_struct.num_bins = num_bins;
    result_struct.num_frames = num_frames;

    return result_struct;
}

#if 1
STFTResult MelSpectrogram(std::span<const float> signal, STFTOptions& options, size_t n_mels, bool flip)
{
    ValidateSTFTOptions(options);

    if (options.samplerate == 0)
    {
        throw std::invalid_argument("Samplerate must be set in STFTOptions for MelSpectrogram computation");
    }

    STFTResult result = STFT(signal, options, false);

    Eigen::MatrixXf mel_weights = GetMelFilter(n_mels, options.fft_size, options.samplerate);

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
#else

STFTResult MelSpectrogram(std::span<const float> signal, STFTOptions& options, size_t n_mels, bool flip)
{
    const Eigen::MatrixXf mel_weights = GetMelFilter(n_mels, options.fft_size, options.samplerate);

    if (options.overlap >= options.fft_size)
    {
        throw std::invalid_argument("Overlap must be less than FFT size");
    }

    if (options.overlap >= options.window_size)
    {
        throw std::invalid_argument("Overlap must be less than window size");
    }

    const uint32_t hop = options.window_size - options.overlap;
    const uint32_t num_frames = (signal.size() - options.overlap) / hop;

    std::vector<float> result;
    result.resize(num_frames * n_mels, -9999.999f);

    std::vector<float> window(options.window_size);
    GetWindow(options.window_type, window);

    FFT fft(options.fft_size);

    auto spec_span = std::span(result);

    std::vector<float> frame(options.window_size);
    std::vector<float> spectrum(fft.GetSpectrumSize());

    for (auto i = 0; i < num_frames; ++i)
    {
        auto signal_span = signal.subspan(i * hop, options.window_size);
        array_math::Multiply(signal_span, window, frame);

        fft.ForwardMag(frame, spectrum, {FFTOutputType::Magnitude, false});
        auto spectrum_subspan = spec_span.subspan(i * n_mels, n_mels);
        Eigen::Map<Eigen::VectorXf> mel_spectrum(spectrum_subspan.data(), spectrum_subspan.size());
        Eigen::Map<const Eigen::VectorXf> spec_vec(spectrum.data(), fft.GetSpectrumSize());
        mel_spectrum = mel_weights * spec_vec;
    }

    if (flip)
    {
        Eigen::Map<Eigen::MatrixXf> spec_map(result.data(), n_mels, num_frames);
        spec_map.colwise().reverseInPlace();
    }

    STFTResult result_struct;
    result_struct.data = std::move(result);
    result_struct.num_bins = n_mels;
    result_struct.num_frames = num_frames;

    return result_struct;
}
#endif
} // namespace audio_utils::analysis