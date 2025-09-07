#include "audio_utils/audio_analysis.h"

#include <iostream>
#include <map>
#include <vector>

#include "audio_utils/fft_utils.h"
#include <Eigen/Core>
#include <fftw3.h>

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
struct FFTWPlanInfo
{
    fftwf_plan plan;
    float* signal;
    fftwf_complex* spectrum;
};

class FFTWCache
{
  public:
    static FFTWCache& Instance()
    {
        static FFTWCache instance;
        return instance;
    }

    FFTWPlanInfo GetForwardPlan(size_t nfft)
    {
        auto it = forward_plans_.find(nfft);
        if (it != forward_plans_.end())
        {
            return it->second;
        }

        FFTWPlanInfo plan_info{};
        plan_info.signal = static_cast<float*>(fftwf_malloc(nfft * sizeof(float)));
        plan_info.spectrum = static_cast<fftwf_complex*>(fftwf_malloc((nfft / 2 + 1) * sizeof(fftwf_complex)));

        plan_info.plan = fftwf_plan_dft_r2c_1d(nfft, plan_info.signal, plan_info.spectrum, FFTW_MEASURE);
        forward_plans_[nfft] = plan_info;
        return plan_info;
    }

    FFTWPlanInfo GetBackwardPlan(size_t nfft)
    {
        auto it = backward_plans_.find(nfft);
        if (it != backward_plans_.end())
        {
            return it->second;
        }

        FFTWPlanInfo plan_info{};
        plan_info.signal = static_cast<float*>(fftwf_malloc(nfft * sizeof(float)));
        plan_info.spectrum = static_cast<fftwf_complex*>(fftwf_malloc((nfft / 2 + 1) * sizeof(fftwf_complex)));
        plan_info.plan = fftwf_plan_dft_c2r_1d(nfft, plan_info.spectrum, plan_info.signal, FFTW_MEASURE);
        backward_plans_[nfft] = plan_info;
        return plan_info;
    }

  private:
    FFTWCache() = default;

    ~FFTWCache()
    {
        // fftwf_cleanup_threads();

        for (auto& pair : forward_plans_)
        {
            fftwf_destroy_plan(pair.second.plan);
            fftwf_free(pair.second.signal);
            fftwf_free(pair.second.spectrum);
        }
        for (auto& pair : backward_plans_)
        {
            fftwf_destroy_plan(pair.second.plan);
            fftwf_free(pair.second.signal);
            fftwf_free(pair.second.spectrum);
        }
    }

    std::map<size_t, FFTWPlanInfo> forward_plans_;
    std::map<size_t, FFTWPlanInfo> backward_plans_;
};
} // namespace

namespace audio_utils::analysis
{

std::vector<float> Autocorrelation(std::span<const float> signal, bool normalize)
{
    const size_t kNFFT = 2 * signal.size();
    const size_t kInSize = kNFFT * sizeof(float);
    const size_t kOutSize = (kNFFT / 2 + 1) * sizeof(fftwf_complex);
    float* aligned_in = static_cast<float*>(fftwf_malloc(kInSize));
    fftwf_complex* aligned_out = static_cast<fftwf_complex*>(fftwf_malloc(kOutSize));

    fftwf_plan forward_plan = fftwf_plan_dft_r2c_1d(kNFFT, aligned_in, aligned_out, FFTW_ESTIMATE);
    fftwf_plan backward_plan = fftwf_plan_dft_c2r_1d(kNFFT, aligned_out, aligned_in, FFTW_ESTIMATE);

    std::fill(aligned_in, aligned_in + kNFFT, 0.0f);
    // std::fill(aligned_out, aligned_out + kNFFT, 0.0f);
    std::copy(signal.begin(), signal.end(), aligned_in);

    fftwf_execute(forward_plan);

    for (size_t i = 0; i < (kNFFT / 2 + 1); ++i)
    {
        std::complex<float> c(aligned_out[i][0], aligned_out[i][1]);
        aligned_out[i][0] = std::pow(std::abs(c), 2);
        aligned_out[i][1] = 0.0f; // Set imaginary parts to zero
    }

    fftwf_execute(backward_plan);

    std::vector<float> out(signal.size());
    for (size_t i = 0; i < signal.size(); ++i)
    {
        out[i] = aligned_in[i] / static_cast<float>(kNFFT);
    }

    if (normalize)
    {
        const float coeff = out[0];
        for (size_t i = 0; i < signal.size(); ++i)
        {
            out[i] /= coeff; // Normalize so the the first value is 1.0
        }
    }

    fftwf_free(aligned_in);
    fftwf_free(aligned_out);
    fftwf_destroy_plan(forward_plan);
    fftwf_destroy_plan(backward_plan);
    return out;
}

std::vector<float> Spectrogram(std::span<const float> signal, SpectrogramInfo& info)
{
    if (info.overlap >= info.fft_size)
    {
        throw std::invalid_argument("Overlap must be less than FFT size");
    }

    std::vector<float> result;

    const float input_duration = static_cast<float>(signal.size()) / info.samplerate;
    const size_t hop = info.window_size - info.overlap;
    info.num_frames = (signal.size() - info.window_size) / hop + 1;
    info.num_freqs = info.fft_size / 2 + 1;

    result.resize(info.num_frames * info.num_freqs, -50.0f); // Initialize with -50 dB

    std::vector<float> window(info.window_size);
    GetWindow(info.window_type, window.data(), info.window_size);

    FFTWPlanInfo plan_info = FFTWCache::Instance().GetForwardPlan(info.fft_size);

    std::fill(plan_info.signal, plan_info.signal + info.fft_size, 0.0f);

    size_t idx = 0;
    for (size_t b = 0; b < info.num_frames; ++b)
    {
        auto signal_span = signal.subspan(idx);
        if (signal_span.size() < info.window_size)
        {
            std::cerr << "Warning: Not enough input data for the last frame. Padding with zeros." << std::endl;
            break;
        }

        signal_span = signal_span.first(info.window_size);

        for (size_t i = 0; i < signal_span.size(); ++i)
        {
            plan_info.signal[i] = signal_span[i] * window[i];
        }

        for (size_t i = info.window_size; i < info.fft_size; ++i)
        {
            plan_info.signal[i] = 0.0f; // Zero-pad the rest of the input
        }

        fftwf_execute(plan_info.plan);

        for (int f = 0; f < info.num_freqs; ++f)
        {
            std::complex<float> val = reinterpret_cast<std::complex<float>*>(plan_info.spectrum)[f];
            result.at(f * info.num_frames + b) = 20.0f * std::log10f(std::abs(val));
            if (std::isinf(result.at(f * info.num_frames + b)) || std::isnan(result.at(f * info.num_frames + b)))
            {
                result.at(f * info.num_frames + b) = -50.0f; // Set to -50 dB if log10 is invalid
            }
        }

        idx += hop;
    }

    return result;
}

std::vector<float> MelSpectrogram(std::span<const float> signal, SpectrogramInfo& info, size_t n_mels)
{
    if (info.overlap >= info.fft_size)
    {
        throw std::invalid_argument("Overlap must be less than FFT size");
    }

    std::vector<float> result;

    Eigen::MatrixXf mel_weights = GetMelFilter(n_mels, info.fft_size, info.samplerate);

    const float input_duration = static_cast<float>(signal.size()) / info.samplerate;
    const size_t hop = info.window_size - info.overlap;
    info.num_frames = (signal.size() - info.window_size) / hop + 1;
    info.num_freqs = info.fft_size / 2 + 1;

    result.resize(info.num_frames * n_mels, -50.0f); // Initialize with -50 dB

    Eigen::Map<Eigen::MatrixXf> result_map(result.data(), info.num_frames, n_mels);

    std::vector<float> window(info.window_size);
    GetWindow(info.window_type, window.data(), info.window_size);

    FFTWPlanInfo plan_info = FFTWCache::Instance().GetForwardPlan(info.fft_size);
    float* work_buffer = static_cast<float*>(fftwf_malloc(info.num_freqs * sizeof(float)));

    std::fill(plan_info.signal, plan_info.signal + info.fft_size, 0.0f);

    size_t idx = 0;
    for (size_t b = 0; b < info.num_frames; ++b)
    {
        auto signal_span = signal.subspan(idx);
        if (signal_span.size() < info.window_size)
        {
            std::cerr << "Warning: Not enough input data for the last frame. Padding with zeros." << std::endl;
            break;
        }

        signal_span = signal_span.first(info.window_size);

        for (size_t i = 0; i < signal_span.size(); ++i)
        {
            plan_info.signal[i] = signal_span[i] * window[i];
        }

        for (size_t i = info.window_size; i < info.fft_size; ++i)
        {
            plan_info.signal[i] = 0.0f; // Zero-pad the rest of the input
        }

        fftwf_execute(plan_info.plan);

        for (size_t i = 0; i < info.num_freqs; ++i)
        {
            work_buffer[i] = std::abs(reinterpret_cast<std::complex<float>*>(plan_info.spectrum)[i]);
        }

        Eigen::Map<Eigen::VectorXf> aligned_out_map(work_buffer, info.num_freqs);

        Eigen::VectorXf mel_spectrum = mel_weights * aligned_out_map;

        mel_spectrum = 20.f * mel_spectrum.array().log10();
        mel_spectrum = mel_spectrum.array().isInf().select(-60.f, mel_spectrum.array());
        result_map.row(b) = mel_spectrum.reverse();

        idx += hop;
    }

    fftwf_free(work_buffer);

    info.num_freqs = n_mels;

    return result;
}

} // namespace audio_utils::analysis