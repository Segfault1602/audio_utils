#include "fft_utils.h"

#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <map>
#include <numbers>
#include <string>

#include <Eigen/Core>
// #include <pffft.h>
#include <fftw3.h>

namespace
{
constexpr float k_two_pi = 2.f * std::numbers::pi_v<float>;

size_t next_power_of_two(size_t n)
{
    n--;
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    n |= (n >> 32);
    return ++n;
}

Eigen::ArrayXf FftFrequencies(size_t nfft, size_t sample_rate)
{
    const size_t N = nfft / 2 + 1;
    Eigen::ArrayXf frequencies = Eigen::ArrayXf::LinSpaced(N, 0.f, N - 1);
    float val = sample_rate / static_cast<float>(nfft);
    return frequencies * val;
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

    float mel = change_point / lin_step + std::log(hz / change_point) / logstep;

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

Eigen::ArrayXf MelFrequencies(size_t n_mels, float f_min, float f_max)
{
    float min_mel = HzToMel(f_min);
    float max_mel = HzToMel(f_max);

    Eigen::ArrayXf mel_frequencies = Eigen::ArrayXf::LinSpaced(n_mels, min_mel, max_mel);
    return mel_frequencies.unaryExpr([](float mel) { return MelToHz(mel); });
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

        FFTWPlanInfo plan_info;
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

        FFTWPlanInfo plan_info;
        plan_info.signal = static_cast<float*>(fftwf_malloc(nfft * sizeof(float)));
        plan_info.spectrum = static_cast<fftwf_complex*>(fftwf_malloc((nfft / 2 + 1) * sizeof(fftwf_complex)));
        plan_info.plan = fftwf_plan_dft_c2r_1d(nfft, plan_info.spectrum, plan_info.signal, FFTW_MEASURE);
        backward_plans_[nfft] = plan_info;
        return plan_info;
    }

  private:
    FFTWCache()
    {
        // fftwf_init_threads();
    }

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

void get_window(FFTWindowType type, float* window, size_t count)
{
    assert(window != nullptr);

    switch (type)
    {
    case FFTWindowType::Rectangular:
        for (size_t i = 0; i < count; ++i)
        {
            window[i] = 1.0f;
        }
        break;
    case FFTWindowType::Hamming:
    {
        constexpr float alpha = 0.54f;
        constexpr float beta = 1.0f - alpha;
        for (size_t i = 0; i < count; ++i)
        {
            window[i] = alpha - beta * std::cos(k_two_pi * i / (count - 1));
        }
        break;
    }
    case FFTWindowType::Hann:
    {
        for (size_t i = 0; i < count; ++i)
        {
            window[i] = 0.5f * (1.0f - std::cos(k_two_pi * i / (count - 1)));
        }
        break;
    }
    case FFTWindowType::Blackman:
    {
        for (size_t i = 0; i < count; ++i)
        {
            window[i] =
                0.42f - 0.5 * std::cos(k_two_pi * i / (count - 1)) + 0.08f * std::cos(2 * k_two_pi * i / (count - 1));
        }
        break;
    }
    }
}
} // namespace

std::vector<std::complex<float>> FFT(std::span<const float> in, size_t nfft)
{
    if (nfft == 0)
    {
        nfft = in.size();
    }
    else if (nfft < in.size())
    {
        throw std::invalid_argument("nfft must be greater than or equal to the size of input data");
    }

    FFTWPlanInfo plan_info = FFTWCache::Instance().GetForwardPlan(nfft);

    std::copy(in.begin(), in.end(), plan_info.signal);
    std::fill(plan_info.signal + in.size(), plan_info.signal + nfft, 0.0f); // Zero-pad if necessary

    fftwf_execute(plan_info.plan);

    std::vector<std::complex<float>> out(nfft / 2 + 1, std::complex<float>(0.0f, 0.0f));

    std::complex<float>* spectrum_cast =
        reinterpret_cast<std::complex<float>*>(plan_info.spectrum); // Cast to complex<float> pointer for easier access
    std::copy(spectrum_cast, spectrum_cast + nfft / 2 + 1, out.begin());

    return out;
}

std::vector<float> AbsFFT(std::span<const float> in, size_t nfft, bool db, bool normalize)
{
    if (nfft == 0)
    {
        nfft = in.size();
    }
    else if (nfft < in.size())
    {
        throw std::invalid_argument("nfft must be greater than or equal to the size of input data");
    }

    FFTWPlanInfo plan_info = FFTWCache::Instance().GetForwardPlan(nfft);

    std::copy(in.begin(), in.end(), plan_info.signal);
    std::fill(plan_info.signal + in.size(), plan_info.signal + nfft, 0.0f); // Zero-pad if necessary

    fftwf_execute(plan_info.plan);

    std::vector<float> out(nfft / 2 + 1, 0.f);

    for (size_t i = 0; i < nfft / 2 + 1; ++i)
    {
        out[i] = std::abs(reinterpret_cast<std::complex<float>*>(plan_info.spectrum)[i]);
    }

    if (normalize)
    {
        float max_val = *std::max_element(out.begin(), out.end());
        std::transform(out.begin(), out.end(), out.begin(), [max_val](float val) { return val / max_val; });
    }

    if (db)
    {
        std::transform(out.begin(), out.end(), out.begin(), [](float val) { return 20.0f * std::log10f(val); });
    }

    return out;
}

std::vector<float> AbsCepstrum(std::span<const float> in, size_t nfft)
{
    if (nfft == 0)
    {
        nfft = in.size();
    }
    else if (nfft < in.size())
    {
        throw std::invalid_argument("nfft must be greater than or equal to the size of input data");
    }

    FFTWPlanInfo plan_info = FFTWCache::Instance().GetForwardPlan(nfft);

    std::copy(in.begin(), in.end(), plan_info.signal);
    std::fill(plan_info.signal + in.size(), plan_info.signal + nfft, 0.0f); // Zero-pad if necessary

    fftwf_execute(plan_info.plan);

    FFTWPlanInfo backward_plan_info = FFTWCache::Instance().GetBackwardPlan(nfft);

    size_t num_elements = nfft / 2 + 1;
    for (size_t i = 0; i < num_elements; ++i)
    {
        std::complex<float> val = reinterpret_cast<std::complex<float>*>(plan_info.spectrum)[i];
        backward_plan_info.spectrum[i][0] = std::log(std::abs(val));
        backward_plan_info.spectrum[i][1] = 0.0f; // Set imaginary part to zero
    }

    fftwf_execute(backward_plan_info.plan);

    // The cepstrum is symmetric, so we only need the first half
    std::vector<float> out(nfft / 2, 0.f);
    std::copy(backward_plan_info.signal, backward_plan_info.signal + nfft / 2, out.begin());

    // normalize the output
    float coeff = out[0];
    for (size_t i = 0; i < out.size(); ++i)
    {
        out[i] /= coeff;
    }

    return out;
}

std::vector<float> Spectrogram(const float* in, size_t count, spectrogram_info& info)
{
    if (info.overlap >= info.fft_size)
    {
        throw std::invalid_argument("Overlap must be less than FFT size");
    }

    std::vector<float> result;

    const float input_duration = static_cast<float>(count) / info.samplerate;
    const size_t hop = info.window_size - info.overlap;
    info.num_frames = (count - info.window_size) / hop + 1;
    info.num_freqs = info.fft_size / 2 + 1;

    result.resize(info.num_frames * info.num_freqs, -50.0f); // Initialize with -50 dB

    std::vector<float> window(info.window_size);
    get_window(info.window_type, window.data(), info.window_size);

    FFTWPlanInfo plan_info = FFTWCache::Instance().GetForwardPlan(info.fft_size);

    std::fill(plan_info.signal, plan_info.signal + info.fft_size, 0.0f);

    size_t idx = 0;
    for (size_t b = 0; b < info.num_frames; ++b)
    {
        if (idx + info.window_size >= count)
        {
            std::cerr << "Warning: Not enough input data for the last frame. Padding with zeros." << std::endl;
            break;
        }
        for (size_t i = 0; i < info.window_size; ++i)
        {
            plan_info.signal[i] = in[idx + i] * window[i];
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

std::vector<float> MelSpectrogram(const float* in, size_t count, spectrogram_info& info, size_t n_mels)
{
    if (info.overlap >= info.fft_size)
    {
        throw std::invalid_argument("Overlap must be less than FFT size");
    }

    std::vector<float> result;

    Eigen::MatrixXf mel_weights = GetMelFilter(n_mels, info.fft_size, info.samplerate);

    const float input_duration = static_cast<float>(count) / info.samplerate;
    const size_t hop = info.window_size - info.overlap;
    info.num_frames = (count - info.window_size) / hop + 1;
    info.num_freqs = info.fft_size / 2 + 1;

    result.resize(info.num_frames * n_mels, -50.0f); // Initialize with -50 dB

    std::vector<float> window(info.window_size);
    get_window(info.window_type, window.data(), info.window_size);

    FFTWPlanInfo plan_info = FFTWCache::Instance().GetForwardPlan(info.fft_size);
    float* work_buffer = static_cast<float*>(fftwf_malloc(info.num_freqs * sizeof(float)));

    std::fill(plan_info.signal, plan_info.signal + info.fft_size, 0.0f);

    size_t idx = 0;
    for (size_t b = 0; b < info.num_frames; ++b)
    {
        if (idx + info.window_size >= count)
        {
            std::cerr << "Warning: Not enough input data for the last frame. Padding with zeros." << std::endl;
            break;
        }

        for (size_t i = 0; i < info.window_size; ++i)
        {
            plan_info.signal[i] = in[idx + i] * window[i];
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

        for (int f = 0; f < n_mels; ++f)
        {
            // result.at(f * info.num_frames + b) = (mel_spectrum[n_mels - 1 - f]);
            result.at(f * info.num_frames + b) = 20.0f * std::log10f(mel_spectrum[n_mels - 1 - f]);
            if (std::isinf(result.at(f * info.num_frames + b)) || std::isnan(result.at(f * info.num_frames + b)))
            {
                result.at(f * info.num_frames + b) = -50.0f; // Set to -50 dB if log10 is invalid
            }
        }

        idx += hop;
    }

    fftwf_free(work_buffer);

    info.num_freqs = n_mels;

    return result;
}

std::vector<float> GetMelFrequencies(size_t n_mels, float f_min, float f_max)
{
    std::vector<float> mel_frequencies(n_mels);
    Eigen::ArrayXf mel_freqs = MelFrequencies(n_mels, f_min, f_max);
    for (size_t i = 0; i < n_mels; ++i)
    {
        mel_frequencies[i] = mel_freqs[i];
    }
    return mel_frequencies;
}
