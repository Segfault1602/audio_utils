#include "audio_utils/fft_utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <map>
#include <numbers>

#include <Eigen/Core>

namespace
{
constexpr float k_two_pi = 2.f * std::numbers::pi_v<float>;

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

} // namespace

namespace audio_utils
{

void GetWindow(FFTWindowType type, std::span<float> window)
{
    assert(!window.empty());

    switch (type)
    {
    case FFTWindowType::Rectangular:
        std::ranges::fill(window, 1.0f);
        break;
    case FFTWindowType::Hamming:
    {
        constexpr float alpha = 0.54f;
        constexpr float beta = 1.0f - alpha;
        for (size_t i = 0; i < window.size(); ++i)
        {
            window[i] = alpha - beta * std::cos(k_two_pi * i / (window.size() - 1));
        }
        break;
    }
    case FFTWindowType::Hann:
    {
        for (size_t i = 0; i < window.size(); ++i)
        {
            window[i] = 0.5f * (1.0f - std::cos(k_two_pi * i / (window.size() - 1)));
        }
        break;
    }
    case FFTWindowType::Blackman:
    {
        for (size_t i = 0; i < window.size(); ++i)
        {
            window[i] = 0.42f - 0.5 * std::cos(k_two_pi * i / (window.size() - 1)) +
                        0.08f * std::cos(2 * k_two_pi * i / (window.size() - 1));
        }
        break;
    }
    }
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
} // namespace audio_utils