#include "fft_utils.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <numbers>
#include <string>

#include <pffft.h>

namespace
{
constexpr float k_two_pi = 2.f * std::numbers::pi_v<float>;
}

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

void fft(const float* in, float* out, size_t count)
{
    const size_t kSize = count * sizeof(float);
    float* aligned_in = static_cast<float*>(pffft_aligned_malloc(kSize));
    float* aligned_out = static_cast<float*>(pffft_aligned_malloc(kSize));

    memcpy(static_cast<void*>(aligned_in), static_cast<const void*>(in), kSize);

    PFFFT_Setup* setup = pffft_new_setup(count, PFFFT_REAL);
    pffft_transform_ordered(setup, aligned_in, aligned_out, nullptr, PFFFT_FORWARD);
    pffft_destroy_setup(setup);

    memcpy(static_cast<void*>(out), static_cast<void*>(aligned_out), kSize);

    pffft_aligned_free(aligned_in);
    pffft_aligned_free(aligned_out);
}

void fft_abs(const float* in, float* out, size_t count)
{
    float dc = in[0];
    float nyquist = in[1];

    out[0] = std::abs(dc);
    for (size_t i = 1; i < count / 2; ++i)
    {
        auto real = in[2 * i];
        auto complex = in[(2 * i) + 1];
        out[i] = std::hypotf(real, complex);
    }

    out[count / 2] = std::abs(nyquist);
}

std::vector<float> spectrogram(const float* in, size_t count, const spectrogram_info& info)
{
    if (info.overlap >= info.fft_size)
    {
        throw std::invalid_argument("Overlap must be less than FFT size");
    }

    std::vector<float> result;

    const float input_duration = static_cast<float>(count) / info.samplerate;
    const size_t hop = info.fft_size - info.overlap;
    const size_t num_bins = (count - info.fft_size) / hop;
    const size_t num_freqs = info.fft_size / 2 + 1;
    result.reserve(num_bins * num_freqs);

    std::vector<float> window(info.fft_size);
    get_window(info.window_type, window.data(), info.fft_size);

    const size_t kSize = count * sizeof(float);
    float* aligned_in = static_cast<float*>(pffft_aligned_malloc(kSize));
    float* aligned_out = static_cast<float*>(pffft_aligned_malloc(kSize));

    PFFFT_Setup* setup = pffft_new_setup(info.fft_size, PFFFT_REAL);
    for (size_t idx = 0; idx < (count - info.fft_size); idx += hop)
    {
        size_t i = 0;
        for (i = 0; i < info.fft_size - 4; i += 4)
        {
            aligned_in[i] = in[idx + i] * window[i];
            aligned_in[i + 1] = in[idx + i + 1] * window[i + 1];
            aligned_in[i + 2] = in[idx + i + 2] * window[i + 2];
            aligned_in[i + 3] = in[idx + i + 3] * window[i + 3];
        }
        // Finish the loop
        for (; i < info.fft_size; ++i)
        {
            aligned_in[i] = in[idx + i] * window[i];
        }

        pffft_transform_ordered(setup, aligned_in, aligned_out, nullptr, PFFFT_FORWARD);
        fft_abs(aligned_out, aligned_out, info.fft_size);

        result.insert(result.end(), aligned_out, aligned_out + num_freqs);
    }
    pffft_destroy_setup(setup);

    return result;
}