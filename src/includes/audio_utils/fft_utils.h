#pragma once

#include <complex>
#include <cstddef>
#include <span>
#include <vector>

namespace audio_utils
{

enum class FFTWindowType
{
    Rectangular,
    Hamming,
    Hann,
    Blackman
};

void GetWindow(FFTWindowType type, std::span<float> window);

std::vector<float> GetMelFrequencies(size_t n_mels, float f_min, float f_max);
} // namespace audio_utils
