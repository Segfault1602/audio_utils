#pragma once

#include <cstddef>
#include <vector>

enum class FFTWindowType
{
    Rectangular,
    Hamming,
    Hann,
    Blackman
};

struct spectrogram_info
{
    int fft_size;
    int overlap;
    int samplerate;
    FFTWindowType window_type;
};

void get_window(FFTWindowType type, float* window, size_t count);

void fft(const float* in, float* out, size_t count);

void fft_abs(const float* in, float* out, size_t count);

std::vector<float> spectrogram(const float* in, size_t count, const spectrogram_info& info);