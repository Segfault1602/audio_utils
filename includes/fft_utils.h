#pragma once

#include <complex>
#include <cstddef>
#include <span>
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
    int window_size;
    FFTWindowType window_type;

    int num_freqs;  // Filled by the function
    int num_frames; // Filled by the function
};

std::vector<std::complex<float>> FFT(std::span<const float> in, size_t nfft = 0);
std::vector<float> AbsFFT(std::span<const float> in, size_t nfft = 0, bool db = false, bool normalize = false);

std::vector<float> AbsCepstrum(std::span<const float> in, size_t nfft = 0);

std::vector<float> Spectrogram(const float* in, size_t count, spectrogram_info& info);

std::vector<float> MelSpectrogram(const float* in, size_t count, spectrogram_info& info, size_t n_mels);

std::vector<float> GetMelFrequencies(size_t n_mels, float f_min, float f_max);
