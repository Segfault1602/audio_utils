#pragma once

#include <span>
#include <vector>

#include "audio_utils/fft_utils.h"

namespace audio_utils::analysis
{
struct SpectrogramInfo
{
    int fft_size;
    int overlap;
    int samplerate;
    int window_size;
    FFTWindowType window_type;

    int num_freqs;  // Filled by the function
    int num_frames; // Filled by the function
};

/**
 * @brief Computes the autocorrelation of a signal.
 * @param signal The input signal as a span of floats.
 * @return A vector containing the autocorrelation values. Only the positive lags are returned.
 */
std::vector<float> Autocorrelation(std::span<const float> signal, bool normalize = true);

/**
 * @brief Computes the spectrogram of a signal.
 * @param signal The input signal as a span of floats.
 * @param info Spectrogram information including FFT size, window size, overlap, and sample rate.
 * @return A vector containing the spectrogram values in column-major order.
 */
std::vector<float> Spectrogram(std::span<const float> signal, SpectrogramInfo& info);

/**
 * @brief Computes the Mel spectrogram of a signal.
 * @param signal The input signal as a span of floats.
 * @param info Spectrogram information including FFT size, window size, overlap, and sample rate.
 * @param n_mels The number of Mel frequency bands.
 * @return A vector containing the Mel spectrogram values in column-major order.
 */
std::vector<float> MelSpectrogram(std::span<const float> signal, SpectrogramInfo& info, size_t n_mels);

} // namespace audio_utils::analysis
