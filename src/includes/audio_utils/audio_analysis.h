#pragma once

#include <span>
#include <vector>

#include "audio_utils/fft_utils.h"

namespace audio_utils::analysis
{
struct SpectrogramInfo
{
    uint32_t fft_size;
    uint32_t overlap;
    uint32_t window_size;
    uint32_t samplerate;
    FFTWindowType window_type;
};

struct SpectrogramResult
{
    std::vector<float> data; // Spectrogram data in column-major order
    int num_bins;            // Number of frequency bins
    int num_frames;          // Number of time frames
};

/**
 * @brief Computes the autocorrelation of a signal.
 * @param signal The input signal as a span of floats.
 * @return A vector containing the autocorrelation values. Only the positive lags are returned.
 */
std::vector<float> Autocorrelation(std::span<const float> signal, bool normalize = true);

SpectrogramResult STFT(std::span<const float> signal, SpectrogramInfo& info, bool flip = false);

/**
 * @brief Computes the Mel spectrogram of a signal.
 * @param signal The input signal as a span of floats.
 * @param info Spectrogram information including FFT size, window size, overlap, and sample rate.
 * @param n_mels The number of Mel frequency bands.
 * @return A vector containing the Mel spectrogram values in column-major order.
 */
SpectrogramResult MelSpectrogram(std::span<const float> signal, SpectrogramInfo& info, size_t n_mels,
                                 bool flip = false);

} // namespace audio_utils::analysis
