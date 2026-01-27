#pragma once

#include <span>
#include <vector>

#include "audio_utils/fft_utils.h"

namespace audio_utils::analysis
{
struct STFTOptions
{
    uint32_t fft_size;
    uint32_t overlap;
    uint32_t window_size;
    FFTWindowType window_type;

    uint32_t samplerate; // Only used for Mel spectrograms
};

/**
 * @brief Struct to hold the result of a STFT computation.
 *
 * The data is stored in column-major order, meaning that consecutive frequency bins for a given time frame are stored
 * consecutively in memory.
 *
 * Ex.: [a_0 b_0 c_0 ... a_1 b_1 c_1 ...] where a, b, c are frequency bins and 0, 1 are time frames.
 *
 */
struct STFTResult
{
    std::vector<float> data; // STFT data in column-major order
    int num_bins;            // Number of frequency bins
    int num_frames;          // Number of time frames
};

/**
 * @brief Computes the autocorrelation of a signal.
 * @param signal The input signal as a span of floats.
 * @param normalize If true, the autocorrelation is normalized by the zero-lag value.
 * @return A vector containing the autocorrelation values. Only the positive lags are returned.
 */
std::vector<float> Autocorrelation(std::span<const float> signal, bool normalize = true);

std::span<float> TrimSilence(std::span<float> signal, float threshold);

/**
 * @brief Compute the energy decay curve of a signal.
 *
 * @param signal The input signal.
 * @param to_db If true, convert the energy values to decibels.
 * @return std::vector<float> The short-time energy of the signal.
 */
std::vector<float> EnergyDecayCurve(std::span<const float> signal, bool to_db = false);

/**
 * @brief Convolves a signal with a given kernel.
 *
 * @param signal The input signal.
 * @param kernel The convolution kernel.
 * @return std::vector<float> The convolved signal of size (signal.size() + kernel.size() - 1).
 *
 * @note This function uses FFT-based convolution for efficiency.
 */
std::vector<float> Convolve(std::span<const float> signal, std::span<const float> kernel);

/**
 * @brief Computes the spectral flatness of a power spectrum.
 * @param power_spectrum The input power spectrum as a span of floats.
 * @return The spectral flatness value.
 */
float SpectralFlatness(std::span<const float> power_spectrum);

/**
 * @brief Computes the Short-Time Fourier Transform (STFT) of a signal.
 * @param signal The input signal as a span of floats.
 * @param info Spectrogram information including FFT size, window size, overlap, and sample rate.
 * @param flip If true, the frequency bins in the output spectrogram are reversed such that the Nyquist frequency is
 * first. Useful for certain visualization purposes, ex. ImPlot::PlotHeatmap.
 *
 * @return STFTResult The computed spectrogram result.
 */
STFTResult STFT(std::span<const float> signal, STFTOptions& info, bool flip = false);

/**
 * @brief Computes the Mel spectrogram of a signal.
 * @param signal The input signal as a span of floats.
 * @param info Spectrogram information including FFT size, window size, overlap, and sample rate.
 * @param n_mels The number of Mel frequency bands.
 *
 * @param flip If true, the Mel frequency bins in the output spectrogram are reversed such that the highest Mel
 */
STFTResult MelSpectrogram(std::span<const float> signal, STFTOptions& info, size_t n_mels, bool flip = false);

} // namespace audio_utils::analysis
