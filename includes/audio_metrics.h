#pragma once

#include <span>
#include <vector>

namespace audio_utils
{
namespace metrics
{
/**
 * @brief Computes the autocorrelation of a signal.
 * @param signal The input signal as a span of floats.
 * @return A vector containing the autocorrelation values. Only the positive lags are returned.
 */
std::vector<float> Autocorrelation(std::span<const float> signal, bool normalize = true);

/**
 * @brief Computes the spectrogram of a signal.
 * @param signal The input signal as a span of floats.
 * @param fft_size The size of the FFT to be used.
 * @param window_size The size of the window for each FFT.
 * @param overlap The overlap between consecutive windows.
 * @param samplerate The sample rate of the signal.
 * @param num_freqs Output parameter to store the number of frequency bins.
 * @param num_frames Output parameter to store the number of frames in the spectrogram.
 * @return A vector containing the spectrogram values in column-major order.
 */
std::vector<float> Spectrogram(std::span<const float> signal, int fft_size, int window_size, int overlap,
                               int samplerate, int& num_freqs, int& num_frames);

/** * @brief Computes the Mel spectrogram of a signal.
 * @param signal The input signal as a span of floats.
 * @param fft_size The size of the FFT to be used.
 * @param window_size The size of the window for each FFT.
 * @param overlap The overlap between consecutive windows.
 * @param samplerate The sample rate of the signal.
 * @param n_mels The number of Mel frequency bands.
 * @param num_freqs Output parameter to store the number of frequency bins.
 * @param num_frames Output parameter to store the number of frames in the Mel spectrogram.
 * @return A vector containing the Mel spectrogram values in column-major order.
 */
std::vector<float> MelSpectrogram(std::span<const float> signal, int fft_size, int window_size, int overlap,
                                  int samplerate, size_t n_mels, int& num_freqs, int& num_frames);

} // namespace metrics
} // namespace audio_utils