#include "audio_metrics.h"

#include "fft_utils.h"
#include <fftw3.h>

namespace audio_utils
{
namespace metrics
{

std::vector<float> Autocorrelation(std::span<const float> signal, bool normalize)
{
    const size_t kNFFT = 2 * signal.size();
    const size_t kInSize = kNFFT * sizeof(float);
    const size_t kOutSize = (kNFFT / 2 + 1) * sizeof(fftwf_complex);
    float* aligned_in = static_cast<float*>(fftwf_malloc(kInSize));
    fftwf_complex* aligned_out = static_cast<fftwf_complex*>(fftwf_malloc(kOutSize));

    fftwf_plan forward_plan = fftwf_plan_dft_r2c_1d(kNFFT, aligned_in, aligned_out, FFTW_ESTIMATE);
    fftwf_plan backward_plan = fftwf_plan_dft_c2r_1d(kNFFT, aligned_out, aligned_in, FFTW_ESTIMATE);

    std::fill(aligned_in, aligned_in + kNFFT, 0.0f);
    // std::fill(aligned_out, aligned_out + kNFFT, 0.0f);
    std::copy(signal.begin(), signal.end(), aligned_in);

    fftwf_execute(forward_plan);

    for (size_t i = 0; i < (kNFFT / 2 + 1); ++i)
    {
        std::complex<float> c(aligned_out[i][0], aligned_out[i][1]);
        aligned_out[i][0] = std::pow(std::abs(c), 2);
        aligned_out[i][1] = 0.0f; // Set imaginary parts to zero
    }

    fftwf_execute(backward_plan);

    std::vector<float> out(signal.size());
    for (size_t i = 0; i < signal.size(); ++i)
    {
        out[i] = aligned_in[i] / static_cast<float>(kNFFT);
    }

    if (normalize)
    {
        const float coeff = out[0];
        for (size_t i = 0; i < signal.size(); ++i)
        {
            out[i] /= coeff; // Normalize so the the first value is 1.0
        }
    }

    fftwf_free(aligned_in);
    fftwf_free(aligned_out);
    fftwf_destroy_plan(forward_plan);
    fftwf_destroy_plan(backward_plan);
    return out;
}

} // namespace metrics
} // namespace audio_utils