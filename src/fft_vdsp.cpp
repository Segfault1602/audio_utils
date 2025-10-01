#include "audio_utils/fft.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cassert>
#include <complex>
#include <format>
#include <mach/memory_object_types.h>
#include <span>

namespace
{
constexpr uint32_t IsPowerOfTwo(uint32_t N)
{
    /* https://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2 */
    return N && !(N & (N - 1));
}

float* AllocateBuffer(size_t size)
{
    size_t alignedSizeBytes = size * sizeof(float);
    if (alignedSizeBytes % 64 != 0)
    {
        alignedSizeBytes += 64 - (alignedSizeBytes % 64);
    }

#pragma clang unsafe_buffer_usage begin
    float* buf = (float*)aligned_alloc(64, alignedSizeBytes);
    memset(buf, 0, alignedSizeBytes);
#pragma clang unsafe_buffer_usage end
    return buf;
}

void FreeBuffer(float* buffer)
{
    free(buffer);
}

} // namespace

namespace audio_utils
{
struct FFT::FFTState
{
    vDSP_DFT_Setup mForwardSetup;
    vDSP_DFT_Setup mInverseSetup;

    size_t numRealSamples;
    size_t numComplexSamples;
    std::array<float*, 2> deinterleaved;
    std::span<float> real_span_;
    std::span<float> imag_span_;
};

uint32_t FFT::NextSupportedFFTSize(uint32_t target_size)
{
    if (target_size < 32)
    {
        return 32;
    }

    // For real-to-complex transform, vDSP support lengths of the form
    // f * 2^n, where f is 1, 3, 5, or 15 and n >= 4
    constexpr std::array<uint32_t, 4> kFactors = {1, 3, 5, 15};

    uint32_t best_match = 1 << 31;

    for (auto factor : kFactors)
    {
        // 24 is arbitrary
        for (auto n = 4; n < 24; ++n)
        {
            uint32_t possible_size = factor * (1 << n);
            if (possible_size >= target_size)
            {
                best_match = std::min(best_match, possible_size);
            }
        }
    }

    return best_match;
}

FFT::FFT(uint32_t fft_size)
{
    if (!IsPowerOfTwo(fft_size))
    {
        // throw std::invalid_argument("FFT size must be a power of two");
    }

    state_ = std::make_unique<FFTState>();

    state_->numRealSamples = fft_size;
    state_->numComplexSamples = (state_->numRealSamples / 2 + 1);

    state_->mForwardSetup = vDSP_DFT_zrop_CreateSetup(nullptr, state_->numRealSamples, vDSP_DFT_FORWARD);
    if (!state_->mForwardSetup)
    {
        throw std::runtime_error("Unable to create vDSP forward DFT setup.");
    }

    state_->mInverseSetup = vDSP_DFT_zrop_CreateSetup(nullptr, state_->numRealSamples, vDSP_DFT_INVERSE);
    if (!state_->mInverseSetup)
    {
        throw std::runtime_error("Unable to create vDSP inverse DFT setup.");
    }

    state_->deinterleaved[0] = AllocateBuffer(state_->numComplexSamples);
    state_->deinterleaved[1] = AllocateBuffer(state_->numComplexSamples);
#pragma clang unsafe_buffer_usage begin
    state_->real_span_ = std::span<float>(state_->deinterleaved[0], state_->numComplexSamples);
    state_->imag_span_ = std::span<float>(state_->deinterleaved[1], state_->numComplexSamples);
#pragma clang unsafe_buffer_usage end
}

FFT::~FFT()
{
    if (state_->mForwardSetup)
    {
        vDSP_DFT_DestroySetup(state_->mForwardSetup);
        state_->mForwardSetup = nullptr;
    }

    if (state_->mInverseSetup)
    {
        vDSP_DFT_DestroySetup(state_->mInverseSetup);
        state_->mInverseSetup = nullptr;
    }

    if (state_->deinterleaved[0] != nullptr)
    {
        FreeBuffer(state_->deinterleaved[0]);
        state_->deinterleaved[0] = nullptr;
    }

    if (state_->deinterleaved[1] != nullptr)
    {
        FreeBuffer(state_->deinterleaved[1]);
        state_->deinterleaved[1] = nullptr;
    }
}

void FFT::Forward(std::span<const float> signal, std::span<complex_t> spectrum)
{
    if (signal.size() > state_->numRealSamples)
    {
        throw std::invalid_argument("Input size must be smaller or equal to FFT size");
    }
    if (spectrum.size() != state_->numComplexSamples)
    {
        throw std::invalid_argument(std::format("Output spectrum size is incorrect: expected {}, got {}",
                                                state_->numComplexSamples, spectrum.size()));
    }

    DSPSplitComplex splitComplex{};
    splitComplex.realp = state_->deinterleaved[0];
    splitComplex.imagp = state_->deinterleaved[1];

    std::ranges::fill(state_->real_span_, 0.0f);
    std::ranges::fill(state_->imag_span_, 0.0f);

#pragma clang unsafe_buffer_usage begin
    vDSP_ctoz(reinterpret_cast<const DSPComplex*>(signal.data()), 2, &splitComplex, 1, signal.size() / 2);
#pragma clang unsafe_buffer_usage end

    vDSP_DFT_Execute(state_->mForwardSetup, splitComplex.realp, splitComplex.imagp, splitComplex.realp,
                     splitComplex.imagp);

    vDSP_ztoc(&splitComplex, 1, reinterpret_cast<DSPComplex*>(spectrum.data()), 2, state_->numRealSamples / 2);

    spectrum[state_->numComplexSamples - 1].real(spectrum[0].imag());
    spectrum[0].imag(0.0f);

    float scalar = 0.5f;
    vDSP_vsmul(reinterpret_cast<const float*>(spectrum.data()), 1, &scalar, reinterpret_cast<float*>(spectrum.data()),
               1, 2 * state_->numComplexSamples);
}

void FFT::ForwardAbs(std::span<const float> signal, std::span<float> abs_spectrum, bool to_db, bool normalize)
{
    if (signal.size() > state_->numRealSamples)
    {
        throw std::invalid_argument("Input size must be smaller or equal to FFT size");
    }
    if (abs_spectrum.size() != state_->numComplexSamples)
    {
        throw std::invalid_argument(std::format("Output spectrum size is incorrect: expected {}, got {}",
                                                state_->numComplexSamples, abs_spectrum.size()));
    }

    DSPSplitComplex splitComplex{};
    splitComplex.realp = state_->deinterleaved[0];
    splitComplex.imagp = state_->deinterleaved[1];

    std::ranges::fill(state_->real_span_, 0.0f);
    std::ranges::fill(state_->imag_span_, 0.0f);

#pragma clang unsafe_buffer_usage begin
    vDSP_ctoz(reinterpret_cast<const DSPComplex*>(signal.data()), 2, &splitComplex, 1, signal.size() / 2);
#pragma clang unsafe_buffer_usage end

    vDSP_DFT_Execute(state_->mForwardSetup, splitComplex.realp, splitComplex.imagp, splitComplex.realp,
                     splitComplex.imagp);

#pragma clang unsafe_buffer_usage begin
    splitComplex.realp[state_->numComplexSamples - 1] = splitComplex.imagp[0];
    splitComplex.imagp[0] = 0.0f;
#pragma clang unsafe_buffer_usage end

    vDSP_zvabs(&splitComplex, 1, abs_spectrum.data(), 1, state_->numComplexSamples);

    float scalar = 0.5f;
    vDSP_vsmul(abs_spectrum.data(), 1, &scalar, abs_spectrum.data(), 1, state_->numComplexSamples);

    if (normalize)
    {
        float scale = 1.f / *std::ranges::max_element(abs_spectrum);
        vDSP_vsmul(abs_spectrum.data(), 1, &scale, abs_spectrum.data(), 1, abs_spectrum.size());
    }

    if (to_db)
    {
        float zero_ref = 1.0f;
        vDSP_vdbcon(abs_spectrum.data(), 1, &zero_ref, abs_spectrum.data(), 1, abs_spectrum.size(), 1);
    }
}

void FFT::Inverse(std::span<const complex_t> spectrum, std::span<float> signal)
{
    if (signal.size() < state_->numRealSamples)
    {
        throw std::invalid_argument("Input size must be larger or equal to FFT size");
    }
    if (spectrum.size() != state_->numComplexSamples)
    {
        throw std::invalid_argument(std::format("Output spectrum size is incorrect: expected {}, got {}",
                                                state_->numComplexSamples, spectrum.size()));
    }

    DSPSplitComplex splitComplex{};
    splitComplex.realp = state_->deinterleaved[0];
    splitComplex.imagp = state_->deinterleaved[1];

    vDSP_ctoz(reinterpret_cast<const DSPComplex*>(spectrum.data()), 2, &splitComplex, 1, state_->numComplexSamples);

#pragma clang unsafe_buffer_usage begin
    splitComplex.imagp[0] = splitComplex.realp[state_->numComplexSamples - 1];
    splitComplex.realp[state_->numComplexSamples - 1] = 0.0f;
#pragma clang unsafe_buffer_usage end

    vDSP_DFT_Execute(state_->mInverseSetup, splitComplex.realp, splitComplex.imagp, splitComplex.realp,
                     splitComplex.imagp);

#pragma clang unsafe_buffer_usage begin
    vDSP_ztoc(&splitComplex, 1, reinterpret_cast<DSPComplex*>(signal.data()), 2, state_->numRealSamples / 2);
#pragma clang unsafe_buffer_usage end

    float scalar = 1.0f / state_->numRealSamples;
    vDSP_vsmul(signal.data(), 1, &scalar, signal.data(), 1, state_->numRealSamples);
}

void FFT::RealCepstrum(std::span<const float> signal, std::span<float> cepstrum)
{
    if (signal.size() > state_->numRealSamples)
    {
        throw std::invalid_argument("Input size must be smaller or equal to FFT size");
    }
    if (cepstrum.size() != state_->numRealSamples)
    {
        throw std::invalid_argument(std::format("Output spectrum size is incorrect: expected {}, got {}",
                                                state_->numRealSamples, cepstrum.size()));
    }

    DSPSplitComplex splitComplex{};
    splitComplex.realp = state_->deinterleaved[0];
    splitComplex.imagp = state_->deinterleaved[1];

    std::ranges::fill(state_->real_span_, 0.0f);
    std::ranges::fill(state_->imag_span_, 0.0f);

#pragma clang unsafe_buffer_usage begin
    vDSP_ctoz(reinterpret_cast<const DSPComplex*>(signal.data()), 2, &splitComplex, 1, signal.size() / 2);
#pragma clang unsafe_buffer_usage end

    vDSP_DFT_Execute(state_->mForwardSetup, splitComplex.realp, splitComplex.imagp, splitComplex.realp,
                     splitComplex.imagp);

    state_->real_span_[0] = std::log(std::abs(0.5f * state_->real_span_[0])); // DC component
    state_->imag_span_[0] = std::log(std::abs(0.5f * state_->imag_span_[0])); // Nyquist component

    for (auto i = 1; i < state_->numComplexSamples - 1; ++i)
    {
        auto cpl_val = std::complex<float>(state_->real_span_[i], state_->imag_span_[i]);
        state_->real_span_[i] = std::log(std::abs(cpl_val * 0.5f));
        state_->imag_span_[i] = 0.0f;
    }

    vDSP_DFT_Execute(state_->mInverseSetup, splitComplex.realp, splitComplex.imagp, splitComplex.realp,
                     splitComplex.imagp);

#pragma clang unsafe_buffer_usage begin
    vDSP_ztoc(&splitComplex, 1, reinterpret_cast<DSPComplex*>(cepstrum.data()), 2, state_->numRealSamples / 2);
#pragma clang unsafe_buffer_usage end

    float scalar = 1.0f / state_->numRealSamples;
    vDSP_vsmul(cepstrum.data(), 1, &scalar, cepstrum.data(), 1, state_->numRealSamples);
}

} // namespace audio_utils