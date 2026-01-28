#include "audio_utils/fft.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cassert>
#include <complex>
#include <format>
#include <iostream>
#include <span>
#include <vector>

namespace
{

float* AllocateBuffer(size_t size)
{
    constexpr size_t kAlignment = 64;
    size_t alignedSizeBytes = size * sizeof(float);
    if (alignedSizeBytes % kAlignment != 0)
    {
        alignedSizeBytes += kAlignment - (alignedSizeBytes % kAlignment);
    }

#pragma clang unsafe_buffer_usage begin
    float* buf = (float*)aligned_alloc(kAlignment, alignedSizeBytes);
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

void FFT::ForwardMag(std::span<const float> signal, std::span<float> mag_spectrum, const ForwardFFTOptions& options)
{
    if (signal.size() > state_->numRealSamples)
    {
        throw std::invalid_argument("Input size must be smaller or equal to FFT size");
    }
    if (mag_spectrum.size() != state_->numComplexSamples)
    {
        throw std::invalid_argument(std::format("Output spectrum size is incorrect: expected {}, got {}",
                                                state_->numComplexSamples, mag_spectrum.size()));
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

    if (options.output_type == FFTOutputType::Magnitude)
    {
        vDSP_zvabs(&splitComplex, 1, mag_spectrum.data(), 1, state_->numComplexSamples);
    }
    else if (options.output_type == FFTOutputType::Power)
    {
        vDSP_zvabs(&splitComplex, 1, mag_spectrum.data(), 1, state_->numComplexSamples);
        vDSP_vsq(mag_spectrum.data(), 1, mag_spectrum.data(), 1, state_->numComplexSamples);
    }

    float scalar = 0.5f;
    vDSP_vsmul(mag_spectrum.data(), 1, &scalar, mag_spectrum.data(), 1, state_->numComplexSamples);

    if (options.to_db)
    {
        float zero_ref = 1.0f;
        vDSP_vdbcon(mag_spectrum.data(), 1, &zero_ref, mag_spectrum.data(), 1, mag_spectrum.size(),
                    options.output_type == FFTOutputType::Magnitude ? 1 : 0);
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

void FFT::Convolve(std::span<const float> signal, std::span<const float> filter, std::span<float> result)
{
    if (signal.size() + filter.size() - 1 > result.size())
    {
        throw std::invalid_argument("Result size must be equal to signal size + filter size - 1");
    }

    const uint32_t conv_size = signal.size() + filter.size() - 1;

    if (conv_size > state_->numRealSamples)
    {
        throw std::invalid_argument("Convolution size must be smaller or equal to FFT size");
    }

#pragma clang unsafe_buffer_usage begin
    float* aligned_signal = AllocateBuffer(state_->numRealSamples);
    float* aligned_filter = AllocateBuffer(state_->numRealSamples);
    float* aligned_convolution = AllocateBuffer(state_->numRealSamples);
    std::span<float> aligned_signal_span(aligned_signal, state_->numRealSamples);
    std::span<float> aligned_filter_span(aligned_filter, state_->numRealSamples);
    std::span<float> aligned_convolution_span(aligned_convolution, state_->numRealSamples);
#pragma clang unsafe_buffer_usage end

    std::ranges::copy(signal, aligned_signal_span.begin());
    std::ranges::fill(aligned_signal_span.subspan(signal.size()), 0.0f); // Zero-pad if necessary

    std::ranges::copy(filter, aligned_filter_span.begin());
    std::ranges::fill(aligned_filter_span.subspan(filter.size()), 0.0f); // Zero-pad if necessary

    // Fill the convolution buffer with zeros
    std::ranges::fill(aligned_convolution_span, 0.0f);

    DSPSplitComplex signalSplit{};
    signalSplit.realp = AllocateBuffer(state_->numComplexSamples);
    signalSplit.imagp = AllocateBuffer(state_->numComplexSamples);

    DSPSplitComplex filterSplit{};
    filterSplit.realp = AllocateBuffer(state_->numComplexSamples);
    filterSplit.imagp = AllocateBuffer(state_->numComplexSamples);

    vDSP_ctoz(reinterpret_cast<const DSPComplex*>(aligned_signal), 2, &signalSplit, 1, state_->numRealSamples / 2);
    vDSP_ctoz(reinterpret_cast<const DSPComplex*>(aligned_filter), 2, &filterSplit, 1, state_->numRealSamples / 2);

    vDSP_DFT_Execute(state_->mForwardSetup, signalSplit.realp, signalSplit.imagp, signalSplit.realp, signalSplit.imagp);
    vDSP_DFT_Execute(state_->mForwardSetup, filterSplit.realp, filterSplit.imagp, filterSplit.realp, filterSplit.imagp);

#pragma clang unsafe_buffer_usage begin
    float nyquist = signalSplit.imagp[0] * filterSplit.imagp[0];
    signalSplit.imagp[0] = 0.0f;
    filterSplit.imagp[0] = 0.0f;

    vDSP_zvmul(&signalSplit, 1, &filterSplit, 1, &signalSplit, 1, state_->numRealSamples / 2, 1);

    signalSplit.imagp[0] = nyquist;
#pragma clang unsafe_buffer_usage end

    vDSP_DFT_Execute(state_->mInverseSetup, signalSplit.realp, signalSplit.imagp, signalSplit.realp, signalSplit.imagp);

    vDSP_ztoc(&signalSplit, 1, reinterpret_cast<DSPComplex*>(aligned_convolution), 2, state_->numRealSamples / 2);

    float scalar = 1.0f / (4 * state_->numRealSamples);
    vDSP_vsmul(aligned_convolution, 1, &scalar, aligned_convolution, 1, state_->numRealSamples);
    std::ranges::copy(aligned_convolution_span.subspan(0, result.size()), result.begin());

    FreeBuffer(aligned_signal);
    FreeBuffer(aligned_filter);
    FreeBuffer(aligned_convolution);

    FreeBuffer(signalSplit.realp);
    FreeBuffer(signalSplit.imagp);
    FreeBuffer(filterSplit.realp);
    FreeBuffer(filterSplit.imagp);
}

uint32_t FFT::GetFFTSize() const
{
    return static_cast<uint32_t>(state_->numRealSamples);
}

uint32_t FFT::GetSpectrumSize() const
{
    return static_cast<uint32_t>(state_->numComplexSamples);
}

} // namespace audio_utils