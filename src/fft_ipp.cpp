#include "audio_utils/fft.h"

#include <ipp.h>

#include "audio_utils/array_math.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <complex>
#include <format>
#include <iostream>
#include <span>

namespace
{

} // namespace

namespace audio_utils
{

struct FFT::FFTState
{
    uint32_t fft_size_{0};
    uint32_t complex_sample_count_{0};
    IppsFFTSpec_R_32f* fft_spec_{nullptr};
    Ipp8u* spec_buffer_{nullptr};
    Ipp8u* work_buffer_{nullptr};

    Ipp32f* signal_buffer_{nullptr};
    std::span<Ipp32f> signal_span_;
    Ipp32fc* spectrum_buffer_{nullptr};
    std::span<std::complex<float>> spectrum_span_;
};

uint32_t FFT::NextSupportedFFTSize(uint32_t target_size)
{
    return std::bit_ceil(target_size);
}

FFT::FFT(uint32_t fft_size)
{
    state_ = std::make_unique<FFTState>();
    state_->fft_size_ = fft_size;
    state_->complex_sample_count_ = (state_->fft_size_ / 2) + 1;

    int order = static_cast<int>(std::log2(state_->fft_size_));

    assert((1u << order) == state_->fft_size_ && "FFT size must be a power of two for IPP FFT");

    constexpr int kFlag = IPP_FFT_DIV_INV_BY_N;
    int size_fft_spec = 0;
    int size_fft_init_buffer = 0;
    int size_fft_work_buffer = 0;
    IppStatus status = ippsFFTGetSize_R_32f(order, kFlag, ippAlgHintNone, &size_fft_spec, &size_fft_init_buffer,
                                            &size_fft_work_buffer);
    if (status != ippStsNoErr)
    {
        throw std::runtime_error(std::format("Failed to get FFT spec size for FFT size {}", state_->fft_size_));
    }

    Ipp8u* fft_spec_buffer = ippsMalloc_8u(size_fft_spec);
    Ipp8u* fft_init_buffer = ippsMalloc_8u(size_fft_init_buffer);
    Ipp8u* fft_work_buffer = ippsMalloc_8u(size_fft_work_buffer);
    IppsFFTSpec_R_32f* fft_spec = nullptr;

    status = ippsFFTInit_R_32f(&fft_spec, order, kFlag, ippAlgHintNone, fft_spec_buffer, fft_init_buffer);
    ippsFree(fft_init_buffer);
    if (status != ippStsNoErr)
    {
        ippsFree(fft_spec_buffer);
        ippsFree(fft_work_buffer);
        throw std::runtime_error(std::format("Failed to initialize FFT spec for FFT size {}", state_->fft_size_));
    }

    state_->fft_spec_ = fft_spec;
    state_->spec_buffer_ = fft_spec_buffer;
    state_->work_buffer_ = fft_work_buffer;

    state_->signal_buffer_ = ippsMalloc_32f(state_->fft_size_);
    state_->spectrum_buffer_ = ippsMalloc_32fc(state_->complex_sample_count_);
#pragma clang unsafe_buffer_usage begin
    state_->signal_span_ = std::span<Ipp32f>(state_->signal_buffer_, state_->fft_size_);
    state_->spectrum_span_ = std::span<std::complex<float>>(
        reinterpret_cast<std::complex<float>*>(state_->spectrum_buffer_), state_->complex_sample_count_);
#pragma clang unsafe_buffer_usage end
}

FFT::~FFT()
{
    if (state_->spec_buffer_ != nullptr)
    {
        ippsFree(state_->spec_buffer_);
        state_->spec_buffer_ = nullptr;
    }

    if (state_->work_buffer_ != nullptr)
    {
        ippsFree(state_->work_buffer_);
        state_->work_buffer_ = nullptr;
    }

    if (state_->signal_buffer_ != nullptr)
    {
        ippsFree(state_->signal_buffer_);
        state_->signal_buffer_ = nullptr;
    }

    if (state_->spectrum_buffer_ != nullptr)
    {
        ippsFree(state_->spectrum_buffer_);
        state_->spectrum_buffer_ = nullptr;
    }
}

void FFT::Forward(std::span<const float> signal, std::span<complex_t> spectrum)
{
    if (signal.size() > state_->fft_size_)
    {
        throw std::invalid_argument("Input size must be smaller or equal to FFT size");
    }
    if (spectrum.size() != state_->complex_sample_count_)
    {
        throw std::invalid_argument(std::format("Output spectrum size is incorrect: expected {}, got {}",
                                                state_->complex_sample_count_, spectrum.size()));
    }

    const float* signal_data = signal.data();

    // Do zero-padding if necessary
    if (signal.size() < state_->fft_size_)
    {
        std::copy(signal.begin(), signal.end(), state_->signal_buffer_);
        std::ranges::fill(state_->signal_span_.subspan(signal.size()), 0.0f);
        signal_data = state_->signal_buffer_;
    }

    ippsFFTFwd_RToCCS_32f(signal_data, reinterpret_cast<Ipp32f*>(spectrum.data()), state_->fft_spec_,
                          state_->work_buffer_);
}

void FFT::ForwardMag(std::span<const float> signal, std::span<float> spectrum, const ForwardFFTOptions& options)
{
    if (signal.size() > state_->fft_size_)
    {
        throw std::invalid_argument("Input size must be smaller or equal to FFT size");
    }
    if (spectrum.size() != state_->complex_sample_count_)
    {
        throw std::invalid_argument(std::format("Output spectrum size is incorrect: expected {}, got {}",
                                                state_->complex_sample_count_, spectrum.size()));
    }

    const float* signal_data = signal.data();

    if (signal.size() < state_->fft_size_)
    {
        std::copy(signal.begin(), signal.end(), state_->signal_buffer_);
        std::ranges::fill(state_->signal_span_.subspan(signal.size()), 0.0f);
        signal_data = state_->signal_buffer_;
    }

    ippsFFTFwd_RToCCS_32f(signal_data, reinterpret_cast<Ipp32f*>(state_->spectrum_buffer_), state_->fft_spec_,
                          state_->work_buffer_);

    if (options.output_type == FFTOutputType::Magnitude)
    {
        array_math::Magnitude(state_->spectrum_span_, spectrum);
    }
    else if (options.output_type == FFTOutputType::Power)
    {
        array_math::PowerSpectrum(state_->spectrum_span_, spectrum);
    }

    if (options.to_db)
    {
        audio_utils::array_math::ToDb(spectrum, options.output_type == FFTOutputType::Magnitude ? 20.f : 10.f);
    }
}

void FFT::Inverse(std::span<const complex_t> spectrum, std::span<float> signal)
{
    if (signal.size() < state_->fft_size_)
    {
        throw std::invalid_argument("Input size must be larger or equal to FFT size");
    }
    if (spectrum.size() != state_->complex_sample_count_)
    {
        throw std::invalid_argument(std::format("Output spectrum size is incorrect: expected {}, got {}",
                                                state_->complex_sample_count_, spectrum.size()));
    }

    ippsFFTInv_CCSToR_32f(reinterpret_cast<const Ipp32f*>(spectrum.data()), signal.data(), state_->fft_spec_,
                          state_->work_buffer_);
}

void FFT::RealCepstrum(std::span<const float> signal, std::span<float> cepstrum)
{
    if (signal.size() > state_->fft_size_)
    {
        throw std::invalid_argument("Input size must be smaller or equal to FFT size");
    }

    if (cepstrum.size() != state_->fft_size_)
    {
        throw std::invalid_argument("Cepstrum output size must be equal to FFT size");
    }

    const float* signal_data = signal.data();

    if (signal.size() < state_->fft_size_)
    {
        std::copy(signal.begin(), signal.end(), state_->signal_buffer_);
        std::ranges::fill(state_->signal_span_.subspan(signal.size()), 0.0f);
        signal_data = state_->signal_buffer_;
    }

    ippsFFTFwd_RToCCS_32f(signal_data, reinterpret_cast<Ipp32f*>(state_->spectrum_buffer_), state_->fft_spec_,
                          state_->work_buffer_);

    ippsMagnitude_32fc(state_->spectrum_buffer_, state_->signal_buffer_, state_->complex_sample_count_);
    ippsLn_32f_I(state_->signal_buffer_, state_->complex_sample_count_);
    ippsRealToCplx_32f(state_->signal_buffer_, nullptr, state_->spectrum_buffer_, state_->complex_sample_count_);
    ippsFFTInv_CCSToR_32f(reinterpret_cast<const Ipp32f*>(state_->spectrum_buffer_), cepstrum.data(), state_->fft_spec_,
                          state_->work_buffer_);
}

void FFT::Convolve(std::span<const float> signal, std::span<const float> filter, std::span<float> result)
{
    if (signal.size() + filter.size() - 1 > result.size())
    {
        throw std::invalid_argument("Result size must be equal to signal size + filter size - 1");
    }

    const uint32_t conv_size = signal.size() + filter.size() - 1;

    if (conv_size > state_->fft_size_)
    {
        throw std::invalid_argument("Convolution size must be smaller or equal to FFT size");
    }

    IppEnum fun_cfg = static_cast<IppEnum>(ippAlgAuto);
    int buffer_size = 0;
    Ipp8u* work_buffer = nullptr;
    IppStatus status = ippsConvolveGetBufferSize(signal.size(), filter.size(), ipp32f, fun_cfg, &buffer_size);
    if (status != ippStsNoErr)
    {
        throw std::runtime_error("Failed to get convolution buffer size");
    }

    work_buffer = ippsMalloc_8u(buffer_size);

    status = ippsConvolve_32f(signal.data(), signal.size(), filter.data(), filter.size(), result.data(), fun_cfg,
                              work_buffer);
    ippsFree(work_buffer);

    if (status != ippStsNoErr)
    {
        throw std::runtime_error("ippsConvolve_32f failed with error code " + std::to_string(status));
    }

    return;
}

uint32_t FFT::GetFFTSize() const
{
    return state_->fft_size_;
}

uint32_t FFT::GetSpectrumSize() const
{
    return state_->complex_sample_count_;
}

} // namespace audio_utils