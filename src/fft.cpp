#include "audio_utils/fft.h"

#include <pffft/pffft.h>

#include <algorithm>
#include <cassert>
#include <complex>
#include <format>
#include <span>

namespace
{

// Check if size is of the form  N=(2^a)*(3^b)*(5^c)
bool IsValidSizeForPFFFT(uint32_t size)
{
    constexpr uint32_t kMinSize = 32;

    int R = size;
    while (R >= 5 * kMinSize && (R % 5) == 0)
    {
        R /= 5;
    }
    while (R >= 3 * kMinSize && (R % 3) == 0)
    {
        R /= 3;
    }
    while (R >= 2 * kMinSize && (R % 2) == 0)
    {
        R /= 2;
    }
    return R == kMinSize;
}
} // namespace

namespace audio_utils
{

struct FFT::FFTState
{
    PFFFT_Setup* setup_{nullptr};
    uint32_t fft_size_{0};
    uint32_t complex_sample_count_{0};

    float* work_buffer_storage_{nullptr};
    float* aligned_input_real_storage_{nullptr};
    std::complex<float>* aligned_spectrum_storage_{nullptr};

    std::span<float> aligned_input_real_;
    std::span<std::complex<float>> aligned_spectrum_;
};

uint32_t FFT::NextSupportedFFTSize(uint32_t target_size)
{
    constexpr uint32_t kMinSize = 32;
    if (target_size < kMinSize)
    {
        return kMinSize;
    }

    uint32_t size = target_size;
    uint32_t d = kMinSize;
    size = kMinSize * ((size + kMinSize - 1) / kMinSize);

    for (;; size += d)
    {
        if (IsValidSizeForPFFFT(size))
        {
            return size;
        }
    }
}

FFT::FFT(uint32_t fft_size)
{
    state_ = std::make_unique<FFTState>();
    state_->fft_size_ = fft_size;
    state_->complex_sample_count_ = (state_->fft_size_ / 2) + 1;

    if (state_->fft_size_ % 32 != 0)
    {
        throw std::invalid_argument(std::format("FFT size ({}) must be a multiple of 32 for PFFFT", state_->fft_size_));
    }

    state_->setup_ = pffft_new_setup(state_->fft_size_, PFFFT_REAL);
    if (state_->setup_ == nullptr)
    {
        throw std::runtime_error(std::format("FFT size ({}) is unsuitable for PFFFT", state_->fft_size_));
    }

    if (state_->fft_size_ > 16384)
    {
        const auto work_buffer_size = state_->fft_size_;
        state_->work_buffer_storage_ = static_cast<float*>(pffft_aligned_malloc(work_buffer_size * sizeof(float)));
    }

    state_->aligned_input_real_storage_ = static_cast<float*>(pffft_aligned_malloc(state_->fft_size_ * sizeof(float)));
    state_->aligned_spectrum_storage_ = static_cast<std::complex<float>*>(
        pffft_aligned_malloc((state_->complex_sample_count_ - 1) * sizeof(std::complex<float>)));

#pragma clang unsafe_buffer_usage begin
    state_->aligned_input_real_ = std::span<float>(state_->aligned_input_real_storage_, state_->fft_size_);
    state_->aligned_spectrum_ =
        std::span<std::complex<float>>(state_->aligned_spectrum_storage_, state_->complex_sample_count_ - 1);
#pragma clang unsafe_buffer_usage end

#ifdef NDEBUG
    std::ranges::fill(state_->aligned_input_real_, 0.0f);
    std::ranges::fill(state_->aligned_spectrum_, std::complex<float>(0.0f, 0.0f));
#else
    // Fill garbage values in debug mode to catch bugs
    std::ranges::fill(state_->aligned_input_real_, 1.23345f);
    std::ranges::fill(state_->aligned_spectrum_, std::complex<float>(1.23345f, 1.2344556f));
#endif
}

FFT::~FFT()
{
    if (state_->setup_ != nullptr)
    {
        pffft_destroy_setup(state_->setup_);
        state_->setup_ = nullptr;
    }

    if (state_->work_buffer_storage_ != nullptr)
    {
        pffft_aligned_free(state_->work_buffer_storage_);
        state_->work_buffer_storage_ = nullptr;
    }

    if (state_->aligned_input_real_storage_ != nullptr)
    {
        pffft_aligned_free(state_->aligned_input_real_storage_);
        state_->aligned_input_real_storage_ = nullptr;
    }

    if (state_->aligned_spectrum_storage_ != nullptr)
    {
        pffft_aligned_free(state_->aligned_spectrum_storage_);
        state_->aligned_spectrum_storage_ = nullptr;
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

    std::ranges::copy(signal, state_->aligned_input_real_.begin());
    std::ranges::fill(state_->aligned_input_real_.subspan(signal.size()), 0.0f); // Zero-pad if necessary

    pffft_transform_ordered(state_->setup_, state_->aligned_input_real_.data(),
                            reinterpret_cast<float*>(state_->aligned_spectrum_storage_), state_->work_buffer_storage_,
                            PFFFT_FORWARD);

    std::ranges::copy(state_->aligned_spectrum_, spectrum.begin());

    spectrum[state_->complex_sample_count_ - 1].real(spectrum[0].imag());
    spectrum[0].imag(0.0f);
}

void FFT::ForwardAbs(std::span<const float> signal, std::span<float> abs_spectrum, bool to_db, bool normalize)
{
    if (signal.size() > state_->fft_size_)
    {
        throw std::invalid_argument("Input size must be smaller or equal to FFT size");
    }
    if (abs_spectrum.size() != state_->complex_sample_count_)
    {
        throw std::invalid_argument(std::format("Output spectrum size is incorrect: expected {}, got {}",
                                                state_->complex_sample_count_, abs_spectrum.size()));
    }

    std::ranges::copy(signal, state_->aligned_input_real_.begin());
    std::ranges::fill(state_->aligned_input_real_.subspan(signal.size()), 0.0f); // Zero-pad if necessary

    pffft_transform_ordered(state_->setup_, state_->aligned_input_real_.data(),
                            reinterpret_cast<float*>(state_->aligned_spectrum_storage_), state_->work_buffer_storage_,
                            PFFFT_FORWARD);

    abs_spectrum[0] = std::abs(state_->aligned_spectrum_[0].real()); // DC component
    abs_spectrum[state_->complex_sample_count_ - 1] =
        std::abs(state_->aligned_spectrum_[0].imag()); // Nyquist component

    for (auto i = 1; i < state_->aligned_spectrum_.size(); ++i)
    {
        abs_spectrum[i] = std::abs(state_->aligned_spectrum_[i]);
    }

    if (normalize)
    {
        float max_val = *std::ranges::max_element(abs_spectrum);
        std::ranges::transform(abs_spectrum, abs_spectrum.begin(), [max_val](float val) { return val / max_val; });
    }

    if (to_db)
    {
        std::ranges::transform(abs_spectrum, abs_spectrum.begin(), [](float val) { return 20.0f * std::log10f(val); });
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

    state_->aligned_spectrum_[0].real(spectrum[0].real()); // DC component
    state_->aligned_spectrum_[0].imag(spectrum[state_->complex_sample_count_ - 1].real());

    for (size_t i = 1; i < state_->complex_sample_count_ - 1; ++i)
    {
        state_->aligned_spectrum_[i] = spectrum[i];
    }
    pffft_transform_ordered(state_->setup_, reinterpret_cast<float*>(state_->aligned_spectrum_.data()),
                            state_->aligned_input_real_.data(), state_->work_buffer_storage_, PFFFT_BACKWARD);

    float scale = 1.0f / static_cast<float>(state_->fft_size_);
    std::ranges::transform(state_->aligned_input_real_, signal.begin(), [scale](float val) { return val * scale; });
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

    std::ranges::copy(signal, state_->aligned_input_real_.begin());
    std::ranges::fill(state_->aligned_input_real_.subspan(signal.size()), 0.0f); // Zero-pad if necessary

    pffft_transform_ordered(state_->setup_, state_->aligned_input_real_.data(),
                            reinterpret_cast<float*>(state_->aligned_spectrum_storage_), state_->work_buffer_storage_,
                            PFFFT_FORWARD);

    state_->aligned_spectrum_[0].real(std::log(std::abs(state_->aligned_spectrum_[0].real()))); // DC component
    state_->aligned_spectrum_[0].imag(std::log(std::abs(state_->aligned_spectrum_[0].imag()))); // Nyquist component

    std::ranges::transform(state_->aligned_spectrum_.subspan(1, state_->aligned_spectrum_.size() - 1),
                           state_->aligned_spectrum_.subspan(1, state_->aligned_spectrum_.size() - 1).begin(),
                           [](std::complex<float> val) { return std::log(std::abs(val)); });

    pffft_transform_ordered(state_->setup_, reinterpret_cast<float*>(state_->aligned_spectrum_storage_),
                            reinterpret_cast<float*>(state_->aligned_input_real_storage_), state_->work_buffer_storage_,
                            PFFFT_BACKWARD);

    float scale = 1.0f / static_cast<float>(state_->fft_size_);
    std::ranges::transform(state_->aligned_input_real_, cepstrum.begin(),
                           [scale](float val) -> float { return val * scale; });
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

#pragma clang unsafe_buffer_usage begin
    float* aligned_signal = reinterpret_cast<float*>(pffft_aligned_malloc(state_->fft_size_ * sizeof(float)));
    float* aligned_filter = reinterpret_cast<float*>(pffft_aligned_malloc(state_->fft_size_ * sizeof(float)));
    float* aligned_convolution = reinterpret_cast<float*>(pffft_aligned_malloc(state_->fft_size_ * sizeof(float)));
    std::span<float> aligned_signal_span(aligned_signal, state_->fft_size_);
    std::span<float> aligned_filter_span(aligned_filter, state_->fft_size_);
    std::span<float> aligned_convolution_span(aligned_convolution, state_->fft_size_);
#pragma clang unsafe_buffer_usage end

    std::ranges::copy(signal, aligned_signal_span.begin());
    std::ranges::fill(aligned_signal_span.subspan(signal.size()), 0.0f); // Zero-pad if necessary

    std::ranges::copy(filter, aligned_filter_span.begin());
    std::ranges::fill(aligned_filter_span.subspan(filter.size()), 0.0f); // Zero-pad if necessary

    // Fill the convolution buffer with zeros
    std::ranges::fill(aligned_convolution_span, 0.0f);

    pffft_transform(state_->setup_, aligned_signal, aligned_signal, state_->work_buffer_storage_, PFFFT_FORWARD);
    pffft_transform(state_->setup_, aligned_filter, aligned_filter, state_->work_buffer_storage_, PFFFT_FORWARD);

    const float scale = 1.0f / static_cast<float>(state_->fft_size_);
    pffft_zconvolve_accumulate(state_->setup_, aligned_signal, aligned_filter, aligned_convolution, scale);

    pffft_transform(state_->setup_, aligned_convolution, aligned_convolution, state_->work_buffer_storage_,
                    PFFFT_BACKWARD);

    std::ranges::copy(aligned_convolution_span.subspan(0, result.size()), result.begin());

    pffft_aligned_free(aligned_signal);
    pffft_aligned_free(aligned_filter);
    pffft_aligned_free(aligned_convolution);
}

uint32_t FFT::GetFFTSize() const
{
    return state_->fft_size_;
}

} // namespace audio_utils