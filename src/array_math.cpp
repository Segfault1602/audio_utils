#include "audio_utils/array_math.h"

#ifdef AUDIO_UTILS_USE_IPP
#include <ipp.h>
#endif

#include <numeric>
#include <span>
#include <stdexcept>
#include <string>

namespace audio_utils::array_math
{
void Multiply(std::span<const float> a, std::span<const float> b, std::span<float> result)
{
    // if (a.size() != b.size() || a.size() != result.size())
    // {
    //     throw std::invalid_argument("Input spans must have the same size");
    // }
#ifndef AUDIO_UTILS_USE_IPP
    for (size_t i = 0; i < a.size(); ++i)
    {
        result[i] = a[i] * b[i];
    }
#else
    IppStatus status = ippsMul_32f(a.data(), b.data(), result.data(), static_cast<int>(a.size()));
    if (status != ippStsNoErr)
    {
        throw std::runtime_error("ippsMul_32f failed with error code " + std::to_string(status));
    }
#endif
}

void Divide(std::span<const float> a, float b, std::span<float> result)
{
#ifndef AUDIO_UTILS_USE_IPP
    for (size_t i = 0; i < a.size(); ++i)
    {
        result[i] = a[i] / b;
    }
#else
    IppStatus status = ippsDivC_32f(a.data(), b, result.data(), static_cast<int>(a.size()));
    if (status != ippStsNoErr)
    {
        throw std::runtime_error("ippsDivC_32f failed with error code " + std::to_string(status));
    }
#endif
}

void Square(std::span<float> data)
{
#ifndef AUDIO_UTILS_USE_IPP
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = data[i] * data[i];
    }
#else
    IppStatus status = ippsSqr_32f_I(data.data(), static_cast<int>(data.size()));
    if (status != ippStsNoErr)
    {
        throw std::runtime_error("ippsSqr_32f_I failed with error code " + std::to_string(status));
    }
#endif
}

void Ln(std::span<const float> data, std::span<float> result)
{
#ifndef AUDIO_UTILS_USE_IPP
    for (size_t i = 0; i < data.size(); ++i)
    {
        result[i] = std::log(data[i]);
    }
#else
    IppStatus status = ippsLn_32f(data.data(), result.data(), static_cast<int>(data.size()));
    if (status != ippStsNoErr)
    {
        throw std::runtime_error("ippsLn_32f failed with error code " + std::to_string(status));
    }
#endif
}

float Mean(std::span<const float> data)
{
#ifndef AUDIO_UTILS_USE_IPP
    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    return sum / static_cast<float>(data.size());
#else
    float mean = 0.0f;
    IppStatus status = ippsMean_32f(data.data(), static_cast<int>(data.size()), &mean, ippAlgHintNone);
    if (status != ippStsNoErr)
    {
        throw std::runtime_error("ippsMean_32f failed with error code " + std::to_string(status));
    }

    return mean;
#endif
}

void Magnitude(std::span<const std::complex<float>> spectrum, std::span<float> result)
{
#ifndef AUDIO_UTILS_USE_IPP
    for (size_t i = 0; i < spectrum.size(); ++i)
    {
        result[i] = std::abs(spectrum[i]);
    }
#else
    IppStatus status = ippsMagnitude_32fc(reinterpret_cast<const Ipp32fc*>(spectrum.data()), result.data(),
                                          static_cast<int>(spectrum.size()));
    if (status != ippStsNoErr)
    {
        throw std::runtime_error("ippsMagnitude_32fc failed with error code " + std::to_string(status));
    }
#endif
}

void PowerSpectrum(std::span<const std::complex<float>> spectrum, std::span<float> result)
{
#ifndef AUDIO_UTILS_USE_IPP
    for (size_t i = 0; i < spectrum.size(); ++i)
    {
        result[i] = std::norm(spectrum[i]);
    }
#else
    IppStatus status = ippsPowerSpectr_32fc(reinterpret_cast<const Ipp32fc*>(spectrum.data()), result.data(),
                                            static_cast<int>(spectrum.size()));
    if (status != ippStsNoErr)
    {
        throw std::runtime_error("ippsPowerSpectr_32fc failed with error code " + std::to_string(status));
    }
#endif
}

void PowerSpectrum(std::span<const std::complex<float>> spectrum, std::span<std::complex<float>> result)
{
    for (size_t i = 0; i < spectrum.size(); ++i)
    {
        result[i] = std::norm(spectrum[i]);
    }
}

void ToDb(std::span<float> data, float scale)
{

#ifndef AUDIO_UTILS_USE_IPP
    constexpr float epsilon = 1e-10f;
    for (auto& val : data)
    {
        val = scale * std::log10(val + epsilon);
    }
#else
    ippsLog10_32f_A21(data.data(), data.data(), data.size());
    ippsMulC_32f_I(scale, data.data(), data.size());
#endif
}

} // namespace audio_utils::array_math