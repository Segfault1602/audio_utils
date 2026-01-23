#pragma once

#include <complex>
#include <span>

namespace audio_utils::array_math
{
void Multiply(std::span<const float> a, std::span<const float> b, std::span<float> result);

void Divide(std::span<const float> a, float b, std::span<float> result);

void Square(std::span<float> data);

void Ln(std::span<const float> data, std::span<float> result);

float Mean(std::span<const float> data);

void Magnitude(std::span<const std::complex<float>> spectrum, std::span<float> result);

void PowerSpectrum(std::span<const std::complex<float>> spectrum, std::span<float> result);
void PowerSpectrum(std::span<const std::complex<float>> spectrum, std::span<std::complex<float>> result);

void ToDb(std::span<float> data, float scale = 10.f);

} // namespace audio_utils::array_math