#include "audio_utils/audio_analysis.h"

#include "audio_utils/array_math.h"
#include "audio_utils/fft.h"
#include "audio_utils/fft_utils.h"
#include "octave_band_filters_fir.h"

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cassert>
#include <iostream>
#include <numbers>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

namespace audio_utils::analysis
{
std::span<float> TrimSilence(std::span<float> signal, float threshold)
{
    if (signal.empty())
    {
        return {};
    }

    // Discard silence at the beginning of impulse response
    const float max_val = array_math::MaxAbs(signal);

    const float target = threshold * std::abs(max_val);

    for (auto i = 0u; i < signal.size(); ++i)
    {
        if (std::abs(signal[i]) >= target)
        {
            return signal.subspan(i);
        }
    }
    return signal;
}

std::vector<float> EnergyDecayCurve(std::span<const float> signal, bool to_db)
{
    if (signal.empty())
    {
        return {};
    }

    std::ranges::reverse_view trimmed_signal_reversed{signal};
    auto s = trimmed_signal_reversed | std::views::transform([](float x) { return x * x; });

    // Calculate the energy decay curve
    std::vector<float> decay_curve(signal.size(), 0.0f);

    std::ranges::reverse_view decay_curve_reversed{decay_curve};

    std::partial_sum(s.begin(), s.end(), decay_curve_reversed.begin());

    if (to_db)
    {
        array_math::ToDb(decay_curve, 10.0f);
    }

    return decay_curve;
}

std::array<std::vector<float>, kNumOctaveBands> EnergyDecayCurve_FilterBank(std::span<const float> signal, bool to_db,
                                                                            uint32_t samplerate)
{
    if (samplerate != 48000)
    {
        throw std::invalid_argument("Only 48000 Hz samplerate is supported for EnergyDecayCurve_FilterBank");
    }

    std::array<std::vector<float>, kNumOctaveBands> edc_octaves;

    static_assert(kOctaveBandFirFilters.size() == edc_octaves.size(),
                  "Number of octave band filters must match number of EDC output vectors");

#pragma omp parallel for
    for (auto i = 0; i < kOctaveBandFirFilters.size(); ++i)
    {
        const auto& fir_coeffs = kOctaveBandFirFilters[i];

        std::vector<float> result = Convolve(signal, fir_coeffs);

        // The filters introduce a delay of roughly half the filter length
        const size_t delay = (fir_coeffs.size() / 2) * 0.9;
        std::span<float> result_span = std::span(result).subspan(delay, signal.size());
        edc_octaves[i] = EnergyDecayCurve(result_span, to_db);
    }

    return edc_octaves;
}

std::array<float, kNumOctaveBands> GetOctaveBandFrequencies()
{
    // TODO: Generate these programmatically
    return {62.5f, 125.0f, 250.0f, 500.0f, 1000.0f, 2000.0f, 4000.0f, 8000.0f, 16000.0f};
}

EnergyDecayReliefResult EnergyDecayRelief(std::span<const float> signal, const EnergyDecayReliefOptions& options)
{
    if (signal.empty())
    {
        return {};
    }

    if (options.fft_length < options.window_size)
    {
        throw std::invalid_argument("FFT length must be greater than or equal to window size");
    }

    if (options.hop_size > options.window_size)
    {
        throw std::invalid_argument("Hop size must be less than or equal to window size");
    }

    audio_utils::analysis::STFTOptions spec_info{
        .fft_size = options.fft_length,
        .overlap = options.window_size - options.hop_size,
        .window_size = options.window_size,
        .window_type = options.window_type,
        .samplerate = 48000,
    };

    auto spectrogram = audio_utils::analysis::MelSpectrogram(signal, spec_info, options.n_mels);

    Eigen::Map<const Eigen::MatrixXf> spec_map(spectrogram.data.data(), spectrogram.num_bins, spectrogram.num_frames);

    std::vector<float> edr_data(spectrogram.num_frames * spectrogram.num_bins, 0);
    Eigen::Map<Eigen::MatrixXf> edr_map(edr_data.data(), spectrogram.num_bins, spectrogram.num_frames);

    Eigen::ArrayXf cumulative_energy = Eigen::ArrayXf::Zero(spectrogram.num_bins);
    for (int i = spec_map.cols() - 1; i >= 0; --i)
    {
        cumulative_energy += spec_map.col(i).array().square();
        edr_map.col(i) = cumulative_energy;
    }

    if (options.to_db)
    {
        array_math::ToDb(edr_data, 10.0f);
    }

    EnergyDecayReliefResult edr_data_struct{
        .data = std::move(edr_data), .num_bins = spectrogram.num_bins, .num_frames = spectrogram.num_frames};

    return edr_data_struct;
}

EstimateT60Results EstimateT60(std::span<const float> decay_curve, std::span<const float> time,
                               EstimateT60Options options)
{
    if (options.decay_start_db <= options.decay_end_db)
    {
        throw std::invalid_argument("decay_start_db must be less than decay_end_db");
    }

    if (decay_curve.size() != time.size())
    {
        throw std::invalid_argument("decay_curve and time must have the same size");
    }

    float start_db_value = decay_curve[0];
    options.decay_start_db += start_db_value;
    options.decay_end_db += start_db_value;

    auto it_start = std::ranges::lower_bound(decay_curve, options.decay_start_db, [](float value, float threshold) {
        return std::abs(value) < std::abs(threshold);
    });
    auto it_end = std::ranges::lower_bound(decay_curve, options.decay_end_db, [](float value, float threshold) {
        return std::abs(value) < std::abs(threshold);
    });

    auto start_index = std::distance(decay_curve.begin(), it_start);
    auto end_index = std::distance(decay_curve.begin(), it_end);

    auto decay_span = std::span(decay_curve).subspan(start_index, end_index - start_index);
    auto time_span = std::span(time).subspan(start_index, end_index - start_index);

    if (time_span.empty() || decay_span.empty())
    {
        // This can happen if the decay curve is not within the specified dB range
        return {.t60 = 0.0f, .decay_start_time = 0.0f, .decay_end_time = 0.0f, .intercept = 0.0f, .slope = 0.0f};
    }

    float c0 = 0.0f;
    float c1 = 0.0f;
    if (options.use_linear_regression)
    {
        // auto [c0, c1] = boost::math::statistics::simple_ordinary_least_squares(time_span, decay_span);
        // least square fit using Eigen
        Eigen::Map<const Eigen::VectorXf> x((time_span.data()), time_span.size());
        Eigen::Map<const Eigen::VectorXf> y((decay_span.data()), decay_span.size());
        Eigen::MatrixXf A(time_span.size(), 2);
        A.col(0) = Eigen::VectorXf::Ones(time_span.size());
        A.col(1) = x;
        Eigen::Vector2f coeffs = (A.transpose() * A).ldlt().solve(A.transpose() * y);
        c0 = coeffs[0];
        c1 = coeffs[1];
    }
    else
    {
        // Use only the first and last points for slope calculation
        const float x1 = time_span.front();
        const float x2 = time_span.back();
        const float y1 = decay_span.front();
        const float y2 = decay_span.back();

        c1 = (y2 - y1) / (x2 - x1);
        c0 = y1 - c1 * x1;
    }

    EstimateT60Results results;
    results.t60 = -60.0f / c1;
    results.decay_start_time = time_span.front();
    results.decay_end_time = time_span.back();
    results.intercept = c0;
    results.slope = c1;

    return results;
}

EchoDensityResults EchoDensity(std::span<const float> signal, const EchoDensityOptions& options)
{
    if (signal.empty() || options.window_size == 0 || options.sample_rate == 0)
    {
        return {};
    }

    EchoDensityResults results;

    std::vector<float> win(options.window_size, 0.0f);
    GetWindow(audio_utils::FFTWindowType::Hann, win);
    float win_sum = std::accumulate(win.begin(), win.end(), 0.0f);
    for (auto& w : win)
    {
        w /= win_sum;
    }

    const int half_win = options.window_size / 2;
    results.echo_densities.reserve((signal.size() + options.hop_size - 1) / options.hop_size);
    results.sparse_indices.reserve((signal.size() + options.hop_size - 1) / options.hop_size);

    const float kErfc = std::erfc(1.0f / std::numbers::sqrt2_v<float>);

    for (int n = 0; n < signal.size(); n += options.hop_size)
    {
        int frame_start = n - half_win;
        int frame_end = n + half_win;
        frame_start = std::max(0, frame_start);
        frame_end = std::min(static_cast<int>(signal.size()), frame_end);

        auto signal_span = signal.subspan(frame_start, frame_end - frame_start);
        auto window_span = std::span(win).subspan(win.size() - signal_span.size(), signal_span.size());

        assert(signal_span.size() == window_span.size());

        Eigen::Map<const Eigen::ArrayXf> signal_map(signal_span.data(), signal_span.size());
        Eigen::Map<const Eigen::ArrayXf> win_map(window_span.data(), window_span.size());

        float stddev = std::sqrt((signal_map.square() * win_map).sum());

        // Use Eigen for vectorized computation of echo_density
        Eigen::ArrayXf mask = (signal_map.abs() > stddev).cast<float>();
        float echo_density = (mask * win_map).sum();

        // normalize
        echo_density /= kErfc;

        results.sparse_indices.push_back(n);
        results.echo_densities.push_back(echo_density);
    }

    // Estimate mixing time as the time when echo density first exceeds 1.0
    for (size_t i = 0; i < results.echo_densities.size(); ++i)
    {
        constexpr float kDensityThreshold = 1.0f;
        if (results.echo_densities[i] >= kDensityThreshold)
        {
            float time_gt_thresh =
                static_cast<float>(results.sparse_indices[i]) / static_cast<float>(options.sample_rate);
            float prev_time =
                (i == 0) ? 0.0f
                         : static_cast<float>(results.sparse_indices[i - 1]) / static_cast<float>(options.sample_rate);
            // Linear interpolation to estimate more accurate mixing time
            float previous_density = (i == 0) ? 0.0f : results.echo_densities[i - 1];
            float frac = (kDensityThreshold - previous_density) / (results.echo_densities[i] - previous_density);
            results.mixing_time = prev_time + frac * (time_gt_thresh - prev_time);
            break;
        }
    }

    return results;
}

} // namespace audio_utils::analysis