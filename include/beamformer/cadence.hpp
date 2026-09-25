#pragma once

#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace beamformer {

// Nominal spectrum period in microseconds: 10/3 us per spectrum.
constexpr double spectrum_period_us = 10.0 / 3.0;
// Nominal spectrum period in milliseconds: 10/3000 ms.
constexpr double spectrum_period_ms = spectrum_period_us / 1000.0;

// Kotekan frame duration in milliseconds for a given time count:
// n_time * (10/3 us) / 1000. For default n_time = 15360, this is exactly 51.2 ms.
constexpr double frame_deadline_ms(const std::size_t n_time) {
    return static_cast<double>(n_time) * spectrum_period_ms;
}

// Canonical production Kotekan frame deadline for n_time = 15360.
constexpr double default_frame_deadline_ms = 51.2;

struct CadenceVerdict {
    double deadline_ms = 0.0;
    double observed_ms = 0.0;
    bool passed = false;
    double margin_ms = 0.0;       // deadline_ms - observed_ms
    double duty_cycle_pct = 0.0;   // (observed_ms / deadline_ms) * 100.0
    double realtime_factor = 0.0;  // deadline_ms / observed_ms
};

inline CadenceVerdict evaluate_cadence(const double observed_ms, const double deadline_ms) {
    CadenceVerdict verdict;
    verdict.deadline_ms = deadline_ms;
    verdict.observed_ms = observed_ms;
    verdict.passed = (observed_ms <= deadline_ms);
    verdict.margin_ms = deadline_ms - observed_ms;
    verdict.duty_cycle_pct = (deadline_ms > 0.0) ? (observed_ms / deadline_ms) * 100.0 : 0.0;
    verdict.realtime_factor = (observed_ms > 0.0) ? (deadline_ms / observed_ms) : 0.0;
    return verdict;
}

struct Statistics {
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double stddev = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
};

inline Statistics compute_statistics(std::vector<double> values) {
    if (values.empty()) {
        return {};
    }
    std::sort(values.begin(), values.end());
    Statistics stats;
    stats.min = values.front();
    stats.max = values.back();

    double sum = 0.0;
    for (const double v : values) {
        sum += v;
    }
    stats.mean = sum / static_cast<double>(values.size());

    const std::size_t n = values.size();
    if (n % 2 == 1) {
        stats.median = values[n / 2];
    } else {
        stats.median = 0.5 * (values[n / 2 - 1] + values[n / 2]);
    }

    double variance_sum = 0.0;
    for (const double v : values) {
        variance_sum += (v - stats.mean) * (v - stats.mean);
    }
    stats.stddev = std::sqrt(variance_sum / static_cast<double>(values.size()));

    auto percentile = [&](const double p) {
        if (n == 1) {
            return values[0];
        }
        const double rank = (p / 100.0) * static_cast<double>(n - 1);
        const std::size_t low = static_cast<std::size_t>(rank);
        const std::size_t high = std::min(low + 1, n - 1);
        const double weight = rank - static_cast<double>(low);
        return values[low] * (1.0 - weight) + values[high] * weight;
    };
    stats.p95 = percentile(95.0);
    stats.p99 = percentile(99.0);
    return stats;
}

// Workload and throughput helpers
constexpr double real_flops_per_complex_mac = 8.0;
constexpr double real_flops_per_intensity = 3.0;

constexpr double calculate_cmac_count(const Dimensions& dims, const std::size_t shard_count = 1) {
    return static_cast<double>(dims.n_time) * static_cast<double>(dims.n_freq)
           * static_cast<double>(dims.n_beams) * static_cast<double>(dims.n_ant)
           * static_cast<double>(shard_count);
}

constexpr double calculate_estimated_flops(const Dimensions& dims, const std::size_t shard_count = 1) {
    const double cmac = calculate_cmac_count(dims, shard_count);
    const double intensity_ops = static_cast<double>(dims.n_time)
                                 * static_cast<double>(dims.n_freq)
                                 * static_cast<double>(dims.n_beams)
                                 * static_cast<double>(shard_count);
    return real_flops_per_complex_mac * cmac + real_flops_per_intensity * intensity_ops;
}

} // namespace beamformer
