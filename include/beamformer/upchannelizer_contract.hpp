#pragma once

#include "beamformer/complex.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace beamformer {

// Kotekan-aligned spectral PFB contract. This describes only the numerical
// channelizer; it deliberately does not import Kotekan buffer, gain, fp16, or
// output-quantization conventions.
inline constexpr std::size_t upchannelization_factor = 32;
inline constexpr std::size_t pfb_taps_per_phase = 4;
inline constexpr std::size_t pfb_prototype_length =
    upchannelization_factor * pfb_taps_per_phase;
inline constexpr std::size_t pfb_history_samples =
    (pfb_taps_per_phase - 1) * upchannelization_factor;
inline constexpr std::size_t pfb_history_spectra = pfb_taps_per_phase - 1;

inline void validate_pfb_tap(const std::size_t tap) {
    if (tap >= pfb_prototype_length) {
        throw std::out_of_range("PFB tap is outside the 128-coefficient prototype");
    }
}

inline void validate_upchannelized_bin(const std::size_t fine_bin) {
    if (fine_bin >= upchannelization_factor) {
        throw std::out_of_range("fine bin is outside the 32-point FFT");
    }
}

// This is Wkernel(s, M, U) from Kotekan's julia/kernels/upchan.jl, followed
// by the Kotekan 1/U amplitude scaling. The PoC retains these coefficients as
// float32 rather than quantizing them to float16.
inline float sinc_hanning_pfb_weight(const std::size_t tap) {
    validate_pfb_tap(tap);
    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr double length = static_cast<double>(pfb_prototype_length);
    const double coordinate =
        (2.0 * static_cast<double>(tap) - (length - 1.0)) / (2.0 * (length + 1.0));
    const double argument = static_cast<double>(pfb_taps_per_phase) * coordinate;
    const double sinc = argument == 0.0 ? 1.0 : std::sin(pi * argument) / (pi * argument);
    const double hanning = std::cos(pi * coordinate) * std::cos(pi * coordinate);
    return static_cast<float>(hanning * sinc
                              / static_cast<double>(upchannelization_factor));
}

// Fine bin u maps to u - (U - 1)/2 coarse-bin-width fractions. For U=32 this
// is [-15.5, -14.5, ..., +14.5, +15.5]/32, with no implicit fftshift.
inline float upchannelized_bin_offset(const std::size_t fine_bin) {
    validate_upchannelized_bin(fine_bin);
    return static_cast<float>(fine_bin)
           - static_cast<float>(upchannelization_factor - 1) * 0.5F;
}

inline float upchannelized_bin_fraction(const std::size_t fine_bin) {
    return upchannelized_bin_offset(fine_bin)
           / static_cast<float>(upchannelization_factor);
}

// Forward DFT phase for the Kotekan centered-half-bin convention. The phase
// is expressed over the complete prototype tap so the alternating PFB-phase
// sign is part of one unambiguous reference operator.
inline ComplexFloat upchannelized_forward_phase(const std::size_t fine_bin,
                                                 const std::size_t prototype_tap) {
    validate_upchannelized_bin(fine_bin);
    validate_pfb_tap(prototype_tap);
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double angle = -2.0 * pi * static_cast<double>(upchannelized_bin_offset(fine_bin))
                         * static_cast<double>(prototype_tap)
                         / static_cast<double>(upchannelization_factor);
    return {static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle))};
}

// Output spectrum m consumes the causal 128-sample support
// [m*U - (M-1)*U, m*U + U - 1]. Negative indices use zero history in the
// frame-local baseline and make spectra m=0..M-2 history-incomplete.
inline std::int64_t raw_time_for_pfb_tap(const std::size_t output_time,
                                         const std::size_t prototype_tap) {
    validate_pfb_tap(prototype_tap);
    return static_cast<std::int64_t>(output_time * upchannelization_factor)
           - static_cast<std::int64_t>(pfb_history_samples)
           + static_cast<std::int64_t>(prototype_tap);
}

inline bool pfb_history_complete(const std::size_t output_time) {
    return output_time >= pfb_history_spectra;
}

} // namespace beamformer
