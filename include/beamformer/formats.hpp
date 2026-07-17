#pragma once

#include "beamformer/config.hpp"

#include <cstddef>

namespace beamformer {

constexpr std::size_t voltage_sample_count(const Dimensions& dims) {
    return dims.n_time * dims.n_freq * dims.n_ant;
}

constexpr std::size_t packed_voltage_bytes(const Dimensions& dims) {
    return voltage_sample_count(dims);
}

// little-endian float32 [beam][frequency][antenna][real, imag]
constexpr std::size_t weight_bytes(const Dimensions& dims) {
    return dims.n_beams * dims.n_freq * dims.n_ant * 2 * sizeof(float);
}

// little-endian float32 [time][frequency][beam]
constexpr std::size_t intensity_bytes(const Dimensions& dims) {
    return dims.n_time * dims.n_freq * dims.n_beams * sizeof(float);
}

} // namespace beamformer
