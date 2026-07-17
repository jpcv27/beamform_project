#pragma once

#include "beamformer/config.hpp"

#include <cstddef>

namespace beamformer {

// Physical rfsocHandlerShuffle output: packed[time][frequency][element].
// The handler writes one packed complex byte per element, so element is the
// fastest-changing dimension.
constexpr std::size_t voltage_index(const std::size_t t, const std::size_t f,
                                    const std::size_t element,
                                    const Dimensions& dims) {
    return (t * dims.n_freq + f) * dims.n_ant + element;
}

// rfsocHandlerShuffle reverses the two 32-element RFSoC blocks in the output.
// With two devices: RFSoC 1 -> E[0..31], RFSoC 0 -> E[32..63].
constexpr std::size_t rfsoc_output_element(const std::size_t rfsoc_id,
                                           const std::size_t packet_element,
                                           const std::size_t num_rfsocs = 2,
                                           const std::size_t elements_per_rfsoc =
                                               rfsoc_elements_per_device) {
    return (num_rfsocs - 1 - rfsoc_id) * elements_per_rfsoc + packet_element;
}

// The real deployment has 336 channels per NIC. The PoC stores both halves in
// one 672-channel file: NIC 0 -> [0,335], NIC 1 -> [336,671].
constexpr std::size_t full_band_frequency(const std::size_t nic_id,
                                          const std::size_t local_frequency) {
    return nic_id * rfsoc_channels_per_nic + local_frequency;
}

constexpr std::size_t weight_index(const std::size_t b, const std::size_t f,
                                   const std::size_t a, const Dimensions& dims) {
    return (b * dims.n_freq + f) * dims.n_ant + a;
}

constexpr std::size_t intensity_index(const std::size_t t, const std::size_t f,
                                      const std::size_t b, const Dimensions& dims) {
    return (t * dims.n_freq + f) * dims.n_beams + b;
}

} // namespace beamformer
