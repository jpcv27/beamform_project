#pragma once

#include "beamformer/complex.hpp"
#include "beamformer/config.hpp"
#include "beamformer/upchannelizer_contract.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace beamformer {

// This type intentionally does not accept beamformer::Dimensions. The latter
// describes the direct beamformer input (336 coarse channels), whereas this
// pair of shapes represents the raw and channelized sides of a distinct stage.
struct UpchannelizerConfig {
    std::size_t u = upchannelization_factor;
    std::size_t taps_per_phase = pfb_taps_per_phase;
    std::size_t prototype_length = pfb_prototype_length;
};

inline void validate_upchannelizer_config(const UpchannelizerConfig& config) {
    if (config.u != upchannelization_factor || config.taps_per_phase != pfb_taps_per_phase
        || config.prototype_length != pfb_prototype_length) {
        throw std::invalid_argument("the upchannelizer contract requires U=32, M=4, and L=128");
    }
    if (config.prototype_length != config.u * config.taps_per_phase) {
        throw std::invalid_argument("PFB prototype length must equal U times taps per phase");
    }
}

struct RawVoltageDimensions {
    std::size_t n_time = 0;
    std::size_t n_coarse_frequency = 0;
    std::size_t n_element = 0;
};

struct ChannelizedVoltageDimensions {
    std::size_t n_time = 0;
    std::size_t n_coarse_frequency = 0;
    std::size_t n_fine_bin = 0;
    std::size_t n_element = 0;
};

inline void validate_raw_voltage_dimensions(const RawVoltageDimensions& dims) {
    if (dims.n_time == 0 || dims.n_coarse_frequency == 0 || dims.n_element == 0) {
        throw std::invalid_argument("raw upchannelizer dimensions must be positive");
    }
}

inline void validate_channelized_voltage_dimensions(const ChannelizedVoltageDimensions& dims,
                                                    const UpchannelizerConfig& config = {}) {
    validate_upchannelizer_config(config);
    if (dims.n_time == 0 || dims.n_coarse_frequency == 0 || dims.n_element == 0) {
        throw std::invalid_argument("channelized voltage dimensions must be positive");
    }
    if (dims.n_fine_bin != config.u) {
        throw std::invalid_argument("channelized voltage dimensions must have exactly 32 fine bins");
    }
}

inline ChannelizedVoltageDimensions channelized_dimensions(
    const RawVoltageDimensions& raw, const UpchannelizerConfig& config = {}) {
    validate_raw_voltage_dimensions(raw);
    validate_upchannelizer_config(config);
    if (raw.n_time % config.u != 0) {
        throw std::invalid_argument("raw time length must be divisible by U for a complete frame");
    }
    return {raw.n_time / config.u, raw.n_coarse_frequency, config.u, raw.n_element};
}

inline std::size_t checked_product(const std::size_t lhs, const std::size_t rhs,
                                   const char* const description) {
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        throw std::overflow_error(description);
    }
    return lhs * rhs;
}

inline std::size_t raw_voltage_sample_count(const RawVoltageDimensions& dims) {
    validate_raw_voltage_dimensions(dims);
    return checked_product(checked_product(dims.n_time, dims.n_coarse_frequency,
                                           "raw voltage sample count overflows size_t"),
                           dims.n_element, "raw voltage sample count overflows size_t");
}

inline std::size_t channelized_voltage_count(const ChannelizedVoltageDimensions& dims,
                                             const UpchannelizerConfig& config = {}) {
    validate_channelized_voltage_dimensions(dims, config);
    return checked_product(
        checked_product(checked_product(dims.n_time, dims.n_coarse_frequency,
                                        "channelized voltage count overflows size_t"),
                        dims.n_fine_bin, "channelized voltage count overflows size_t"),
        dims.n_element, "channelized voltage count overflows size_t");
}

inline std::size_t channelized_voltage_bytes(const ChannelizedVoltageDimensions& dims,
                                             const UpchannelizerConfig& config = {}) {
    return checked_product(channelized_voltage_count(dims, config), sizeof(ComplexFloat),
                           "channelized voltage byte count overflows size_t");
}

// One validity byte applies to every element and fine bin of a channelized
// spectrum [time_out][coarse_frequency]. A zero marks an invalid spectrum.
using ChannelizedValidityMask = std::vector<std::uint8_t>;

inline std::size_t channelized_validity_count(const ChannelizedVoltageDimensions& dims,
                                              const UpchannelizerConfig& config = {}) {
    validate_channelized_voltage_dimensions(dims, config);
    return checked_product(dims.n_time, dims.n_coarse_frequency,
                           "channelized validity count overflows size_t");
}

inline std::size_t flattened_fine_frequency_count(const ChannelizedVoltageDimensions& dims,
                                                   const UpchannelizerConfig& config = {}) {
    validate_channelized_voltage_dimensions(dims, config);
    return checked_product(dims.n_coarse_frequency, dims.n_fine_bin,
                           "flattened fine-frequency count overflows size_t");
}

inline void validate_channelized_coordinates(const std::size_t time_out,
                                             const std::size_t coarse_frequency,
                                             const std::size_t fine_bin,
                                             const std::size_t element,
                                             const ChannelizedVoltageDimensions& dims,
                                             const UpchannelizerConfig& config = {}) {
    validate_channelized_voltage_dimensions(dims, config);
    if (time_out >= dims.n_time || coarse_frequency >= dims.n_coarse_frequency
        || fine_bin >= dims.n_fine_bin || element >= dims.n_element) {
        throw std::out_of_range("channelized voltage coordinate is outside its dimensions");
    }
}

// Canonical channelized layout: [time_out][coarse_frequency][fine_bin][element] (TFUE).
// Element is contiguous; adjacent fine bins, then coarse channels, then output times follow.
inline std::size_t channelized_index(const std::size_t time_out,
                                     const std::size_t coarse_frequency,
                                     const std::size_t fine_bin,
                                     const std::size_t element,
                                     const ChannelizedVoltageDimensions& dims,
                                     const UpchannelizerConfig& config = {}) {
    validate_channelized_coordinates(time_out, coarse_frequency, fine_bin, element, dims, config);
    return (((time_out * dims.n_coarse_frequency + coarse_frequency) * dims.n_fine_bin + fine_bin)
            * dims.n_element
            + element);
}

inline std::size_t flattened_fine_frequency(const std::size_t coarse_frequency,
                                            const std::size_t fine_bin,
                                            const ChannelizedVoltageDimensions& dims,
                                            const UpchannelizerConfig& config = {}) {
    validate_channelized_voltage_dimensions(dims, config);
    if (coarse_frequency >= dims.n_coarse_frequency || fine_bin >= dims.n_fine_bin) {
        throw std::out_of_range("fine-frequency coordinate is outside its dimensions");
    }
    return coarse_frequency * dims.n_fine_bin + fine_bin;
}

inline std::size_t coarse_frequency_from_flattened(const std::size_t fine_frequency,
                                                    const ChannelizedVoltageDimensions& dims,
                                                    const UpchannelizerConfig& config = {}) {
    const auto count = flattened_fine_frequency_count(dims, config);
    if (fine_frequency >= count) {
        throw std::out_of_range("flattened fine-frequency index is outside its dimensions");
    }
    return fine_frequency / dims.n_fine_bin;
}

inline std::size_t fine_bin_from_flattened(const std::size_t fine_frequency,
                                           const ChannelizedVoltageDimensions& dims,
                                           const UpchannelizerConfig& config = {}) {
    const auto count = flattened_fine_frequency_count(dims, config);
    if (fine_frequency >= count) {
        throw std::out_of_range("flattened fine-frequency index is outside its dimensions");
    }
    return fine_frequency % dims.n_fine_bin;
}

enum class ChannelizedLayout { TFUE };
enum class UpchannelizerPrototype { KotekanSincHanning };
enum class UpchannelizerFftDirection { Forward };
enum class UpchannelizerFftNormalization { Unnormalized };
enum class FineBinOrder { CenteredHalfBin };
enum class UpchannelizerHistoryMode { FrameLocalZeroPadded };

struct CoarseFrequencyGrid {
    // Center frequency of absolute coarse channel zero, not an edge frequency.
    double first_coarse_center_hz = default_frequency_start_hz;
    double coarse_channel_width_hz = default_channel_width_hz;
};

struct ChannelizedShardMetadata {
    ShardDescriptor shard;
    RawVoltageDimensions raw_dimensions;
    ChannelizedVoltageDimensions channelized_dimensions;
    UpchannelizerConfig config;
    CoarseFrequencyGrid frequency_grid;
    ChannelizedLayout layout = ChannelizedLayout::TFUE;
    UpchannelizerPrototype prototype = UpchannelizerPrototype::KotekanSincHanning;
    UpchannelizerFftDirection fft_direction = UpchannelizerFftDirection::Forward;
    UpchannelizerFftNormalization fft_normalization = UpchannelizerFftNormalization::Unnormalized;
    FineBinOrder fine_bin_order = FineBinOrder::CenteredHalfBin;
    UpchannelizerHistoryMode history_mode = UpchannelizerHistoryMode::FrameLocalZeroPadded;
};

inline void validate_channelized_shard_metadata(const ChannelizedShardMetadata& metadata) {
    validate_shard_descriptor(metadata.shard);
    validate_raw_voltage_dimensions(metadata.raw_dimensions);
    validate_upchannelizer_config(metadata.config);
    const auto expected = channelized_dimensions(metadata.raw_dimensions, metadata.config);
    validate_channelized_voltage_dimensions(metadata.channelized_dimensions, metadata.config);
    if (metadata.raw_dimensions.n_coarse_frequency != metadata.shard.local_frequency_count
        || metadata.channelized_dimensions.n_coarse_frequency
               != metadata.shard.local_frequency_count) {
        throw std::invalid_argument("channelized metadata must describe exactly one shard");
    }
    if (metadata.channelized_dimensions.n_time != expected.n_time
        || metadata.channelized_dimensions.n_fine_bin != expected.n_fine_bin
        || metadata.channelized_dimensions.n_element != expected.n_element) {
        throw std::invalid_argument("channelized dimensions do not match the raw dimensions");
    }
    if (!std::isfinite(metadata.frequency_grid.first_coarse_center_hz)
        || !std::isfinite(metadata.frequency_grid.coarse_channel_width_hz)
        || metadata.frequency_grid.coarse_channel_width_hz == 0.0) {
        throw std::invalid_argument("channelized frequency-grid metadata is invalid");
    }
}

inline double channelized_frequency_hz(const ChannelizedShardMetadata& metadata,
                                       const std::size_t coarse_frequency,
                                       const std::size_t fine_bin) {
    validate_channelized_shard_metadata(metadata);
    if (coarse_frequency >= metadata.channelized_dimensions.n_coarse_frequency
        || fine_bin >= metadata.channelized_dimensions.n_fine_bin) {
        throw std::out_of_range("channelized frequency coordinate is outside its dimensions");
    }
    const auto absolute_coarse_frequency = metadata.shard.absolute_frequency_start + coarse_frequency;
    return metadata.frequency_grid.first_coarse_center_hz
           + (static_cast<double>(absolute_coarse_frequency)
              + static_cast<double>(upchannelized_bin_fraction(fine_bin)))
                 * metadata.frequency_grid.coarse_channel_width_hz;
}

struct ChannelizedVoltageView {
    ComplexFloat* data = nullptr;
    ChannelizedVoltageDimensions dimensions;

    ComplexFloat& at(const std::size_t time_out, const std::size_t coarse_frequency,
                     const std::size_t fine_bin, const std::size_t element) const {
        if (data == nullptr) {
            throw std::invalid_argument("channelized voltage view has no data");
        }
        return data[channelized_index(time_out, coarse_frequency, fine_bin, element, dimensions)];
    }
};

struct ConstChannelizedVoltageView {
    const ComplexFloat* data = nullptr;
    ChannelizedVoltageDimensions dimensions;

    const ComplexFloat& at(const std::size_t time_out, const std::size_t coarse_frequency,
                           const std::size_t fine_bin, const std::size_t element) const {
        if (data == nullptr) {
            throw std::invalid_argument("channelized voltage view has no data");
        }
        return data[channelized_index(time_out, coarse_frequency, fine_bin, element, dimensions)];
    }
};

} // namespace beamformer
