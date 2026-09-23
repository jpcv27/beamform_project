#pragma once

#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/physics.hpp"
#include "beamformer/upchannelizer_contract.hpp"
#include "beamformer/upchannelizer_indexing.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace beamformer {

inline std::size_t raw_tfe_index(const std::size_t time, const std::size_t coarse_frequency,
                                 const std::size_t element,
                                 const RawVoltageDimensions& dims) {
    validate_raw_voltage_dimensions(dims);
    if (time >= dims.n_time || coarse_frequency >= dims.n_coarse_frequency
        || element >= dims.n_element) {
        throw std::out_of_range("raw TFE coordinate is outside its dimensions");
    }
    return (time * dims.n_coarse_frequency + coarse_frequency) * dims.n_element + element;
}

inline std::size_t raw_loss_mask_index(const std::size_t time,
                                       const std::size_t coarse_frequency,
                                       const RawVoltageDimensions& dims) {
    validate_raw_voltage_dimensions(dims);
    if (time >= dims.n_time || coarse_frequency >= dims.n_coarse_frequency) {
        throw std::out_of_range("raw loss-mask coordinate is outside its dimensions");
    }
    return time * dims.n_coarse_frequency + coarse_frequency;
}

inline LossMask make_upchannelizer_loss_mask(const RawVoltageDimensions& dims,
                                             const std::uint32_t seed = 1,
                                             const float loss_probability = 0.0F) {
    validate_raw_voltage_dimensions(dims);
    if (!std::isfinite(loss_probability) || loss_probability < 0.0F
        || loss_probability > 1.0F) {
        throw std::invalid_argument("loss probability must be in [0, 1]");
    }
    LossMask mask(dims.n_time * dims.n_coarse_frequency, 1U);
    std::mt19937 random(seed);
    std::bernoulli_distribution lost(loss_probability);
    for (auto& valid : mask) {
        valid = lost(random) ? 0U : 1U;
    }
    return mask;
}

inline PackedVoltage make_upchannelizer_one_hot(const RawVoltageDimensions& dims,
                                                const std::size_t active_time,
                                                const std::size_t active_frequency,
                                                const std::size_t active_element,
                                                const ComplexInt4 value = {3, -2}) {
    PackedVoltage result(raw_voltage_sample_count(dims), pack_complex_int4(0, 0));
    result[raw_tfe_index(active_time, active_frequency, active_element, dims)] =
        pack_complex_int4(value.real, value.imag);
    return result;
}

inline PackedVoltage make_upchannelizer_constant(const RawVoltageDimensions& dims,
                                                 const ComplexInt4 value = {1, 0}) {
    return PackedVoltage(raw_voltage_sample_count(dims), pack_complex_int4(value.real, value.imag));
}

inline PackedVoltage make_upchannelizer_noise(const RawVoltageDimensions& dims,
                                              const std::uint32_t seed = 1) {
    PackedVoltage result(raw_voltage_sample_count(dims));
    std::mt19937 random(seed);
    std::uniform_int_distribution<int> component(-7, 7);
    for (auto& packed : result) {
        packed = pack_complex_int4(static_cast<std::int8_t>(component(random)),
                                   static_cast<std::int8_t>(component(random)));
    }
    return result;
}

// bin_offset is expressed in fine-bin spacings relative to the coarse-channel
// center. Exact output bin u therefore uses u - 15.5. A non-integral value is
// an off-bin tone. The temporal phase advances by 2*pi*bin_offset/U per raw
// sample, matching the forward PFB convention in upchannelizer_contract.hpp.
struct UpchannelizerTone {
    std::size_t coarse_frequency = 0;
    double bin_offset = 0.0;
    float amplitude = 1.0F;
    double phase_radians = 0.0;
    double antenna_phase_radians = 0.0;
    std::int64_t start_time = 0;
};

inline UpchannelizerTone exact_upchannelizer_bin_tone(const std::size_t coarse_frequency,
                                                       const std::size_t fine_bin,
                                                       const float amplitude = 1.0F,
                                                       const double phase_radians = 0.0,
                                                       const double antenna_phase_radians = 0.0,
                                                       const std::int64_t start_time = 0) {
    validate_upchannelized_bin(fine_bin);
    return {coarse_frequency, static_cast<double>(upchannelized_bin_offset(fine_bin)), amplitude,
            phase_radians, antenna_phase_radians, start_time};
}

inline double upchannelizer_tone_phase_advance(const UpchannelizerTone& tone) {
    return two_pi * tone.bin_offset / static_cast<double>(upchannelization_factor);
}

inline void validate_upchannelizer_tone(const UpchannelizerTone& tone,
                                        const RawVoltageDimensions& dims) {
    validate_raw_voltage_dimensions(dims);
    if (tone.coarse_frequency >= dims.n_coarse_frequency) {
        throw std::out_of_range("tone coarse frequency is outside raw dimensions");
    }
    if (!std::isfinite(tone.bin_offset)
        || tone.bin_offset < -static_cast<double>(upchannelization_factor) * 0.5
        || tone.bin_offset > static_cast<double>(upchannelization_factor) * 0.5) {
        throw std::invalid_argument("tone bin offset must be within the coarse-channel Nyquist range");
    }
    if (!std::isfinite(tone.amplitude) || tone.amplitude <= 0.0F || tone.amplitude > 7.0F) {
        throw std::invalid_argument("tone amplitude must be in (0, 7]");
    }
    if (!std::isfinite(tone.phase_radians) || !std::isfinite(tone.antenna_phase_radians)) {
        throw std::invalid_argument("tone phases must be finite");
    }
}

inline std::int8_t quantize_upchannelizer_component(const double value) {
    const auto rounded = std::lround(value);
    if (rounded < -8 || rounded > 7) {
        throw std::invalid_argument("synthetic tone exceeds the int4 range after quantization");
    }
    return static_cast<std::int8_t>(rounded);
}

inline PackedVoltage make_upchannelizer_tones(const RawVoltageDimensions& dims,
                                              const std::vector<UpchannelizerTone>& tones,
                                              const std::int64_t raw_time_origin = 0) {
    validate_raw_voltage_dimensions(dims);
    if (tones.empty()) {
        throw std::invalid_argument("at least one synthetic tone is required");
    }
    std::vector<double> amplitude_bounds(dims.n_coarse_frequency, 0.0);
    for (const auto& tone : tones) {
        validate_upchannelizer_tone(tone, dims);
        amplitude_bounds[tone.coarse_frequency] += tone.amplitude;
    }
    for (const auto bound : amplitude_bounds) {
        if (bound > 7.0) {
            throw std::invalid_argument("multi-tone fixture can exceed the int4 range");
        }
    }

    PackedVoltage result(raw_voltage_sample_count(dims), pack_complex_int4(0, 0));
    for (std::size_t time = 0; time < dims.n_time; ++time) {
        const auto absolute_time = raw_time_origin + static_cast<std::int64_t>(time);
        for (std::size_t coarse_frequency = 0; coarse_frequency < dims.n_coarse_frequency;
             ++coarse_frequency) {
            for (std::size_t element = 0; element < dims.n_element; ++element) {
                double real = 0.0;
                double imag = 0.0;
                for (const auto& tone : tones) {
                    if (tone.coarse_frequency != coarse_frequency || absolute_time < tone.start_time) {
                        continue;
                    }
                    const auto phase = tone.phase_radians
                                       + static_cast<double>(element) * tone.antenna_phase_radians
                                       + static_cast<double>(absolute_time)
                                             * upchannelizer_tone_phase_advance(tone);
                    real += static_cast<double>(tone.amplitude) * std::cos(phase);
                    imag += static_cast<double>(tone.amplitude) * std::sin(phase);
                }
                result[raw_tfe_index(time, coarse_frequency, element, dims)] =
                    pack_complex_int4(quantize_upchannelizer_component(real),
                                      quantize_upchannelizer_component(imag));
            }
        }
    }
    return result;
}

struct UpchannelizerPointSource {
    std::size_t coarse_frequency = 0;
    double bin_offset = 0.0;
    double coarse_frequency_center_hz = default_frequency_start_hz;
    double coarse_channel_width_hz = default_channel_width_hz;
    float amplitude = 1.0F;
    double phase_radians = 0.0;
    std::int64_t start_time = 0;
};

inline PackedVoltage make_upchannelizer_point_source(
    const RawVoltageDimensions& dims, const UpchannelizerPointSource& source,
    const std::vector<Vec3>& positions_m, const Vec3& source_direction,
    const std::int64_t raw_time_origin = 0) {
    UpchannelizerTone temporal{source.coarse_frequency, source.bin_offset, source.amplitude,
                                source.phase_radians, 0.0, source.start_time};
    validate_upchannelizer_tone(temporal, dims);
    if (positions_m.size() != dims.n_element) {
        throw std::invalid_argument("point-source position count must match raw elements");
    }
    if (!std::isfinite(source.coarse_frequency_center_hz)
        || !std::isfinite(source.coarse_channel_width_hz)
        || source.coarse_frequency_center_hz <= 0.0 || source.coarse_channel_width_hz == 0.0) {
        throw std::invalid_argument("point-source frequency grid is invalid");
    }
    double direction_norm_squared = 0.0;
    for (const auto component : source_direction) {
        if (!std::isfinite(component)) {
            throw std::invalid_argument("point-source direction must be finite");
        }
        direction_norm_squared += static_cast<double>(component) * component;
    }
    if (std::abs(direction_norm_squared - 1.0) > 1.0e-3) {
        throw std::invalid_argument("point-source direction must be a unit vector");
    }

    const double fine_frequency_hz = source.coarse_frequency_center_hz
                                     + (static_cast<double>(source.coarse_frequency)
                                        + source.bin_offset
                                              / static_cast<double>(upchannelization_factor))
                                           * source.coarse_channel_width_hz;
    PackedVoltage result(raw_voltage_sample_count(dims), pack_complex_int4(0, 0));
    for (std::size_t time = 0; time < dims.n_time; ++time) {
        const auto absolute_time = raw_time_origin + static_cast<std::int64_t>(time);
        if (absolute_time < source.start_time) {
            continue;
        }
        for (std::size_t element = 0; element < dims.n_element; ++element) {
            const auto& position = positions_m[element];
            const double path_m = static_cast<double>(position[0]) * source_direction[0]
                                  + static_cast<double>(position[1]) * source_direction[1]
                                  + static_cast<double>(position[2]) * source_direction[2];
            const double spatial_phase = -two_pi * fine_frequency_hz * path_m / speed_of_light_m_per_s;
            const double phase = source.phase_radians
                                 + static_cast<double>(absolute_time)
                                       * upchannelizer_tone_phase_advance(temporal)
                                 + spatial_phase;
            result[raw_tfe_index(time, source.coarse_frequency, element, dims)] =
                pack_complex_int4(quantize_upchannelizer_component(source.amplitude * std::cos(phase)),
                                  quantize_upchannelizer_component(source.amplitude * std::sin(phase)));
        }
    }
    return result;
}

inline void validate_upchannelizer_packed_shards(const PackedShardSet& shards,
                                                 const RawVoltageDimensions& dims) {
    validate_raw_voltage_dimensions(dims);
    if (dims.n_coarse_frequency != local_frequency_channels) {
        throw std::invalid_argument("two-shard fixtures require exactly 336 coarse channels per shard");
    }
    for (std::size_t shard_id = 0; shard_id < frequency_shard_count; ++shard_id) {
        const auto& shard = shards[shard_id];
        validate_shard_descriptor(shard.descriptor);
        if (shard.descriptor.shard_id != shard_id
            || shard.payload.size() != raw_voltage_sample_count(dims)
            || shard.loss_mask.size() != dims.n_time * dims.n_coarse_frequency) {
            throw std::invalid_argument("invalid upchannelizer packed shard fixture");
        }
    }
    if (shards[0].payload.data() == shards[1].payload.data()
        || shards[0].loss_mask.data() == shards[1].loss_mask.data()
        || shards[0].descriptor.loss_mask_id == shards[1].descriptor.loss_mask_id) {
        throw std::invalid_argument("upchannelizer shard fixtures must remain independent");
    }
}

inline PackedShardSet make_two_shard_upchannelizer_tones(
    const RawVoltageDimensions& dims,
    const std::array<std::vector<UpchannelizerTone>, frequency_shard_count>& tones,
    const std::array<std::uint32_t, frequency_shard_count> loss_seeds = {1, 2},
    const float loss_probability = 0.0F,
    const std::int64_t raw_time_origin = 0) {
    if (dims.n_coarse_frequency != local_frequency_channels) {
        throw std::invalid_argument("two-shard fixtures require exactly 336 coarse channels per shard");
    }
    const auto descriptors = default_shard_descriptors();
    PackedShardSet shards{};
    for (std::size_t shard_id = 0; shard_id < frequency_shard_count; ++shard_id) {
        shards[shard_id] = {descriptors[shard_id],
                            make_upchannelizer_tones(dims, tones[shard_id], raw_time_origin),
                            make_upchannelizer_loss_mask(dims, loss_seeds[shard_id], loss_probability)};
    }
    validate_upchannelizer_packed_shards(shards, dims);
    return shards;
}

} // namespace beamformer
