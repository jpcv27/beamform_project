#pragma once

#include "beamformer/config.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/int4.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace beamformer {

using PackedVoltage = std::vector<std::uint8_t>;

PackedVoltage make_one_hot(const Dimensions& dims, std::size_t active_time,
                           std::size_t active_frequency, std::size_t active_element,
                           ComplexInt4 value = ComplexInt4{3, -2});

PackedVoltage make_constant(const Dimensions& dims,
                            ComplexInt4 value = ComplexInt4{1, 0});

PackedVoltage make_point_source(const Dimensions& dims,
                                const std::vector<Vec3>& positions_m,
                                const std::vector<float>& frequencies_hz,
                                const Vec3& source_direction,
                                float amplitude = 4.0F);

PackedVoltage make_noise(const Dimensions& dims, std::uint32_t seed = 1);

void write_packed_voltage(const std::filesystem::path& path,
                          const PackedVoltage& voltage,
                          const Dimensions& dims);

} // namespace beamformer
