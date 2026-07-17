#pragma once

#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/int4.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace beamformer {

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

} // namespace beamformer
