#pragma once

#include "beamformer/config.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <vector>

namespace beamformer {

using Vec3 = std::array<float, 3>;

std::vector<Vec3> regular_array(std::size_t rows, std::size_t columns,
                                float spacing_m = 1.0F);
std::vector<Vec3> default_positions(std::size_t n_ant, float spacing_m = 1.0F);
std::vector<float> constant_frequencies(
    std::size_t n_freq = default_frequency_channels,
    float frequency_hz = default_frequency_hz);
Vec3 direction_from_lm(float l, float m);
std::vector<Vec3> default_beam_grid(std::size_t n_beams, float l_step = 0.02F,
                                    float m = 0.0F);

// Text files accept whitespace or commas and optional comments starting with '#'.
std::vector<Vec3> load_positions(const std::filesystem::path& path,
                                 std::size_t expected_count);
std::vector<float> load_frequencies(const std::filesystem::path& path,
                                    std::size_t expected_count);
std::vector<Vec3> load_directions(const std::filesystem::path& path,
                                  std::size_t expected_count);

} // namespace beamformer
