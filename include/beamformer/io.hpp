#pragma once

#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"

#include <filesystem>

namespace beamformer {

PackedVoltage read_packed_voltage(const std::filesystem::path& path,
                                  const Dimensions& dims);
void write_packed_voltage(const std::filesystem::path& path,
                          const PackedVoltage& voltage,
                          const Dimensions& dims);

Weights read_weights(const std::filesystem::path& path, const Dimensions& dims);
void write_weights(const std::filesystem::path& path, const Weights& weights,
                   const Dimensions& dims);

void write_intensities(const std::filesystem::path& path,
                       const Intensities& intensities,
                       const Dimensions& dims);

} // namespace beamformer
