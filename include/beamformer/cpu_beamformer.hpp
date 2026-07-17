#pragma once

#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"

namespace beamformer {

ComplexVoltage unpack_voltage(const PackedVoltage& packed,
                              const Dimensions& dims);

Intensities cpu_beamform_intensity(const ComplexVoltage& voltage,
                                   const Weights& weights,
                                   const Dimensions& dims);

} // namespace beamformer
