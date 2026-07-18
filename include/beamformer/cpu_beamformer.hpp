#pragma once

#include "beamformer/config.hpp"
#include "beamformer/formats.hpp"

namespace beamformer {

ComplexVoltage unpack_voltage(const PackedVoltage& packed,
                              const Dimensions& dims);

Intensities cpu_beamform_intensity(const ComplexVoltage& voltage,
                                   const Weights& weights,
                                   const Dimensions& dims);

// Benchmark-oriented variant that writes into a preallocated output. Voltage
// may contain additional time samples after the prefix described by dims.
void cpu_beamform_intensity_into(const ComplexVoltage& voltage,
                                 const Weights& weights,
                                 const Dimensions& dims,
                                 Intensities& intensity);

} // namespace beamformer
