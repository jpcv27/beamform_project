#include "beamformer/synthetic_data.hpp"

#include "beamformer/formats.hpp"
#include "beamformer/indexing.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

namespace beamformer {
namespace {

constexpr double speed_of_light_m_per_s = 299'792'458.0;
constexpr double two_pi = 6.283185307179586476925286766559;

void validate_point_source_inputs(const Dimensions& dims,
                                  const std::vector<Vec3>& positions_m,
                                  const std::vector<float>& frequencies_hz,
                                  const Vec3& source_direction,
                                  const float amplitude) {
    validate_dimensions(dims);
    if (positions_m.size() != dims.n_ant) {
        throw std::invalid_argument("position count must match n_ant");
    }
    if (frequencies_hz.size() != dims.n_freq) {
        throw std::invalid_argument("frequency count must match n_freq");
    }
    if (!std::isfinite(amplitude) || amplitude <= 0.0F || amplitude > 7.0F) {
        throw std::invalid_argument("point-source amplitude must be in (0, 7]");
    }

    double norm_squared = 0.0;
    for (const float component : source_direction) {
        if (!std::isfinite(component)) {
            throw std::invalid_argument("source direction must be finite");
        }
        norm_squared += static_cast<double>(component) * component;
    }
    if (std::abs(norm_squared - 1.0) > 1.0e-3) {
        throw std::invalid_argument("source direction must be a unit vector");
    }
    for (const float frequency : frequencies_hz) {
        if (!std::isfinite(frequency) || frequency <= 0.0F) {
            throw std::invalid_argument("frequencies must be positive and finite");
        }
    }
}

} // namespace

PackedVoltage make_one_hot(const Dimensions& dims, const std::size_t active_time,
                           const std::size_t active_frequency,
                           const std::size_t active_element, const ComplexInt4 value) {
    validate_dimensions(dims);
    if (active_time >= dims.n_time || active_frequency >= dims.n_freq
        || active_element >= dims.n_ant) {
        throw std::out_of_range("one-hot index is outside the voltage dimensions");
    }

    PackedVoltage voltage(voltage_sample_count(dims), pack_complex_int4(0, 0));
    voltage[voltage_index(active_time, active_frequency, active_element, dims)] =
        pack_complex_int4(value.real, value.imag);
    return voltage;
}

PackedVoltage make_constant(const Dimensions& dims, const ComplexInt4 value) {
    validate_dimensions(dims);
    return PackedVoltage(voltage_sample_count(dims),
                         pack_complex_int4(value.real, value.imag));
}

PackedVoltage make_point_source(const Dimensions& dims,
                                const std::vector<Vec3>& positions_m,
                                const std::vector<float>& frequencies_hz,
                                const Vec3& source_direction,
                                const float amplitude) {
    validate_point_source_inputs(dims, positions_m, frequencies_hz, source_direction,
                                 amplitude);

    const std::size_t spectrum_size = dims.n_freq * dims.n_ant;
    PackedVoltage spectrum(spectrum_size);
    for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
        const double wave_number = two_pi * static_cast<double>(frequencies_hz[frequency])
                                   / speed_of_light_m_per_s;
        for (std::size_t element = 0; element < dims.n_ant; ++element) {
            const auto& position = positions_m[element];
            const double delay_m =
                static_cast<double>(position[0]) * source_direction[0]
                + static_cast<double>(position[1]) * source_direction[1]
                + static_cast<double>(position[2]) * source_direction[2];
            const double phase = wave_number * delay_m;
            const auto real = static_cast<std::int8_t>(
                std::lround(static_cast<double>(amplitude) * std::cos(phase)));
            const auto imag = static_cast<std::int8_t>(
                std::lround(-static_cast<double>(amplitude) * std::sin(phase)));
            spectrum[frequency * dims.n_ant + element] = pack_complex_int4(real, imag);
        }
    }

    PackedVoltage voltage(voltage_sample_count(dims));
    for (std::size_t time = 0; time < dims.n_time; ++time) {
        std::copy(spectrum.begin(), spectrum.end(),
                  voltage.begin() + static_cast<std::ptrdiff_t>(time * spectrum_size));
    }
    return voltage;
}

PackedVoltage make_noise(const Dimensions& dims, const std::uint32_t seed) {
    validate_dimensions(dims);
    PackedVoltage voltage(voltage_sample_count(dims));
    std::mt19937 random(seed);
    for (auto& packed : voltage) {
        const std::uint32_t bits = random();
        const auto real = static_cast<std::int8_t>(static_cast<int>(bits & 0xFFFFU) % 15 - 7);
        const auto imag =
            static_cast<std::int8_t>(static_cast<int>((bits >> 16) & 0xFFFFU) % 15 - 7);
        packed = pack_complex_int4(real, imag);
    }
    return voltage;
}

void write_packed_voltage(const std::filesystem::path& path,
                          const PackedVoltage& voltage,
                          const Dimensions& dims) {
    validate_dimensions(dims);
    if (voltage.size() != packed_voltage_bytes(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open output file: " + path.string());
    }
    output.write(reinterpret_cast<const char*>(voltage.data()),
                 static_cast<std::streamsize>(voltage.size()));
    if (!output) {
        throw std::runtime_error("failed to write packed voltage file: " + path.string());
    }
}

} // namespace beamformer
