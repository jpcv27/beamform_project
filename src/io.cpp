#include "beamformer/io.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace beamformer {
namespace {

void require_file_size(const std::filesystem::path& path,
                       const std::uintmax_t expected_bytes) {
    std::error_code error;
    const auto actual_bytes = std::filesystem::file_size(path, error);
    if (error) {
        throw std::runtime_error("cannot inspect input file: " + path.string());
    }
    if (actual_bytes != expected_bytes) {
        throw std::runtime_error(path.string() + " has " + std::to_string(actual_bytes)
                                 + " bytes; expected "
                                 + std::to_string(expected_bytes));
    }
}

template <typename Value>
std::vector<Value> read_binary(const std::filesystem::path& path,
                               const std::size_t count) {
    require_file_size(path, count * sizeof(Value));
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open input file: " + path.string());
    }
    std::vector<Value> values(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(count * sizeof(Value)));
    if (!input) {
        throw std::runtime_error("failed to read input file: " + path.string());
    }
    return values;
}

template <typename Value>
void write_binary(const std::filesystem::path& path,
                  const std::vector<Value>& values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open output file: " + path.string());
    }
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(Value)));
    if (!output) {
        throw std::runtime_error("failed to write output file: " + path.string());
    }
}

} // namespace

PackedVoltage read_packed_voltage(const std::filesystem::path& path,
                                  const Dimensions& dims) {
    validate_dimensions(dims);
    return read_binary<std::uint8_t>(path, voltage_sample_count(dims));
}

void write_packed_voltage(const std::filesystem::path& path,
                          const PackedVoltage& voltage,
                          const Dimensions& dims) {
    validate_dimensions(dims);
    if (voltage.size() != voltage_sample_count(dims)) {
        throw std::invalid_argument("packed voltage size does not match dimensions");
    }
    write_binary(path, voltage);
}

Weights read_weights(const std::filesystem::path& path, const Dimensions& dims) {
    validate_dimensions(dims);
    return read_binary<ComplexFloat>(path, dims.n_beams * dims.n_freq * dims.n_ant);
}

void write_weights(const std::filesystem::path& path, const Weights& weights,
                   const Dimensions& dims) {
    validate_dimensions(dims);
    if (weights.size() != dims.n_beams * dims.n_freq * dims.n_ant) {
        throw std::invalid_argument("weight count does not match dimensions");
    }
    write_binary(path, weights);
}

void write_intensities(const std::filesystem::path& path,
                       const Intensities& intensities,
                       const Dimensions& dims) {
    validate_dimensions(dims);
    if (intensities.size() != dims.n_time * dims.n_freq * dims.n_beams) {
        throw std::invalid_argument("intensity count does not match dimensions");
    }
    write_binary(path, intensities);
}

} // namespace beamformer
