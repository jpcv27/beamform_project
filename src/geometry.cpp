#include "beamformer/geometry.hpp"

#include "beamformer/physics.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace beamformer {
namespace {

std::string data_part(std::string line) {
    const auto comment = line.find('#');
    if (comment != std::string::npos) {
        line.erase(comment);
    }
    std::replace(line.begin(), line.end(), ',', ' ');
    return line;
}

std::runtime_error parse_error(const std::filesystem::path& path, const std::size_t line_number,
                               const std::string& reason) {
    return std::runtime_error(path.string() + ":" + std::to_string(line_number) + ": " + reason);
}

} // namespace

std::vector<Vec3> regular_array(const std::size_t rows, const std::size_t columns,
                                const float spacing_m) {
    if (rows == 0 || columns == 0) {
        throw std::invalid_argument("array rows and columns must be positive");
    }
    if (spacing_m <= 0.0F) {
        throw std::invalid_argument("antenna spacing must be positive");
    }

    std::vector<Vec3> positions;
    positions.reserve(rows * columns);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            positions.push_back({static_cast<float>(column) * spacing_m,
                                 static_cast<float>(row) * spacing_m, 0.0F});
        }
    }
    return positions;
}

std::vector<Vec3> default_positions(const std::size_t n_ant, const float spacing_m) {
    if (n_ant == 32) {
        return regular_array(4, 8, spacing_m);
    }
    if (n_ant == 64) {
        return regular_array(8, 8, spacing_m);
    }
    throw std::invalid_argument("default geometry is available only for 32 or 64 antennas");
}

std::vector<float> constant_frequencies(const std::size_t n_freq, const float frequency_hz) {
    if (n_freq == 0) {
        throw std::invalid_argument("frequency count must be positive");
    }
    if (frequency_hz <= 0.0F) {
        throw std::invalid_argument("frequency must be positive");
    }
    return std::vector<float>(n_freq, frequency_hz);
}

std::vector<float> channelized_frequencies(const std::size_t n_freq,
                                           const float start_hz,
                                           const float channel_width_hz) {
    if (n_freq == 0) {
        throw std::invalid_argument("frequency count must be positive");
    }
    if (!std::isfinite(start_hz) || start_hz <= 0.0F
        || !std::isfinite(channel_width_hz) || channel_width_hz <= 0.0F) {
        throw std::invalid_argument("frequency start and channel width must be positive");
    }
    std::vector<float> frequencies(n_freq);
    for (std::size_t channel = 0; channel < n_freq; ++channel) {
        frequencies[channel] =
            start_hz + static_cast<float>(channel) * channel_width_hz;
    }
    return frequencies;
}

Vec3 direction_from_lm(const float l, const float m) {
    if (!std::isfinite(l) || !std::isfinite(m)) {
        throw std::invalid_argument("direction cosines l and m must be finite");
    }
    const float transverse_squared = l * l + m * m;
    if (transverse_squared > 1.0F) {
        throw std::invalid_argument("direction cosines must satisfy l*l + m*m <= 1");
    }
    return {l, m, std::sqrt(1.0F - transverse_squared)};
}

std::vector<Vec3> default_beam_grid(const std::size_t n_beams, const float l_step,
                                    const float m) {
    if (n_beams == 0 || n_beams > 64) {
        throw std::invalid_argument("beam grid size must be between 1 and 64");
    }
    if (!std::isfinite(l_step) || l_step <= 0.0F) {
        throw std::invalid_argument("beam l step must be positive and finite");
    }

    std::vector<Vec3> directions;
    directions.reserve(n_beams);
    const auto center = static_cast<long long>(n_beams / 2);
    for (std::size_t beam = 0; beam < n_beams; ++beam) {
        const auto offset = static_cast<long long>(beam) - center;
        directions.push_back(direction_from_lm(static_cast<float>(offset) * l_step, m));
    }
    return directions;
}

std::vector<Vec3> rectangular_beam_grid(const std::size_t n_ant,
                                        const float spacing_m,
                                        const float design_frequency_hz) {
    if (spacing_m <= 0.0F || !std::isfinite(spacing_m)) {
        throw std::invalid_argument("antenna spacing must be positive and finite");
    }
    if (design_frequency_hz <= 0.0F || !std::isfinite(design_frequency_hz)) {
        throw std::invalid_argument("beam-grid design frequency must be positive and finite");
    }

    const std::size_t rows = n_ant == 32 ? 4 : n_ant == 64 ? 8 : 0;
    const std::size_t columns = n_ant == 32 || n_ant == 64 ? 8 : 0;
    if (rows == 0) {
        throw std::invalid_argument("rectangular beam grid is available only for 32 or 64 beams");
    }

    const double wavelength_m = speed_of_light_m_per_s
                                / static_cast<double>(design_frequency_hz);
    const double delta_l = wavelength_m / (static_cast<double>(columns) * spacing_m);
    const double delta_m = wavelength_m / (static_cast<double>(rows) * spacing_m);

    std::vector<Vec3> directions;
    directions.reserve(n_ant);
    for (std::size_t row = 0; row < rows; ++row) {
        const double m = (static_cast<double>(row)
                          - static_cast<double>(rows - 1) / 2.0) * delta_m;
        for (std::size_t column = 0; column < columns; ++column) {
            const double l = (static_cast<double>(column)
                              - static_cast<double>(columns - 1) / 2.0) * delta_l;
            directions.push_back(direction_from_lm(static_cast<float>(l),
                                                   static_cast<float>(m)));
        }
    }
    return directions;
}

std::vector<Vec3> load_positions(const std::filesystem::path& path,
                                 const std::size_t expected_count) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open positions file: " + path.string());
    }

    std::vector<Vec3> positions;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        std::istringstream values(data_part(std::move(line)));
        Vec3 position{};
        if (!(values >> position[0])) {
            continue;
        }
        if (!(values >> position[1] >> position[2])) {
            throw parse_error(path, line_number, "expected three coordinates");
        }
        std::string extra;
        if (values >> extra) {
            throw parse_error(path, line_number, "unexpected value after third coordinate");
        }
        positions.push_back(position);
    }

    if (positions.size() != expected_count) {
        throw std::runtime_error("positions file contains " + std::to_string(positions.size())
                                 + " rows; expected " + std::to_string(expected_count));
    }
    return positions;
}

std::vector<float> load_frequencies(const std::filesystem::path& path,
                                    const std::size_t expected_count) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open frequency file: " + path.string());
    }

    std::vector<float> frequencies;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        std::istringstream values(data_part(std::move(line)));
        float frequency = 0.0F;
        if (!(values >> frequency)) {
            continue;
        }
        if (frequency <= 0.0F) {
            throw parse_error(path, line_number, "frequency must be positive");
        }
        std::string extra;
        if (values >> extra) {
            throw parse_error(path, line_number, "expected one frequency per line");
        }
        frequencies.push_back(frequency);
    }

    if (frequencies.size() != expected_count) {
        throw std::runtime_error("frequency file contains " + std::to_string(frequencies.size())
                                 + " rows; expected " + std::to_string(expected_count));
    }
    return frequencies;
}

std::vector<Vec3> load_directions(const std::filesystem::path& path,
                                  const std::size_t expected_count) {
    const auto directions = load_positions(path, expected_count);
    for (const auto& direction : directions) {
        const float norm_squared = direction[0] * direction[0] + direction[1] * direction[1]
                                   + direction[2] * direction[2];
        if (!std::isfinite(norm_squared) || std::abs(norm_squared - 1.0F) > 1.0e-3F) {
            throw std::runtime_error("beam directions must be finite unit vectors");
        }
    }
    return directions;
}

} // namespace beamformer
