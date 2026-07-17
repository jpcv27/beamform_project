#include "beamformer/config.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/synthetic_data.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

struct Options {
    std::string type = "point-source";
    std::filesystem::path output;
    std::optional<std::filesystem::path> positions_file;
    std::optional<std::filesystem::path> frequencies_file;
    std::size_t n_time = 32;
    std::size_t n_ant = 64;
    std::size_t active_time = 0;
    std::size_t active_frequency = 0;
    std::size_t active_element = 0;
    std::int8_t value_real = 3;
    std::int8_t value_imag = -2;
    std::uint32_t seed = 1;
    float spacing_m = 1.0F;
    float frequency_hz = beamformer::default_frequency_hz;
    float source_l = 0.04F;
    float source_m = 0.0F;
    float amplitude = 4.0F;
};

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " --output FILE [options]\n"
        << "\n"
        << "Data types: one-hot, constant, point-source, noise\n"
        << "\n"
        << "Common options:\n"
        << "  --type TYPE             default: point-source\n"
        << "  --n-time N              default: 32\n"
        << "  --n-ant N               32 or 64; default: 64\n"
        << "  --value-real N          int4 value; default: 3\n"
        << "  --value-imag N          int4 value; default: -2\n"
        << "  --seed N                noise seed; default: 1\n"
        << "\n"
        << "One-hot options:\n"
        << "  --active-time N --active-frequency N --active-element N\n"
        << "\n"
        << "Point-source options:\n"
        << "  --source-l L --source-m M --amplitude A\n"
        << "  --spacing-m M           default geometry spacing: 1 m\n"
        << "  --frequency-hz HZ       default for all 672 channels: 400 MHz\n"
        << "  --positions FILE        optional x,y,z rows indexed by output element\n"
        << "  --frequencies FILE      optional one-Hz-value-per-line override\n";
}

const char* require_value(const int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + argv[index]);
    }
    return argv[++index];
}

std::size_t parse_size(const char* value, const char* option) {
    const std::string text(value);
    std::size_t used = 0;
    const auto parsed = std::stoull(text, &used);
    if (used != text.size()) {
        throw std::invalid_argument(std::string("invalid integer for ") + option);
    }
    return static_cast<std::size_t>(parsed);
}

std::int8_t parse_int4(const char* value, const char* option) {
    const int parsed = std::stoi(value);
    if (parsed < -8 || parsed > 7) {
        throw std::invalid_argument(std::string(option) + " must be in [-8, 7]");
    }
    return static_cast<std::int8_t>(parsed);
}

Options parse_options(const int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (argument == "--output") {
            options.output = require_value(argc, argv, i);
        } else if (argument == "--type") {
            options.type = require_value(argc, argv, i);
        } else if (argument == "--n-time") {
            options.n_time = parse_size(require_value(argc, argv, i), "--n-time");
        } else if (argument == "--n-ant") {
            options.n_ant = parse_size(require_value(argc, argv, i), "--n-ant");
        } else if (argument == "--active-time") {
            options.active_time = parse_size(require_value(argc, argv, i), "--active-time");
        } else if (argument == "--active-frequency") {
            options.active_frequency =
                parse_size(require_value(argc, argv, i), "--active-frequency");
        } else if (argument == "--active-element") {
            options.active_element =
                parse_size(require_value(argc, argv, i), "--active-element");
        } else if (argument == "--value-real") {
            options.value_real = parse_int4(require_value(argc, argv, i), "--value-real");
        } else if (argument == "--value-imag") {
            options.value_imag = parse_int4(require_value(argc, argv, i), "--value-imag");
        } else if (argument == "--seed") {
            options.seed = static_cast<std::uint32_t>(
                parse_size(require_value(argc, argv, i), "--seed"));
        } else if (argument == "--spacing-m") {
            options.spacing_m = std::stof(require_value(argc, argv, i));
        } else if (argument == "--frequency-hz") {
            options.frequency_hz = std::stof(require_value(argc, argv, i));
        } else if (argument == "--source-l") {
            options.source_l = std::stof(require_value(argc, argv, i));
        } else if (argument == "--source-m") {
            options.source_m = std::stof(require_value(argc, argv, i));
        } else if (argument == "--amplitude") {
            options.amplitude = std::stof(require_value(argc, argv, i));
        } else if (argument == "--positions") {
            options.positions_file = require_value(argc, argv, i);
        } else if (argument == "--frequencies") {
            options.frequencies_file = require_value(argc, argv, i);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.output.empty()) {
        throw std::invalid_argument("--output is required");
    }
    return options;
}

beamformer::PackedVoltage generate(const Options& options,
                                   const beamformer::Dimensions& dims) {
    const beamformer::ComplexInt4 value{options.value_real, options.value_imag};
    if (options.type == "one-hot") {
        return beamformer::make_one_hot(dims, options.active_time, options.active_frequency,
                                        options.active_element, value);
    }
    if (options.type == "constant") {
        return beamformer::make_constant(dims, value);
    }
    if (options.type == "noise") {
        return beamformer::make_noise(dims, options.seed);
    }
    if (options.type == "point-source") {
        const auto positions = options.positions_file
                                   ? beamformer::load_positions(*options.positions_file, dims.n_ant)
                                   : beamformer::default_positions(dims.n_ant, options.spacing_m);
        const auto frequencies =
            options.frequencies_file
                ? beamformer::load_frequencies(*options.frequencies_file, dims.n_freq)
                : beamformer::constant_frequencies(dims.n_freq, options.frequency_hz);
        const auto direction =
            beamformer::direction_from_lm(options.source_l, options.source_m);
        return beamformer::make_point_source(dims, positions, frequencies, direction,
                                             options.amplitude);
    }
    throw std::invalid_argument("unknown synthetic type: " + options.type);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const beamformer::Dimensions dims{
            options.n_time,
            beamformer::default_frequency_channels,
            options.n_ant,
            1,
        };
        beamformer::validate_dimensions(dims);
        const auto voltage = generate(options, dims);
        beamformer::write_packed_voltage(options.output, voltage, dims);

        std::cout << "Wrote " << voltage.size() << " bytes to " << options.output << "\n"
                  << "layout=[T=" << dims.n_time << "][F=" << dims.n_freq
                  << "][E=" << dims.n_ant << "] type=" << options.type << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generate_fake_data: " << error.what() << "\n";
        return 1;
    }
}
