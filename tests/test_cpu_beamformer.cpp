#include "beamformer/config.hpp"
#include "beamformer/cpu_beamformer.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/io.hpp"
#include "beamformer/synthetic_data.hpp"
#include "beamformer/weights.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

bool close(const float actual, const float expected, const float tolerance = 1.0e-3F) {
    return std::abs(actual - expected) <= tolerance;
}

} // namespace

int main() {
    using namespace beamformer;

    const Dimensions dims{1, default_frequency_channels, 32, 5};
    const auto positions = default_positions(dims.n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);
    const auto directions = default_beam_grid(dims.n_beams);
    const auto weights = generate_weights(dims, positions, frequencies, directions);
    assert(weights.size() == dims.n_beams * dims.n_freq * dims.n_ant);
    for (const auto& weight : weights) {
        assert(close(weight.real * weight.real + weight.imag * weight.imag, 1.0F));
    }

    const std::size_t active_frequency = 17;
    const auto packed_one_hot =
        make_one_hot(dims, 0, active_frequency, 7, ComplexInt4{3, -2});
    const auto one_hot = cpu_beamform_intensity(
        unpack_voltage(packed_one_hot, dims), weights, dims);
    for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
        for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
            const float expected = frequency == active_frequency ? 13.0F : 0.0F;
            assert(close(one_hot[intensity_index(0, frequency, beam, dims)], expected));
        }
    }

    const Dimensions broadside_dims{1, default_frequency_channels, 32, 1};
    const auto broadside_weights = generate_weights(
        broadside_dims, positions, frequencies,
        std::vector<Vec3>{direction_from_lm(0.0F, 0.0F)});
    const auto constant = cpu_beamform_intensity(
        unpack_voltage(make_constant(broadside_dims, ComplexInt4{1, 0}),
                       broadside_dims),
        broadside_weights, broadside_dims);
    const float coherent_intensity =
        static_cast<float>(broadside_dims.n_ant * broadside_dims.n_ant);
    assert(std::all_of(constant.begin(), constant.end(),
                       [coherent_intensity](const float value) {
                           return close(value, coherent_intensity);
                       }));

    const auto source_direction = direction_from_lm(0.04F, 0.0F);
    const auto point_source = make_point_source(
        dims, positions, frequencies, source_direction, 4.0F);
    const auto point_intensity = cpu_beamform_intensity(
        unpack_voltage(point_source, dims), weights, dims);
    std::vector<double> integrated(dims.n_beams, 0.0);
    for (std::size_t frequency = 0; frequency < dims.n_freq; ++frequency) {
        for (std::size_t beam = 0; beam < dims.n_beams; ++beam) {
            integrated[beam] +=
                point_intensity[intensity_index(0, frequency, beam, dims)];
        }
    }
    const auto peak = std::distance(
        integrated.begin(), std::max_element(integrated.begin(), integrated.end()));
    assert(peak == 4);

    const Dimensions final_dims{1, default_frequency_channels, 32, 32};
    const auto final_directions = rectangular_beam_grid(final_dims.n_ant);
    const auto final_weights =
        generate_weights(final_dims, positions, frequencies, final_directions);
    const std::size_t injected_beam = 12;
    const auto final_source = make_point_source(
        final_dims, positions, frequencies, final_directions[injected_beam], 4.0F);
    const auto final_intensity = cpu_beamform_intensity(
        unpack_voltage(final_source, final_dims), final_weights, final_dims);
    std::vector<double> final_integrated(final_dims.n_beams, 0.0);
    for (std::size_t frequency = 0; frequency < final_dims.n_freq; ++frequency) {
        for (std::size_t beam = 0; beam < final_dims.n_beams; ++beam) {
            final_integrated[beam] +=
                final_intensity[intensity_index(0, frequency, beam, final_dims)];
        }
    }
    const auto final_peak = std::distance(
        final_integrated.begin(),
        std::max_element(final_integrated.begin(), final_integrated.end()));
    assert(static_cast<std::size_t>(final_peak) == injected_beam);

    const auto temporary = std::filesystem::temp_directory_path();
    const auto voltage_path = temporary / "beamformer_cpu_voltage_test.bin";
    const auto weights_path = temporary / "beamformer_cpu_weights_test.bin";
    const auto malformed_path = temporary / "beamformer_cpu_malformed_test.bin";
    write_packed_voltage(voltage_path, point_source, dims);
    write_weights(weights_path, weights, dims);
    assert(read_packed_voltage(voltage_path, dims) == point_source);
    const auto loaded_weights = read_weights(weights_path, dims);
    assert(loaded_weights.size() == weights.size());
    for (std::size_t index = 0; index < weights.size(); ++index) {
        assert(loaded_weights[index].real == weights[index].real);
        assert(loaded_weights[index].imag == weights[index].imag);
    }

    {
        std::ofstream malformed(malformed_path, std::ios::binary | std::ios::trunc);
        malformed.put('\0');
    }
    bool malformed_rejected = false;
    try {
        static_cast<void>(read_packed_voltage(malformed_path, dims));
    } catch (const std::runtime_error&) {
        malformed_rejected = true;
    }
    assert(malformed_rejected);

    std::filesystem::remove(voltage_path);
    std::filesystem::remove(weights_path);
    std::filesystem::remove(malformed_path);
    return 0;
}
