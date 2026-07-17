#include "beamformer/config.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/io.hpp"
#include "beamformer/synthetic_data.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

int main() {
    using namespace beamformer;

    const Dimensions dims{2, 672, 64, 1};
    const auto zero = pack_complex_int4(0, 0);
    const auto active = pack_complex_int4(3, -2);

    const auto one_hot = make_one_hot(dims, 1, 671, 63, {3, -2});
    assert(one_hot.size() == dims.n_time * dims.n_freq * dims.n_ant);
    assert(one_hot[voltage_index(1, 671, 63, dims)] == active);
    assert(std::count(one_hot.begin(), one_hot.end(), active) == 1);
    assert(std::count(one_hot.begin(), one_hot.end(), zero)
           == static_cast<std::ptrdiff_t>(one_hot.size() - 1));

    const auto constant = make_constant(dims, {-4, 7});
    const auto constant_value = pack_complex_int4(-4, 7);
    assert(std::all_of(constant.begin(), constant.end(),
                       [constant_value](const auto value) {
                           return value == constant_value;
                       }));

    const auto noise_a = make_noise(dims, 1234);
    const auto noise_b = make_noise(dims, 1234);
    const auto noise_c = make_noise(dims, 1235);
    assert(noise_a == noise_b);
    assert(noise_a != noise_c);
    for (const auto packed : noise_a) {
        const auto sample = unpack_complex_int4(packed);
        assert(sample.real >= -7 && sample.real <= 7);
        assert(sample.imag >= -7 && sample.imag <= 7);
    }

    const auto positions = default_positions(dims.n_ant);
    const auto frequencies = channelized_frequencies(dims.n_freq);
    const auto broadside =
        make_point_source(dims, positions, frequencies, direction_from_lm(0.0F, 0.0F));
    const auto broadside_value = pack_complex_int4(4, 0);
    assert(std::all_of(broadside.begin(), broadside.end(),
                       [broadside_value](const auto value) {
                           return value == broadside_value;
                       }));

    const auto off_axis =
        make_point_source(dims, positions, frequencies, direction_from_lm(0.04F, 0.0F));
    const std::size_t spectrum_size = dims.n_freq * dims.n_ant;
    assert(std::equal(off_axis.begin(), off_axis.begin() + spectrum_size,
                      off_axis.begin() + spectrum_size));
    assert(std::any_of(off_axis.begin(), off_axis.begin() + spectrum_size,
                       [broadside_value](const auto value) {
                           return value != broadside_value;
                       }));

    const auto path =
        std::filesystem::temp_directory_path() / "beamformer_poc_one_hot_test.bin";
    write_packed_voltage(path, one_hot, dims);
    assert(std::filesystem::file_size(path) == one_hot.size());
    std::ifstream input(path, std::ios::binary);
    const std::vector<std::uint8_t> loaded{std::istreambuf_iterator<char>(input),
                                           std::istreambuf_iterator<char>()};
    assert(loaded == one_hot);
    std::filesystem::remove(path);

    bool invalid_index_rejected = false;
    try {
        static_cast<void>(make_one_hot(dims, dims.n_time, 0, 0));
    } catch (const std::out_of_range&) {
        invalid_index_rejected = true;
    }
    assert(invalid_index_rejected);

    bool invalid_geometry_rejected = false;
    try {
        static_cast<void>(
            make_point_source(dims, std::vector<Vec3>(1), frequencies,
                              direction_from_lm(0.0F, 0.0F)));
    } catch (const std::invalid_argument&) {
        invalid_geometry_rejected = true;
    }
    assert(invalid_geometry_rejected);

    return 0;
}
