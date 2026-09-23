#include "beamformer/upchannelizer_indexing.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

template <typename Exception, typename Function>
bool throws(Function&& function) {
    try {
        function();
    } catch (const Exception&) {
        return true;
    }
    return false;
}

bool close(const double actual, const double expected, const double tolerance = 1.0e-6) {
    return actual >= expected - tolerance && actual <= expected + tolerance;
}

} // namespace

int main() {
    using namespace beamformer;

    const RawVoltageDimensions raw{15360, local_frequency_channels, 64};
    const auto channelized = channelized_dimensions(raw);
    assert(channelized.n_time == 480);
    assert(channelized.n_coarse_frequency == 336);
    assert(channelized.n_fine_bin == 32);
    assert(channelized.n_element == 64);
    assert(raw_voltage_sample_count(raw) == 330301440U);
    assert(channelized_voltage_count(channelized) == 330301440U);
    assert(channelized_voltage_bytes(channelized) == 2642411520ULL);
    assert(channelized_validity_count(channelized) == 161280U);
    assert(flattened_fine_frequency_count(channelized) == 10752U);

    assert(channelized_index(0, 0, 0, 0, channelized) == 0);
    assert(channelized_index(0, 0, 0, 63, channelized) == 63);
    assert(channelized_index(0, 0, 1, 0, channelized) == 64);
    assert(channelized_index(0, 1, 0, 0, channelized) == 32U * 64U);
    assert(channelized_index(1, 0, 0, 0, channelized) == 336U * 32U * 64U);
    assert(channelized_index(479, 335, 31, 63, channelized) == 330301439U);

    assert(flattened_fine_frequency(0, 0, channelized) == 0);
    assert(flattened_fine_frequency(0, 31, channelized) == 31);
    assert(flattened_fine_frequency(1, 0, channelized) == 32);
    assert(flattened_fine_frequency(335, 31, channelized) == 10751);
    assert(coarse_frequency_from_flattened(32, channelized) == 1);
    assert(fine_bin_from_flattened(32, channelized) == 0);
    assert(coarse_frequency_from_flattened(10751, channelized) == 335);
    assert(fine_bin_from_flattened(10751, channelized) == 31);

    const auto shards = default_shard_descriptors();
    ChannelizedShardMetadata first{shards[0], raw, channelized};
    ChannelizedShardMetadata second{shards[1], raw, channelized};
    validate_channelized_shard_metadata(first);
    validate_channelized_shard_metadata(second);
    assert(close(channelized_frequency_hz(first, 0, 0), 299854687.5));
    assert(close(channelized_frequency_hz(first, 335, 31), 400645312.5));
    assert(close(channelized_frequency_hz(second, 0, 0), 400654687.5));
    assert(close(channelized_frequency_hz(second, 335, 31), 501445312.5));

    std::vector<ComplexFloat> data(128);
    const ChannelizedVoltageDimensions compact{1, 1, 32, 4};
    ChannelizedVoltageView view{data.data(), compact};
    view.at(0, 0, 1, 2) = {3.0F, -2.0F};
    const ConstChannelizedVoltageView const_view{data.data(), compact};
    assert(const_view.at(0, 0, 1, 2).real == 3.0F);
    assert(const_view.at(0, 0, 1, 2).imag == -2.0F);

    assert(throws<std::invalid_argument>([] {
        validate_upchannelizer_config({16, 4, 64});
    }));
    assert(throws<std::invalid_argument>([] {
        channelized_dimensions({15361, 1, 1});
    }));
    assert(throws<std::invalid_argument>([] {
        validate_raw_voltage_dimensions({0, 1, 1});
    }));
    assert(throws<std::invalid_argument>([] {
        validate_channelized_voltage_dimensions({1, 1, 31, 1});
    }));
    assert(throws<std::out_of_range>([&] {
        channelized_index(480, 0, 0, 0, channelized);
    }));
    assert(throws<std::out_of_range>([&] {
        flattened_fine_frequency(336, 0, channelized);
    }));
    assert(throws<std::out_of_range>([&] {
        fine_bin_from_flattened(10752, channelized);
    }));
    assert(throws<std::overflow_error>([] {
        checked_product(std::numeric_limits<std::size_t>::max(), 2, "overflow");
    }));
    assert(throws<std::invalid_argument>([&] {
        auto invalid = first;
        invalid.channelized_dimensions.n_fine_bin = 31;
        validate_channelized_shard_metadata(invalid);
    }));
    assert(throws<std::invalid_argument>([&] {
        auto invalid = second;
        invalid.raw_dimensions.n_coarse_frequency = 335;
        validate_channelized_shard_metadata(invalid);
    }));

    return 0;
}
