#include "beamformer/int4.hpp"
#include "beamformer/upchannelizer_synthetic_data.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

} // namespace

int main() {
    using namespace beamformer;

    for (int real = -8; real <= 7; ++real) {
        for (int imag = -8; imag <= 7; ++imag) {
            const auto packed = pack_complex_int4(static_cast<std::int8_t>(real),
                                                  static_cast<std::int8_t>(imag));
            const auto decoded = unpack_complex_int4(packed);
            assert(decoded.real == real);
            assert(decoded.imag == imag);
        }
    }

    const RawVoltageDimensions compact{64, 3, 4};
    const auto one_hot = make_upchannelizer_one_hot(compact, 63, 2, 3, {-8, 7});
    assert(one_hot.size() == 64U * 3U * 4U);
    assert(unpack_complex_int4(one_hot[raw_tfe_index(63, 2, 3, compact)]).real == -8);
    assert(raw_tfe_index(0, 0, 3, compact) == 3);
    assert(raw_tfe_index(0, 1, 0, compact) == 4);
    assert(raw_tfe_index(1, 0, 0, compact) == 12);

    const auto noise_a = make_upchannelizer_noise(compact, 17);
    const auto noise_b = make_upchannelizer_noise(compact, 17);
    const auto noise_c = make_upchannelizer_noise(compact, 18);
    assert(noise_a == noise_b);
    assert(noise_a != noise_c);

    const auto exact = exact_upchannelizer_bin_tone(1, 24, 6.0F, 0.0, 0.0);
    assert(std::abs(exact.bin_offset - 8.5) < 1.0e-12);
    assert(std::abs(upchannelizer_tone_phase_advance(exact) - two_pi * 8.5 / 32.0) < 1.0e-12);
    const auto tone = make_upchannelizer_tones(compact, {exact});
    for (std::size_t time = 0; time < compact.n_time; ++time) {
        for (std::size_t element = 0; element < compact.n_element; ++element) {
            const auto sample = unpack_complex_int4(tone[raw_tfe_index(time, 1, element, compact)]);
            const double phase = static_cast<double>(time) * upchannelizer_tone_phase_advance(exact);
            assert(sample.real == std::lround(6.0 * std::cos(phase)));
            assert(sample.imag == std::lround(6.0 * std::sin(phase)));
        }
    }

    UpchannelizerTone off_bin{1, 8.75, 2.0F};
    const auto off_bin_tone = make_upchannelizer_tones(compact, {off_bin});
    assert(off_bin_tone != tone);
    const auto multi_tone = make_upchannelizer_tones(
        compact, {exact_upchannelizer_bin_tone(0, 16, 2.0F), UpchannelizerTone{0, 0.25, 2.0F}});
    for (const auto byte : multi_tone) {
        const auto value = unpack_complex_int4(byte);
        assert(value.real >= -4 && value.real <= 4);
        assert(value.imag >= -4 && value.imag <= 4);
    }

    const auto full = make_upchannelizer_tones(compact, {exact}, 0);
    const RawVoltageDimensions first_chunk{32, 3, 4};
    const RawVoltageDimensions second_chunk{32, 3, 4};
    const auto first = make_upchannelizer_tones(first_chunk, {exact}, 0);
    const auto second = make_upchannelizer_tones(second_chunk, {exact}, 32);
    for (std::size_t time = 0; time < 32; ++time) {
        for (std::size_t frequency = 0; frequency < 3; ++frequency) {
            for (std::size_t element = 0; element < 4; ++element) {
                assert(first[raw_tfe_index(time, frequency, element, first_chunk)]
                       == full[raw_tfe_index(time, frequency, element, compact)]);
                assert(second[raw_tfe_index(time, frequency, element, second_chunk)]
                       == full[raw_tfe_index(time + 32, frequency, element, compact)]);
            }
        }
    }

    const UpchannelizerPointSource source{1, 8.5, 400000000.0, 300000.0, 6.0F};
    const std::vector<Vec3> positions{{0.0F, 0.0F, 0.0F}, {0.6F, 0.0F, 0.0F},
                                      {1.2F, 0.0F, 0.0F}, {1.8F, 0.0F, 0.0F}};
    const auto point_source = make_upchannelizer_point_source(
        compact, source, positions, Vec3{1.0F, 0.0F, 0.0F});
    assert(point_source[raw_tfe_index(0, 0, 0, compact)] == pack_complex_int4(0, 0));
    assert(point_source[raw_tfe_index(0, 1, 0, compact)] == pack_complex_int4(6, 0));
    assert(point_source[raw_tfe_index(1, 1, 0, compact)] != pack_complex_int4(6, 0));

    const RawVoltageDimensions shard_dims{32, local_frequency_channels, 64};
    const auto shards = make_two_shard_upchannelizer_tones(
        shard_dims,
        {{{exact_upchannelizer_bin_tone(2, 16, 3.0F)},
          {exact_upchannelizer_bin_tone(333, 31, 2.0F, 0.3)}}},
        {101, 202}, 0.0F);
    validate_upchannelizer_packed_shards(shards, shard_dims);
    assert(shards[0].payload != shards[1].payload);
    assert(shards[0].loss_mask.data() != shards[1].loss_mask.data());
    const auto shard_one_mask = shards[1].loss_mask;
    auto isolated = shards;
    isolated[0].loss_mask[raw_loss_mask_index(0, 2, shard_dims)] = 0;
    assert(isolated[1].loss_mask == shard_one_mask);
    assert(isolated[1].payload == shards[1].payload);

    assert(throws<std::invalid_argument>([&] {
        make_upchannelizer_tones(compact, {UpchannelizerTone{0, 0.0, 7.0F},
                                           UpchannelizerTone{0, 0.0, 1.0F}});
    }));
    assert(throws<std::out_of_range>([&] {
        make_upchannelizer_tones(compact, {UpchannelizerTone{3, 0.0, 1.0F}});
    }));
    assert(throws<std::invalid_argument>([&] {
        make_two_shard_upchannelizer_tones(compact, {{{exact}, {exact}}});
    }));
    assert(throws<std::out_of_range>([&] {
        raw_tfe_index(64, 0, 0, compact);
    }));

    return 0;
}
