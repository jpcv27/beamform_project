#include "beamformer/upchannelizer_contract.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

bool close(const float actual, const float expected, const float tolerance = 3.0e-8F) {
    return std::abs(actual - expected) <= tolerance;
}

template <typename Function>
bool throws_out_of_range(Function&& function) {
    try {
        function();
    } catch (const std::out_of_range&) {
        return true;
    }
    return false;
}

} // namespace

int main() {
    using namespace beamformer;

    static_assert(upchannelization_factor == 32);
    static_assert(pfb_taps_per_phase == 4);
    static_assert(pfb_prototype_length == 128);
    static_assert(pfb_history_samples == 96);
    static_assert(pfb_history_spectra == 3);

    const std::array<std::pair<std::size_t, float>, 10> reference_weights{{
        {0, -2.9135580499791748e-7F},
        {1, -2.3555151274883506e-6F},
        {31, -1.1871706050970648e-4F},
        {32, 3.8526791050982645e-4F},
        {63, 3.1233013979852217e-2F},
        {64, 3.1233013979852217e-2F},
        {95, 3.8526791050982645e-4F},
        {96, -1.1871706050970648e-4F},
        {126, -2.3555151274883506e-6F},
        {127, -2.9135580499791748e-7F},
    }};
    for (const auto [tap, expected] : reference_weights) {
        assert(close(sinc_hanning_pfb_weight(tap), expected));
        assert(close(sinc_hanning_pfb_weight(tap),
                     sinc_hanning_pfb_weight(pfb_prototype_length - 1 - tap)));
    }

    float weight_sum = 0.0F;
    for (std::size_t tap = 0; tap < pfb_prototype_length; ++tap) {
        weight_sum += sinc_hanning_pfb_weight(tap);
    }
    assert(close(weight_sum, 1.0206149608443131F, 3.0e-7F));

    assert(close(upchannelized_bin_offset(0), -15.5F));
    assert(close(upchannelized_bin_offset(15), -0.5F));
    assert(close(upchannelized_bin_offset(16), 0.5F));
    assert(close(upchannelized_bin_offset(31), 15.5F));
    assert(close(upchannelized_bin_fraction(0), -15.5F / 32.0F));
    assert(close(upchannelized_bin_fraction(31), 15.5F / 32.0F));

    const auto phase_zero = upchannelized_forward_phase(0, 0);
    assert(close(phase_zero.real, 1.0F));
    assert(close(phase_zero.imag, 0.0F));
    for (std::size_t bin = 0; bin < upchannelization_factor; ++bin) {
        const auto phase = upchannelized_forward_phase(bin, 7);
        const auto phase_next_block =
            upchannelized_forward_phase(bin, 7 + upchannelization_factor);
        assert(close(phase_next_block.real, -phase.real));
        assert(close(phase_next_block.imag, -phase.imag));
    }

    assert(raw_time_for_pfb_tap(0, 0) == -96);
    assert(raw_time_for_pfb_tap(0, 127) == 31);
    assert(raw_time_for_pfb_tap(3, 0) == 0);
    assert(raw_time_for_pfb_tap(479, 127) == 15359);
    assert(!pfb_history_complete(0));
    assert(!pfb_history_complete(2));
    assert(pfb_history_complete(3));

    assert(throws_out_of_range([] { sinc_hanning_pfb_weight(128); }));
    assert(throws_out_of_range([] { upchannelized_bin_offset(32); }));
    assert(throws_out_of_range([] { upchannelized_forward_phase(0, 128); }));
    assert(throws_out_of_range([] { raw_time_for_pfb_tap(0, 128); }));

    return 0;
}
