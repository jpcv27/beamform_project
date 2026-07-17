#pragma once

#include <type_traits>

namespace beamformer {

struct ComplexFloat {
    float real;
    float imag;
};

static_assert(sizeof(ComplexFloat) == 2 * sizeof(float));
static_assert(std::is_trivially_copyable_v<ComplexFloat>);

} // namespace beamformer
