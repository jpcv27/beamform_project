# Upchannelizer Ticket 1: locked PFB convention

This document fixes the numerical convention for the standalone CHARTS voltage
upchannelizer. It is informed by Kotekan's `julia/kernels/upchan.jl`, but this PoC does not
link to, invoke, or integrate Kotekan code.

## Scope and exclusions

The contract includes the PFB weights, causal sample support, forward spectral transform,
fine-bin order, normalization, and frame-local history semantics. It deliberately excludes
Kotekan's CHIME-specific `swapped_withoffset` decode, gain multiplication, float16 PFB-weight
storage, and int4 output requantization. The PoC instead receives the already documented
RFSoC packed-int4 byte contract, decodes it with `beamformer/int4.hpp`, retains PFB weights
and channelized voltage as float32, and applies no per-fine-channel gain at this stage.

## Constants

```text
U = 32                 upchannelization factor and forward FFT length
M = 4                  taps per polyphase branch
L = M * U = 128        prototype coefficients
T_raw = 15360          default raw time samples per shard
T_out = T_raw / U = 480
F_coarse = 336         independent channels per shard
E = 64                 antenna elements
```

The raw payload remains `packed[time][coarse_frequency][element]` with element contiguous.
Each shard is processed separately.

## Prototype coefficients

For `s = 0, ..., L-1`, define:

```text
q_s = (2s - (L - 1)) / (2(L + 1))
sinc(x) = 1                              if x = 0
          sin(pi x) / (pi x)              otherwise
h_s = cos(pi q_s)^2 * sinc(M q_s) / U
```

`h_s` is the sinc-Hanning prototype used by Kotekan's `Wkernel`, including its `1/U`
amplitude scaling. The sum of the 128 coefficients is approximately `1.0206149608443131`;
it is deliberately not renormalized to one.

The canonical PoC representation is float32 coefficients generated from this equation. The
production Kotekan Julia generator evaluates the equation in float32 and stores weights in
float16; that quantization is a later optional optimization/parity experiment, not part of
the initial reference contract.

`include/beamformer/upchannelizer_contract.hpp` is the executable source of this definition.
`tests/test_upchannelizer_contract.cpp` locks selected coefficients and their symmetry.

## Causal PFB plus forward transform

For a decoded complex raw voltage `x[t, f, e]`, output time `m`, coarse channel `f`, fine bin
`u`, and element `e`, define out-of-range negative `t` as complex zero:

```text
t(m, s) = mU - (M - 1)U + s

Y[m, f, u, e] = sum(s = 0 .. L-1)
                  h_s * x[t(m, s), f, e]
                  * exp(-2 pi i * (u - (U - 1)/2) * s / U)
```

This is a forward C2C transform. The complete-prototype phase is intentional: because
`U` is even, advancing `s` by one polyphase block changes the phase sign. An equivalent
polyphase implementation therefore applies the alternating sign `(-1)^j` to branch `j`
before a centered-half-bin forward 32-point FFT.

The fine-bin order is not an implicit `fftshift`. It is the storage order `u = 0 .. 31` with
coarse-channel-relative centers:

```text
offset(u) = u - 15.5
fraction(u) = offset(u) / 32
```

Consequently bin 0 is `-15.5/32` coarse-channel widths from the coarse center, bins 15 and
16 are `-0.5/32` and `+0.5/32`, and bin 31 is `+15.5/32`. If a coarse-channel center frequency
is `nu_coarse[f]` with width `delta_nu_coarse`, then:

```text
nu_fine[f, u] = nu_coarse[f] + fraction(u) * delta_nu_coarse
```

The forward cuFFT baseline must use its unnormalized `CUFFT_FORWARD` convention; `h_s`
already supplies the only amplitude scaling in this contract.

## Frame-local history and validity

The causal support of output `m` is:

```text
[mU - 96, mU + 31]
```

The Kotekan generator initializes its three prior polyphase blocks to zero at the beginning
of the execution. The PoC adopts this frame-local baseline:

- it emits all 480 spectra for a 15360-sample raw frame;
- samples with negative time index are numerically zero;
- outputs `m=0,1,2` have `history_complete=false`;
- output `m=3` is the first output whose support lies entirely within the frame;
- an output validity entry is false when its history is incomplete or any available raw
  `[time][coarse_frequency]` loss-mask entry in its support is false.

Ticket 9 may add an explicitly selected continuous-history mode. It must preserve this
frame-local mode for reproducibility and must never silently change timestamp or validity
semantics.

## Canonical output and metadata

The channelized voltage contract for one shard is:

```text
type:  ComplexFloat (two float32 values)
shape: [time_out][coarse_frequency][fine_bin][element]
order: TFUE, element contiguous
index: (((time_out * 336 + coarse_frequency) * 32 + fine_bin) * 64 + element)
```

The future output descriptor records `U`, `M`, `L`, the formula/version of `h_s`, forward
direction, half-bin order, normalization, input/output time ranges, group-delay convention,
history mode, output validity, shard identity, coarse-frequency origin, and fine-frequency
mapping. It does not concatenate the two shards.

## Contract tests

`test_upchannelizer_contract` verifies:

- `U=32`, `M=4`, `L=128`, and 96 raw history samples;
- selected sinc-Hanning coefficients, symmetry, and unnormalized coefficient sum;
- half-bin fine-frequency order;
- the forward phase's alternating sign across adjacent polyphase blocks;
- the exact raw support at output times 0, 3, and 479;
- frame-local history validity; and
- rejection of invalid taps and fine bins.
