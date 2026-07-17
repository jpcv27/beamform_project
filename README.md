# Offline CHARTS Voltage Beamformer PoC

Small standalone proof of concept for comparing a direct CPU voltage beamformer with an
equivalent CUDA implementation. It is intentionally not integrated into Kotekan.

## Current implementation

The first two foundation steps are available:

- CMake/C++17 project with optional CUDA detection;
- the packed output layout produced by `rfsocHandlerShuffle`;
- signed complex `int4` packing and unpacking;
- common voltage, weight, and intensity indexing and size contracts;
- regular 4x8 and 8x8 array geometries with 1 m default spacing;
- 672 synthetic frequency channels at a default reference of 400 MHz;
- optional position and frequency overrides from text files.
- deterministic beam grids in direction cosines;
- one-hot, constant, point-source, and seeded-noise voltage generation;
- a small `generate_fake_data` CLI that writes headerless RFSoC-layout files.

CPU/GPU beamforming, numerical comparison, metrics, and plots are the next steps.

## Build and test

```bash
cmake -S . -B build -DBEAMFORMER_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA is detected during configuration but no CUDA kernel is built yet.

## Generate synthetic voltage files

Each command below writes exactly `n_time * 672 * n_ant` bytes:

```bash
./build/generate_fake_data \
    --type one-hot --n-time 32 --n-ant 64 \
    --active-time 1 --active-frequency 100 --active-element 63 \
    --output one_hot.bin

./build/generate_fake_data \
    --type constant --n-time 32 --n-ant 64 \
    --value-real 1 --value-imag 0 \
    --output constant.bin

./build/generate_fake_data \
    --type point-source --n-time 32 --n-ant 64 \
    --source-l 0.04 --source-m 0.0 --amplitude 4 \
    --output point_source.bin

./build/generate_fake_data \
    --type noise --n-time 32 --n-ant 64 --seed 1 \
    --output noise.bin
```

The point source uses
`x_a[f] = A * exp(-j * 2*pi*frequency[f]*dot(position[a], direction)/c)`,
quantized to signed `int4`, and repeats that spectrum for every requested time sample.
The default beam grid uses `l=(beam-floor(n_beams/2))*0.02`, `m=0`, and
`n=sqrt(1-l*l-m*m)`.

## Input contract

The PoC input is one headerless binary frame with one packed complex byte per sample:

```text
voltage[time][frequency][element]
index = (time * 672 + frequency) * n_elements + element
```

The real handler creates two separate `[T][336][64]` buffers, one per NIC. The PoC
combines them into a single `[T][672][64]` file:

- NIC 0 local frequencies `0..335` become global frequencies `0..335`;
- NIC 1 local frequencies `0..335` become global frequencies `336..671`.

The element order reproduces the handler:

```text
element = (1 - rfsoc_id) * 32 + packet_element
RFSoC 1 -> element 0..31
RFSoC 0 -> element 32..63
```

The handler copies payload bytes without changing their bits. For this PoC, each byte
uses signed two's-complement `int4`, with real in the low nibble and imaginary in the
high nibble. If firmware capture proves offset-binary encoding, only the packing helper
must change; the `[T][F][E]` layout remains unchanged.

Default positions are row-major `(x, y, z)` coordinates in metres and are indexed by
the output `element` above. Position override files contain three whitespace- or
comma-separated values per line. Frequency override files contain one positive frequency
in hertz per line. Blank lines and `#` comments are accepted.
