# Offline CHARTS Voltage Beamformer PoC

Small standalone proof of concept for comparing a direct CPU voltage beamformer with an
equivalent CUDA implementation. It is intentionally not integrated into Kotekan.

## Current implementation

The input foundation, synthetic data, weight generation, and serial CPU reference are
available:

- CMake/C++17 project with optional CUDA detection;
- the packed output layout produced by `rfsocHandlerShuffle`;
- signed complex `int4` packing and unpacking;
- common voltage, weight, and intensity indexing and size contracts;
- regular 4x8 and 8x8 array geometries with 0.6 m spacing in x and y;
- 672 frequency centers `300 + 0.3*channel` MHz, spanning 300 to 501.3 MHz;
- optional position and frequency overrides from text files;
- deterministic beam grids in direction cosines;
- 1 to 10 beams for compact validation and final 32/64-beam rectangular grids;
- one-hot, constant, point-source, and seeded-noise voltage generation;
- a small `generate_fake_data` CLI that writes headerless RFSoC-layout files;
- geometric complex `float32` weights in `[beam][frequency][element]` order;
- a serial CPU beamformer that produces `float32` intensity in
  `[time][frequency][beam]` order;
- strict binary-size validation and optional per-run CSV timing metrics.

The CUDA reference, CPU/GPU numerical comparison, repeatable benchmark sweep, and runtime
plots remain pending. CPU validation and full-sky coverage plots are available now.

## Build and test

```bash
cmake -S . -B build -DBEAMFORMER_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA is detected during configuration but no CUDA kernel is built yet. The CPU code is
kept serial to serve as a transparent numerical reference.

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
For 1 to 10 validation beams, the default line uses
`l=(beam-floor(n_beams/2))*0.02`, `m=0`. When `n_beams=n_ant`, a rectangular grid is
designed at 400 MHz using `delta_l=lambda/D_x` and `delta_m=lambda/D_y`; its fixed
directions are reused at every channel while weights remain frequency-dependent.

## Generate weights and run the CPU reference

The following commands use five beams at `l = -0.04, -0.02, 0, 0.02, 0.04`, matching
the default synthetic point source with the last beam:

```bash
./build/generate_weights \
    --n-ant 64 --n-beams 5 \
    --output weights.bin

./build/beamformer_cpu \
    --input point_source.bin --weights weights.bin \
    --n-time 32 --n-ant 64 --n-beams 5 \
    --output cpu_intensity.bin --metrics cpu_metrics.csv
```

`beamformer_cpu` reports the peak integrated beam and separates `load_ms`, `unpack_ms`,
`compute_ms`, `write_ms`, and `total_ms`. The CSV also records output elements per second
and complex multiply-accumulates per second. Repeated invocations append rows, allowing a
later CPU/GPU benchmark sweep to use one common table.

## Binary products

- Voltage: one byte per complex signed `int4` sample, `[T][F][E]`.
- Weights: `{float real, float imag}`, `[B][F][E]`.
- Intensity: one native `float32` (little-endian on the tested x86-64 host), `[T][F][B]`.

All products are headerless. Dimensions are supplied on the command line, and readers
reject files whose byte count differs from the exact expected size.

## Visualize and compare results

Run the backend-independent visualizer with the same dimensions and synthetic-source
parameters used to generate the input:

```bash
conda run -n kotekan_test python tools/plot_results.py \
    --input cpu_intensity.bin \
    --n-time 32 --n-freq 672 --n-ant 64 --n-beams 5 \
    --synthetic-type point-source \
    --source-l 0.04 --source-m 0.0 --amplitude 4 \
    --label CPU \
    --output results/cpu_point_source_validation.png
```

The validation dashboard shows:

- array geometry and output-element blocks;
- `u-v` baseline coverage in wavelengths for a selected frequency channel;
- beam and injected-source positions in the directional `l-m` plane;
- integrated intensity by beam;
- spectra for the recovered and offset beams;
- frequency-integrated intensity versus time and beam.

The default spectrum uses the physical centers from 300 to 501.3 MHz. `--frequency-hz`
remains available only as an explicit constant-frequency override, and a frequency text
file can replace all default centers.

### Full-sky 32/64-beam coverage

Generate the complete local `l-m` coverage without requiring an intensity file:

```bash
conda run -n kotekan_test python tools/plot_results.py \
    --n-ant 32 --n-beams 32 \
    --spacing-m 0.6 \
    --frequency-start-hz 300000000 \
    --channel-width-hz 300000 \
    --design-frequency-hz 400000000 \
    --overlap-db -3 \
    --sky-output results/beam_grid_32_full_sky.png
```

The sky dashboard contains dominant-beam regions, individual -3 dB contours and maximum
absolute gain at 300/400/500 MHz, an exact 672-channel average, and the number of
overlapping beams at 400 MHz. Antenna `BW_E`, `BW_H`, and `gain_dBi` are linearly
interpolated and extrapolated from the supplied 300/400/500 MHz values. Absolute array
gain assumes uniform weights with fixed total power, so the coherent array contribution is
`10*log10(n_ant)` above the interpolated element gain.

The current generated artifacts are:

- `results/beam_grid_32_full_sky.png`;
- `results/cpu_32beam_point_source_validation.png`.

Once a CUDA output exists, use the same script for numerical and graphical comparison:

```bash
conda run -n kotekan_test python tools/plot_results.py \
    --input cpu_intensity.bin --label CPU \
    --compare cuda_intensity.bin --compare-label CUDA \
    --n-time 32 --n-freq 672 --n-ant 64 --n-beams 5 \
    --synthetic-type point-source --source-l 0.04 --source-m 0.0 \
    --output results/cpu_cuda_validation.png \
    --comparison-output results/cpu_cuda_comparison.png \
    --summary-json results/cpu_cuda_errors.json
```

This second dashboard contains an intensity scatter plot, relative-error histogram,
maximum absolute error over frequency, and per-beam integrated-power difference. The JSON
reports maximum absolute error, mean/p99/maximum relative error, correlation, and the
number of samples outside the selected tolerances.

The plotting helpers have standalone tests:

```bash
conda run -n kotekan_test python tests/test_plot_results.py
```

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
