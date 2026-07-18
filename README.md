# Offline CHARTS Voltage Beamformer PoC

Small standalone proof of concept for comparing a direct CPU voltage beamformer with an
equivalent CUDA implementation. It is intentionally not integrated into Kotekan.

## Current implementation

The input foundation, synthetic data, weight generation, serial CPU reference, and direct
CUDA implementation are available:

- CMake/C++17 project with optional CUDA detection;
- the packed output layout produced by `rfsocHandlerShuffle`;
- signed complex `int4` packing and unpacking;
- common voltage, weight, and intensity indexing and size contracts;
- regular 4x8 and 8x8 array geometries with 0.6 m spacing in x and y;
- 672 frequency centers `300 + 0.3*channel` MHz, spanning 300 to 501.3 MHz;
- optional position and frequency overrides from text files;
- deterministic beam grids in direction cosines;
- 1 to `n_ant` beams, including compact validation and final 32/64-beam grids;
- one-hot, constant, point-source, and seeded-noise voltage generation;
- a small `generate_fake_data` CLI that writes headerless RFSoC-layout files;
- geometric complex `float32` weights in `[beam][frequency][element]` order;
- a serial CPU beamformer that produces `float32` intensity in
  `[time][frequency][beam]` order;
- a CUDA beamformer with one thread per `[time][frequency][beam]` output and a direct
  complex sum over all elements;
- CPU and CUDA paths that share the host-side signed `int4` unpacking and consume the
  same `ComplexFloat` voltage and weight arrays;
- strict binary-size validation and optional per-run CSV timing metrics.

The complete CPU/CUDA validation matrix, repeatable timing sweep, summary tables, and
comparison plots are available. CPU point-source validation and full-sky beam-coverage
plots are also included.

## Build and test

```bash
cmake -S . -B build \
    -DBEAMFORMER_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

When a CUDA compiler is found, CMake builds `beamformer_cuda`; otherwise the CPU-only
targets remain available. `CMAKE_CUDA_ARCHITECTURES=89` can replace `native` for the
tested RTX 4090. The CPU code remains serial as a transparent numerical reference.

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
For non-final validation and benchmark beam counts, the default line uses
`l=(beam-floor(n_beams/2))*0.02`, `m=0`. When `n_beams=n_ant`, a rectangular grid is
designed at 400 MHz using `delta_l=lambda/D_x` and `delta_m=lambda/D_y`; its fixed
directions are reused at every channel while weights remain frequency-dependent.

## Generate weights and run CPU/CUDA

The following commands use five beams at `l = -0.04, -0.02, 0, 0.02, 0.04`, matching
the default synthetic point source with the last beam:

```bash
./build/generate_weights \
    --n-ant 64 --n-beams 5 \
    --output weights.bin

./build/beamformer_cpu \
    --input point_source.bin --weights weights.bin \
    --n-time 32 --n-ant 64 --n-beams 5 \
    --output cpu_intensity.bin --metrics metrics.csv

./build/beamformer_cuda \
    --input point_source.bin --weights weights.bin \
    --n-time 32 --n-ant 64 --n-beams 5 \
    --output cuda_intensity.bin --metrics metrics.csv
```

Both executables report the peak integrated beam and write the same output layout. CUDA
timing separates device/context setup, host-to-device copies, kernel execution, and the
device-to-host copy. In the common CSV, `compute_ms` means the serial loop for CPU and the
kernel event time for CUDA; CPU rows store zero for CUDA-only stages. Throughput and
complex GMAC/s are derived from `compute_ms`. Repeated invocations append rows to one
table, but a proper benchmark should include warmup and repeated samples rather than use
the first smoke run.

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

Use the same script for numerical and graphical CPU/CUDA comparison:

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

## Initial CUDA smoke check

The first manual check used `T=4`, `F=672`, `E=32`, and five beams with the synthetic
point source at `l=0.04`, `m=0`. CPU and CUDA both selected beam 4. Across 13440 output
values, `max_abs_error=0.005859375`, `max_relative_error=4.76e-7`, and NumPy
`allclose(rtol=1e-5, atol=1e-3)` passed. On the tested RTX 4090 the tiny workload took
about 0.070 ms in the kernel, while the first-process CUDA setup took about 91 ms. These
numbers only establish that the implementation runs and is numerically sensible; they are
not the final CPU/GPU performance result.

## Reproducible CPU/CUDA benchmark

The benchmark keeps `F=672` fixed and runs the agreed matrix:

- `n_ant=32`, `n_beams={1,4,16,32}`;
- `n_ant=64`, `n_beams={1,4,16,32,48,64}`;
- `n_time={1,16,256,1024,4096,15360}`;
- three warmups and ten measured repetitions by default.

```bash
./build/benchmark_cpu_cuda \
    --output-prefix results/cpu_cuda_benchmark

conda run -n kotekan_test python tools/plot_benchmark.py \
    --input-prefix results/cpu_cuda_benchmark
```

The process generates one deterministic signed-`int4` noise spectrum per antenna count
and repeats it over time outside all timed regions. CPU and CUDA consume the same
unpacked values and weights. CPU and GPU measurements run in separate warm blocks so a
long serial CPU iteration does not cool the GPU immediately before its sample.

The principal steady-state comparison keeps weights resident on the GPU. It reports a
resident CUDA kernel time and a pipeline wall time containing voltage H2D, kernel, output
D2H, and synchronization. Context/buffer setup and the one-time weight upload are recorded
separately. For the maximum 64-antenna case, the estimated working sets are approximately
9.86 GiB on the host and 7.40 GiB on the GPU.

Work and throughput use these explicit conventions:

```text
Ncmac = n_time * n_freq * n_beams * n_ant
estimated_FLOP = 8 * Ncmac + 3 * (n_time * n_freq * n_beams)
```

`Ncmac/time` is reported as CMAC/s. Estimated FLOP/s counts eight real operations per
complex multiply-accumulate and three for the final squared magnitude. CSV results contain
every repetition; plots use medians and p25/p75 intervals.

### RTX 4090 benchmark result

The full default sweep produced 600 timing samples and 60 numerical comparisons. With one
beam, the end-to-end GPU pipeline first exceeds the serial CPU at `n_time=16` for both
antenna counts. With four or more beams, it is already faster at `n_time=1`.

For `n_ant=64`, `n_beams=64`, and `n_time=15360`, median time was 42.331 s on the serial
CPU, 254.321 ms for the resident CUDA kernel, and 953.775 ms for voltage H2D + kernel +
intensity D2H. This corresponds to `166.45x` kernel-only and `44.38x` pipeline speedup.
The same configuration reached an estimated 1.338 TFLOP/s in the kernel and 356.7 GFLOP/s
including transfers. Its CPU and pipeline p25/p75 intervals were 39.083-43.097 s and
953.235-955.690 ms, respectively, so plots should be interpreted from medians and their
intervals rather than individual samples.

All 60 configurations passed the combined `atol=1e-3`, `rtol=1e-5` criterion with zero
values outside tolerance and no CPU/GPU peak-beam mismatches. Across the sweep,
`max_absolute_error=0.005859375`, maximum sampled p99 relative error was `9.77e-7`, maximum
normalized RMSE was `1.18e-7`, and minimum correlation was `0.999999999945`.

Generated products are:

- `results/cpu_cuda_benchmark_timings.csv`;
- `results/cpu_cuda_benchmark_validation.csv`;
- `results/cpu_cuda_benchmark_metadata.json`;
- `results/cpu_cuda_benchmark_summary.csv`;
- `results/cpu_cuda_benchmark_performance.png`;
- `results/cpu_cuda_benchmark_speedup_heatmaps.png`;
- `results/cpu_cuda_benchmark_validation.png`.

For a short functional check before a full run:

```bash
./build/benchmark_cpu_cuda \
    --output-prefix /tmp/cpu_cuda_benchmark_smoke \
    --n-ant 32 --times 1,16 --beams-32 1,4 \
    --warmup 1 --repetitions 2
```
