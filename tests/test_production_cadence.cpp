#include "beamformer/cadence.hpp"
#include "beamformer/config.hpp"
#include "beamformer/cuda_offline_runner.hpp"
#include "beamformer/cuda_two_shard_runner.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"
#include "beamformer/temporal_integration.hpp"
#include "beamformer/weights.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_cadence_definitions() {
    using namespace beamformer;
    const double deadline_15360 = frame_deadline_ms(15360);
    require(std::abs(deadline_15360 - 51.2) < 1e-9,
            "15360 spectra deadline must be exactly 51.2 ms");

    const double deadline_480 = frame_deadline_ms(480);
    require(std::abs(deadline_480 - 1.6) < 1e-9,
            "480 spectra deadline must be exactly 1.6 ms");

    // Pass case
    const auto pass_verdict = evaluate_cadence(20.0, 51.2);
    require(pass_verdict.passed, "20.0 ms must pass 51.2 ms deadline");
    require(std::abs(pass_verdict.margin_ms - 31.2) < 1e-9, "margin must be 31.2 ms");
    require(std::abs(pass_verdict.duty_cycle_pct - (20.0 / 51.2 * 100.0)) < 1e-6,
            "duty cycle check failed");
    require(std::abs(pass_verdict.realtime_factor - (51.2 / 20.0)) < 1e-6,
            "realtime factor check failed");

    // Fail case
    const auto fail_verdict = evaluate_cadence(60.0, 51.2);
    require(!fail_verdict.passed, "60.0 ms must fail 51.2 ms deadline");
    require(std::abs(fail_verdict.margin_ms - (-8.8)) < 1e-9, "margin must be -8.8 ms");

    // Statistics helper
    std::vector<double> samples = {10.0, 20.0, 30.0, 40.0, 50.0};
    const auto stats = compute_statistics(samples);
    require(std::abs(stats.min - 10.0) < 1e-9, "min must be 10.0");
    require(std::abs(stats.max - 50.0) < 1e-9, "max must be 50.0");
    require(std::abs(stats.mean - 30.0) < 1e-9, "mean must be 30.0");
    require(std::abs(stats.median - 30.0) < 1e-9, "median must be 30.0");
    require(std::abs(stats.p99 - 49.6) < 1e-2, "p99 calculation check");
}

void test_production_dimensions_and_sizes() {
    using namespace beamformer;
    const Dimensions prod{15360, 336, 64, 64};
    validate_dimensions(prod);

    const std::size_t payload_bytes = packed_voltage_bytes(prod);
    require(payload_bytes == 330301440ULL,
            "production shard payload must be exactly 330,301,440 bytes");

    const std::size_t tiled_weights = tiled_weight_bytes(prod);
    require(tiled_weights == 11010048ULL,
            "production tiled weights must be exactly 11,010,048 bytes");

    const TemporalIntegrationConfig direct_int{320};
    const std::size_t out_times = integrated_time_count(prod.n_time, direct_int);
    require(out_times == 48, "15360 / 320 must produce 48 integrated output times");

    const Dimensions out_dims{out_times, prod.n_freq, prod.n_ant, prod.n_beams};
    const std::size_t float32_bytes = intensity_bytes(out_dims);
    require(float32_bytes == 4128768ULL,
            "integrated float32 output must be exactly 4,128,768 bytes");

    const std::size_t int8_codes_bytes = quantized_intensity_bytes(out_dims);
    require(int8_codes_bytes == 1032192ULL,
            "quantized int8 codes must be exactly 1,032,192 bytes");

    const std::size_t param_count = quantization_parameter_count(out_dims);
    require(param_count == 4032ULL, "quantization parameter count must be 4,032");

    const std::size_t total_int8_bytes = int8_codes_bytes
        + param_count * sizeof(Int8QuantizationParameters);
    require(total_int8_bytes == 1064448ULL,
            "quantized total bytes must be exactly 1,064,448 bytes");

    const double reduction_pct = (1.0 - static_cast<double>(total_int8_bytes)
                                            / static_cast<double>(float32_bytes))
                                 * 100.0;
    require(reduction_pct > 74.0 && reduction_pct < 75.0,
            "quantization reduction must be ~74.2%");

    // Workload calculation
    const double cmac_1shard = calculate_cmac_count(prod, 1);
    require(std::abs(cmac_1shard - 21139292160.0) < 1.0, "1-shard CMAC count check");

    const double flops_1shard = calculate_estimated_flops(prod, 1);
    require(flops_1shard > 1.7e11, "1-shard FLOP estimate check");
}

void test_compact_runner_execution() {
    using namespace beamformer;
    const Dimensions dims{320, 336, 32, 16};
    const TemporalIntegrationConfig integration{320};

    auto shards = make_two_shard_noise(dims, 42);
    const auto descriptors = default_shard_descriptors();
    const auto positions = default_positions(dims.n_ant);
    const auto directions = default_beam_grid(dims.n_beams);

    std::array<TiledWeights, frequency_shard_count> weights;
    for (std::size_t s = 0; s < frequency_shard_count; ++s) {
        const float start_hz = default_frequency_start_hz
            + static_cast<float>(descriptors[s].absolute_frequency_start)
                  * default_channel_width_hz;
        weights[s] = generate_tiled_weights(
            dims, positions, channelized_frequencies(dims.n_freq, start_hz), directions);
    }

    // 1-shard float32
    CudaOfflineFrameRunner runner_1s(
        dims, CudaBeamformerKernel::Tiled, integration, CudaBeamformerOutput::Float32);
    const auto res_1s = runner_1s.run(
        shards[0].payload, weights[0], dims, 101, shards[0].descriptor);
    require(res_1s.output_dims.n_time == 1, "output time count must be 1");
    require(res_1s.float32_output.size() == dims.n_freq * dims.n_beams,
            "output size mismatch");
    require(res_1s.timings.kernel_ms > 0.0, "kernel timing must be recorded");

    // 2-shard int8
    CudaOfflineTwoShardRunner runner_2s(
        dims, CudaBeamformerKernel::Tiled, integration, CudaBeamformerOutput::QuantizedInt8);
    const auto res_2s = runner_2s.run(shards, weights, dims, 102);
    require(res_2s.shards[0].output_dims.n_time == 1, "output time count must be 1");
    require(res_2s.shards[0].quantized_output.codes.size() == dims.n_freq * dims.n_beams,
            "int8 output code count mismatch");
    require(res_2s.aggregate_wall_ms > 0.0, "aggregate wall time must be positive");
    require(res_2s.output_shards[0].shard_id == 0 && res_2s.output_shards[1].shard_id == 1,
            "shard descriptors must be preserved");

    // Cadence check for compact configuration
    const double deadline = frame_deadline_ms(dims.n_time);
    const auto verdict = evaluate_cadence(res_2s.aggregate_wall_ms, deadline);
    require(verdict.deadline_ms == deadline, "deadline must match");
}

} // namespace

int main() {
    try {
        test_cadence_definitions();
        test_production_dimensions_and_sizes();
        test_compact_runner_execution();
        std::cout << "All production cadence tests passed successfully.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_production_cadence failed: " << error.what() << '\n';
        return 1;
    }
}
