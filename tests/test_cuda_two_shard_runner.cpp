#include "beamformer/cuda_offline_runner.hpp"
#include "beamformer/cuda_two_shard_runner.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"
#include "beamformer/temporal_integration.hpp"
#include "beamformer/weights.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void compare_float(const beamformer::Intensities& expected,
                   const beamformer::Intensities& actual,
                   const std::string& label) {
    require(expected.size() == actual.size(), label + ": size differs");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float allowed = 1.0e-5F * std::abs(expected[index]) + 1.0e-4F;
        require(std::abs(expected[index] - actual[index]) <= allowed,
                label + ": output differs");
    }
}

void compare_quantized(const beamformer::QuantizedIntegratedOutput& expected,
                       const beamformer::QuantizedIntegratedOutput& actual,
                       const std::string& label) {
    require(expected.codes == actual.codes, label + ": codes differ");
    require(expected.parameters.size() == actual.parameters.size(),
            label + ": parameter size differs");
    for (std::size_t index = 0; index < expected.parameters.size(); ++index) {
        require(std::abs(expected.parameters[index].offset - actual.parameters[index].offset)
                    <= 1.0e-4F,
                label + ": offset differs");
        require(std::abs(expected.parameters[index].scale - actual.parameters[index].scale)
                    <= 1.0e-6F,
                label + ": scale differs");
    }
}

std::array<beamformer::TiledWeights, beamformer::frequency_shard_count>
make_shard_weights(const beamformer::Dimensions& dims) {
    const auto descriptors = beamformer::default_shard_descriptors();
    const auto positions = beamformer::default_positions(dims.n_ant);
    const auto directions = beamformer::default_beam_grid(dims.n_beams);
    std::array<beamformer::TiledWeights, beamformer::frequency_shard_count> weights;
    for (std::size_t shard_id = 0; shard_id < beamformer::frequency_shard_count; ++shard_id) {
        const float start_hz = beamformer::default_frequency_start_hz
            + static_cast<float>(descriptors[shard_id].absolute_frequency_start)
                  * beamformer::default_channel_width_hz;
        weights[shard_id] = beamformer::generate_tiled_weights(
            dims, positions,
            beamformer::channelized_frequencies(dims.n_freq, start_hz), directions);
    }
    return weights;
}

} // namespace

int main() {
    try {
        using namespace beamformer;
        const Dimensions dims{11, default_frequency_channels, 32, 8};
        auto shards = make_two_shard_one_hot(dims, 4, {3, 9}, {1, 7}, {3, -2},
                                              {31, 47}, 0.15F);
        const auto weights = make_shard_weights(dims);
        require(shards[0].payload.data() != shards[1].payload.data(),
                "synthetic shards unexpectedly alias their packed storage");
        require(shards[0].loss_mask != shards[1].loss_mask,
                "synthetic shards must retain independent loss masks");

        CudaOfflineTwoShardRunner dual_runner(
            dims, CudaBeamformerKernel::Tiled, integration_after_upchan);
        const auto dual = dual_runner.run(shards, weights, dims, 91);
        CudaOfflineFrameRunner sequential_runner(
            dims, CudaBeamformerKernel::Tiled, integration_after_upchan);
        for (std::size_t shard_id = 0; shard_id < frequency_shard_count; ++shard_id) {
            const auto sequential = sequential_runner.run(
                shards[shard_id].payload, weights[shard_id], dims, 91,
                shards[shard_id].descriptor);
            compare_float(sequential.float32_output, dual.shards[shard_id].float32_output,
                          "independently scheduled float shard");
            require(dual.output_shards[shard_id].shard_id == shard_id,
                    "output shard ID was not propagated");
            require(dual.output_shards[shard_id].absolute_frequency_start
                        == shards[shard_id].descriptor.absolute_frequency_start,
                    "output frequency origin was not propagated");
            require(dual.output_shards[shard_id].timestamp_start
                        == shards[shard_id].descriptor.timestamp_start
                        && dual.output_shards[shard_id].timestamp_step
                               == shards[shard_id].descriptor.timestamp_step,
                    "output time metadata was not propagated");
            require(dual.output_shards[shard_id].loss_mask_id
                        == shards[shard_id].descriptor.loss_mask_id,
                    "output loss-mask identity was not propagated");
        }
        require(dual.aggregate_wall_ms >= 0.0
                    && dual.aggregate_stage_max_timings.kernel_ms >= 0.0,
                "two-shard timing report is invalid");

        CudaOfflineTwoShardRunner dual_quantized(
            dims, CudaBeamformerKernel::Tiled, integration_after_upchan,
            CudaBeamformerOutput::QuantizedInt8);
        const auto dual_int8 = dual_quantized.run(shards, weights, dims, 92);
        CudaOfflineFrameRunner sequential_quantized(
            dims, CudaBeamformerKernel::Tiled, integration_after_upchan,
            CudaBeamformerOutput::QuantizedInt8);
        for (std::size_t shard_id = 0; shard_id < frequency_shard_count; ++shard_id) {
            const auto sequential = sequential_quantized.run(
                shards[shard_id].payload, weights[shard_id], dims, 92,
                shards[shard_id].descriptor);
            compare_quantized(sequential.quantized_output,
                              dual_int8.shards[shard_id].quantized_output,
                              "independently scheduled int8 shard");
        }

        auto mismatched_time = shards;
        ++mismatched_time[1].descriptor.timestamp_start;
        bool rejected_mismatched_time = false;
        try {
            static_cast<void>(dual_runner.run(mismatched_time, weights, dims, 93));
        } catch (const std::invalid_argument&) {
            rejected_mismatched_time = true;
        }
        require(rejected_mismatched_time,
                "two-shard runner accepted incompatible time ranges");
    } catch (const std::exception& error) {
        std::cerr << "test_cuda_two_shard_runner: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
