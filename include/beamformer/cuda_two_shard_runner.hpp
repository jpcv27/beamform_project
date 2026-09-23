#pragma once

#include "beamformer/cuda_offline_runner.hpp"
#include "beamformer/formats.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

namespace beamformer {

// Result for two independent local-frequency shards processed by one GPU.
// Each entry preserves the input shard descriptor; aggregate_wall_ms covers
// enqueuing and completing both streams in one call. aggregate_stage_max_timings
// is the maximum duration for each stage across shards, not a sum: the streams
// are intentionally allowed to overlap.
struct CudaOfflineTwoShardResult {
    std::array<CudaOfflineFrameResult, frequency_shard_count> shards;
    std::array<ShardDescriptor, frequency_shard_count> output_shards;
    std::uint64_t frame_id = 0;
    double aggregate_wall_ms = 0.0;
    CudaBeamformerTimings aggregate_stage_max_timings;
};

// Offline two-shard pipeline. It retains independent pinned host staging,
// device buffers, CUDA streams, and events for the two 336-channel inputs.
// Each run queues both shards before waiting, and never concatenates payloads.
class CudaOfflineTwoShardRunner {
  public:
    explicit CudaOfflineTwoShardRunner(
        const Dimensions& capacity,
        CudaBeamformerKernel kernel = CudaBeamformerKernel::Tiled,
        std::optional<TemporalIntegrationConfig> temporal_integration = std::nullopt,
        CudaBeamformerOutput output = CudaBeamformerOutput::Float32);
    ~CudaOfflineTwoShardRunner();

    CudaOfflineTwoShardRunner(const CudaOfflineTwoShardRunner&) = delete;
    CudaOfflineTwoShardRunner& operator=(const CudaOfflineTwoShardRunner&) = delete;

    CudaOfflineTwoShardResult run(
        const PackedShardSet& shards,
        const std::array<Weights, frequency_shard_count>& weights,
        const Dimensions& dims, std::uint64_t frame_id);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace beamformer
