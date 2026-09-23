#include "beamformer/cuda_two_shard_runner.hpp"

#include "beamformer/cuda_frame.hpp"
#include "beamformer/cuda_stage_api.hpp"
#include "beamformer/synthetic_data.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace beamformer {
namespace {

using Clock = std::chrono::steady_clock;

void check_cuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": "
                                 + cudaGetErrorString(status));
    }
}

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

class CudaStream {
  public:
    CudaStream() { check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate two-shard"); }
    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    cudaStream_t get() const { return stream_; }

  private:
    cudaStream_t stream_ = nullptr;
};

class CudaEvent {
  public:
    CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate two-shard"); }
    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    cudaEvent_t get() const { return event_; }

  private:
    cudaEvent_t event_ = nullptr;
};

double event_elapsed_ms(const CudaEvent& start, const CudaEvent& end) {
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), end.get()),
               "cudaEventElapsedTime two-shard");
    return milliseconds;
}

template <typename Value>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(const std::size_t count) : count_(count) {
        if (count_ == 0 || count_ > std::numeric_limits<std::size_t>::max() / sizeof(Value)) {
            throw std::invalid_argument("two-shard device buffer has an invalid count");
        }
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), bytes()),
                   "cudaMalloc two-shard buffer");
    }
    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    Value* data() const { return data_; }
    std::size_t bytes() const { return count_ * sizeof(Value); }

  private:
    Value* data_ = nullptr;
    std::size_t count_ = 0;
};

template <typename Value>
class PinnedBuffer {
  public:
    explicit PinnedBuffer(const std::size_t count) : count_(count) {
        if (count_ == 0 || count_ > std::numeric_limits<std::size_t>::max() / sizeof(Value)) {
            throw std::invalid_argument("two-shard pinned buffer has an invalid count");
        }
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&data_), bytes()),
                   "cudaMallocHost two-shard buffer");
    }
    ~PinnedBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    Value* data() const { return data_; }
    std::size_t bytes() const { return count_ * sizeof(Value); }

    void copy_from(const std::vector<Value>& source, const std::size_t count) {
        if (count > count_ || source.size() != count) {
            throw std::invalid_argument("two-shard pinned input size does not match");
        }
        std::copy(source.begin(), source.end(), data_);
    }

    void copy_to(std::vector<Value>& destination, const std::size_t count) const {
        if (count > count_ || destination.size() != count) {
            throw std::invalid_argument("two-shard pinned output size does not match");
        }
        std::copy(data_, data_ + count, destination.begin());
    }

  private:
    Value* data_ = nullptr;
    std::size_t count_ = 0;
};

std::size_t weight_count(const Dimensions& dims, const CudaBeamformerKernel kernel) {
    return kernel == CudaBeamformerKernel::Direct
               ? dims.n_beams * dims.n_freq * dims.n_ant
               : tiled_weight_count(dims);
}

Dimensions output_dimensions(const Dimensions& dims,
                             const std::optional<TemporalIntegrationConfig>& integration) {
    Dimensions output = dims;
    if (integration) {
        output.n_time = integrated_time_count(dims.n_time, *integration);
    }
    return output;
}

} // namespace

struct CudaOfflineTwoShardRunner::Impl {
    struct Pipeline {
        Pipeline(const Dimensions& capacity, const Dimensions& output_capacity,
                 const std::size_t weights_count, const CudaBeamformerOutput output)
            : device_voltage(packed_voltage_bytes(capacity)),
              device_weights(weights_count),
              host_voltage(packed_voltage_bytes(capacity)),
              host_weights(weights_count),
              output_format(output) {
            device_float_output = std::make_unique<DeviceBuffer<float>>(
                output_capacity.n_time * output_capacity.n_freq * output_capacity.n_beams);
            if (output_format == CudaBeamformerOutput::Float32) {
                host_float_output = std::make_unique<PinnedBuffer<float>>(
                    output_capacity.n_time * output_capacity.n_freq * output_capacity.n_beams);
            } else {
                device_int8_output = std::make_unique<DeviceBuffer<std::int8_t>>(
                    quantized_intensity_bytes(output_capacity));
                host_int8_output = std::make_unique<PinnedBuffer<std::int8_t>>(
                    quantized_intensity_bytes(output_capacity));
                device_parameters = std::make_unique<DeviceBuffer<Int8QuantizationParameters>>(
                    quantization_parameter_count(output_capacity));
                host_parameters = std::make_unique<PinnedBuffer<Int8QuantizationParameters>>(
                    quantization_parameter_count(output_capacity));
            }
        }

        DeviceBuffer<std::uint8_t> device_voltage;
        DeviceBuffer<ComplexFloat> device_weights;
        PinnedBuffer<std::uint8_t> host_voltage;
        PinnedBuffer<ComplexFloat> host_weights;
        CudaBeamformerOutput output_format;
        std::unique_ptr<DeviceBuffer<float>> device_float_output;
        std::unique_ptr<PinnedBuffer<float>> host_float_output;
        std::unique_ptr<DeviceBuffer<std::int8_t>> device_int8_output;
        std::unique_ptr<PinnedBuffer<std::int8_t>> host_int8_output;
        std::unique_ptr<DeviceBuffer<Int8QuantizationParameters>> device_parameters;
        std::unique_ptr<PinnedBuffer<Int8QuantizationParameters>> host_parameters;
        CudaStream stream;
        CudaEvent start;
        CudaEvent h2d_end;
        CudaEvent compute_end;
        CudaEvent quantization_end;
        CudaEvent d2h_end;
    };

    Impl(const Dimensions& requested_capacity, const CudaBeamformerKernel selected_kernel,
         const std::optional<TemporalIntegrationConfig>& selected_integration,
         const CudaBeamformerOutput selected_output)
        : capacity(requested_capacity), kernel(selected_kernel),
          temporal_integration(selected_integration), output(selected_output),
          output_capacity(output_dimensions(capacity, temporal_integration)) {
        for (auto& pipeline : pipelines) {
            pipeline = std::make_unique<Pipeline>(
                capacity, output_capacity, weight_count(capacity, kernel), output);
        }
    }

    Dimensions capacity;
    CudaBeamformerKernel kernel;
    std::optional<TemporalIntegrationConfig> temporal_integration;
    CudaBeamformerOutput output;
    Dimensions output_capacity;
    // One persistent device workspace, pinned staging area, stream, and event
    // set per local-frequency shard. They are deliberately never shared.
    std::array<std::unique_ptr<Pipeline>, frequency_shard_count> pipelines;
};

CudaOfflineTwoShardRunner::CudaOfflineTwoShardRunner(
    const Dimensions& capacity, const CudaBeamformerKernel kernel,
    const std::optional<TemporalIntegrationConfig> temporal_integration,
    const CudaBeamformerOutput output) {
    validate_dimensions(capacity);
    if (output == CudaBeamformerOutput::QuantizedInt8 && !temporal_integration) {
        throw std::invalid_argument("two-shard int8 output requires temporal integration");
    }
    if (temporal_integration) {
        validate_temporal_config(*temporal_integration);
        const auto spectra = temporal_integration->integration_spectra;
        if (spectra != integration_after_upchan.integration_spectra
            && spectra != integration_direct.integration_spectra) {
            throw std::invalid_argument("two-shard integration supports only 10 or 320 spectra");
        }
        if (kernel == CudaBeamformerKernel::Direct
            && spectra != integration_direct.integration_spectra) {
            throw std::invalid_argument("Direct two-shard integration supports only 320 spectra");
        }
    }
    impl_ = std::make_unique<Impl>(capacity, kernel, temporal_integration, output);
}

CudaOfflineTwoShardRunner::~CudaOfflineTwoShardRunner() = default;

CudaOfflineTwoShardResult CudaOfflineTwoShardRunner::run(
    const PackedShardSet& shards,
    const std::array<Weights, frequency_shard_count>& weights,
    const Dimensions& dims, const std::uint64_t frame_id) {
    validate_dimensions(dims);
    validate_packed_shards(shards, dims);
    if (shards[0].descriptor.timestamp_start != shards[1].descriptor.timestamp_start
        || shards[0].descriptor.timestamp_step != shards[1].descriptor.timestamp_step) {
        throw std::invalid_argument("two-shard inputs must cover the same time range");
    }
    if (dims.n_freq != impl_->capacity.n_freq || dims.n_ant != impl_->capacity.n_ant
        || dims.n_time > impl_->capacity.n_time || dims.n_beams > impl_->capacity.n_beams) {
        throw std::invalid_argument("two-shard dimensions exceed runner capacity");
    }

    const std::size_t packed_count = packed_voltage_bytes(dims);
    const std::size_t expected_weights = weight_count(dims, impl_->kernel);
    const Dimensions output_dims = output_dimensions(dims, impl_->temporal_integration);
    CudaOfflineTwoShardResult result;
    result.frame_id = frame_id;
    for (std::size_t shard_id = 0; shard_id < frequency_shard_count; ++shard_id) {
        if (shards[shard_id].payload.size() != packed_count
            || weights[shard_id].size() != expected_weights) {
            throw std::invalid_argument("two-shard payload or weights size does not match");
        }
        result.output_shards[shard_id] = shards[shard_id].descriptor;
        result.shards[shard_id].output_dims = output_dims;
        if (impl_->output == CudaBeamformerOutput::Float32) {
            result.shards[shard_id].float32_output.resize(
                output_dims.n_time * output_dims.n_freq * output_dims.n_beams);
        } else {
            result.shards[shard_id].quantized_output.codes.resize(
                quantized_intensity_bytes(output_dims));
            result.shards[shard_id].quantized_output.parameters.resize(
                quantization_parameter_count(output_dims));
        }
    }

    const auto wall_start = Clock::now();
    for (std::size_t shard_id = 0; shard_id < frequency_shard_count; ++shard_id) {
        auto& pipeline = *impl_->pipelines[shard_id];
        const auto& shard = shards[shard_id];
        pipeline.host_voltage.copy_from(shard.payload, packed_count);
        pipeline.host_weights.copy_from(weights[shard_id], expected_weights);
        const auto stream = pipeline.stream.get();
        check_cuda(cudaEventRecord(pipeline.start.get(), stream),
                   "cudaEventRecord two-shard start");
        check_cuda(cudaMemcpyAsync(pipeline.device_voltage.data(), pipeline.host_voltage.data(),
                                   packed_count, cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync two-shard voltage host to device");
        check_cuda(cudaMemcpyAsync(pipeline.device_weights.data(), pipeline.host_weights.data(),
                                   expected_weights * sizeof(ComplexFloat),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync two-shard weights host to device");
        check_cuda(cudaEventRecord(pipeline.h2d_end.get(), stream),
                   "cudaEventRecord two-shard H2D end");

        const auto voltage_frame = make_packed_voltage_frame_view(
            pipeline.device_voltage.data(), dims, frame_id, shard.descriptor);
        const auto weights_frame = impl_->kernel == CudaBeamformerKernel::Direct
                                       ? make_weights_frame_view(
                                             pipeline.device_weights.data(), dims, frame_id,
                                             shard.descriptor)
                                       : make_tiled_weights_frame_view(
                                             pipeline.device_weights.data(), dims, frame_id,
                                             shard.descriptor);
        const auto output_frame = impl_->output == CudaBeamformerOutput::Float32
                                      ? make_intensity_frame_view(
                                            pipeline.device_float_output->data(), output_dims,
                                            frame_id, shard.descriptor)
                                      : make_quantized_intensity_frame_view(
                                            pipeline.device_int8_output->data(), output_dims,
                                            frame_id, shard.descriptor);
        validate_cuda_frame_view(voltage_frame);
        validate_cuda_frame_view(weights_frame);
        validate_cuda_frame_view(output_frame);
        if (impl_->output == CudaBeamformerOutput::QuantizedInt8) {
            const auto parameters_frame = make_quantization_parameters_frame_view(
                pipeline.device_parameters->data(), output_dims, frame_id,
                shard.descriptor);
            validate_cuda_frame_view(parameters_frame);
        }

        if (impl_->temporal_integration) {
            launch_packed_integrated_beamformer(
                impl_->kernel, stream, pipeline.device_voltage.data(),
                pipeline.device_weights.data(),
                pipeline.device_float_output->data(),
                dims, *impl_->temporal_integration);
        } else {
            launch_packed_beamformer(impl_->kernel, stream, pipeline.device_voltage.data(),
                                     pipeline.device_weights.data(),
                                     pipeline.device_float_output->data(), dims);
        }
        check_cuda(cudaEventRecord(pipeline.compute_end.get(), stream),
                   "cudaEventRecord two-shard compute end");
    }

    for (std::size_t shard_id = 0; shard_id < frequency_shard_count; ++shard_id) {
        auto& pipeline = *impl_->pipelines[shard_id];
        const auto stream = pipeline.stream.get();
        if (impl_->output == CudaBeamformerOutput::QuantizedInt8) {
            launch_quantize_integrated_intensity(
                stream, pipeline.device_float_output->data(), pipeline.device_int8_output->data(),
                pipeline.device_parameters->data(), output_dims);
            check_cuda(cudaEventRecord(pipeline.quantization_end.get(), stream),
                       "cudaEventRecord two-shard quantization end");
            check_cuda(cudaMemcpyAsync(pipeline.host_int8_output->data(),
                                       pipeline.device_int8_output->data(),
                                       result.shards[shard_id].quantized_output.codes.size(),
                                       cudaMemcpyDeviceToHost, stream),
                       "cudaMemcpyAsync two-shard int8 output device to host");
            check_cuda(cudaMemcpyAsync(pipeline.host_parameters->data(),
                                       pipeline.device_parameters->data(),
                                       result.shards[shard_id].quantized_output.parameters.size()
                                           * sizeof(Int8QuantizationParameters),
                                       cudaMemcpyDeviceToHost, stream),
                       "cudaMemcpyAsync two-shard parameters device to host");
        } else {
            check_cuda(cudaMemcpyAsync(pipeline.host_float_output->data(),
                                       pipeline.device_float_output->data(),
                                       result.shards[shard_id].float32_output.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream),
                       "cudaMemcpyAsync two-shard float output device to host");
        }
        check_cuda(cudaEventRecord(pipeline.d2h_end.get(), stream),
                   "cudaEventRecord two-shard D2H end");
    }

    for (const auto& pipeline : impl_->pipelines) {
        check_cuda(cudaEventSynchronize(pipeline->d2h_end.get()),
                   "cudaEventSynchronize two-shard output");
    }
    const auto wall_end = Clock::now();
    result.aggregate_wall_ms = elapsed_ms(wall_start, wall_end);

    for (std::size_t shard_id = 0; shard_id < frequency_shard_count; ++shard_id) {
        auto& pipeline = *impl_->pipelines[shard_id];
        auto& shard_result = result.shards[shard_id];
        shard_result.timings.host_to_device_ms = event_elapsed_ms(pipeline.start, pipeline.h2d_end);
        shard_result.timings.kernel_ms = event_elapsed_ms(pipeline.h2d_end, pipeline.compute_end);
        if (impl_->output == CudaBeamformerOutput::QuantizedInt8) {
            shard_result.timings.quantization_ms =
                event_elapsed_ms(pipeline.compute_end, pipeline.quantization_end);
            shard_result.timings.device_to_host_ms =
                event_elapsed_ms(pipeline.quantization_end, pipeline.d2h_end);
            pipeline.host_int8_output->copy_to(
                shard_result.quantized_output.codes,
                shard_result.quantized_output.codes.size());
            pipeline.host_parameters->copy_to(
                shard_result.quantized_output.parameters,
                shard_result.quantized_output.parameters.size());
        } else {
            shard_result.timings.device_to_host_ms = event_elapsed_ms(pipeline.compute_end,
                                                                       pipeline.d2h_end);
            pipeline.host_float_output->copy_to(shard_result.float32_output,
                                                shard_result.float32_output.size());
        }
        result.aggregate_stage_max_timings.host_to_device_ms = std::max(
            result.aggregate_stage_max_timings.host_to_device_ms,
            shard_result.timings.host_to_device_ms);
        result.aggregate_stage_max_timings.kernel_ms = std::max(
            result.aggregate_stage_max_timings.kernel_ms, shard_result.timings.kernel_ms);
        result.aggregate_stage_max_timings.quantization_ms = std::max(
            result.aggregate_stage_max_timings.quantization_ms,
            shard_result.timings.quantization_ms);
        result.aggregate_stage_max_timings.device_to_host_ms = std::max(
            result.aggregate_stage_max_timings.device_to_host_ms,
            shard_result.timings.device_to_host_ms);
    }
    return result;
}

} // namespace beamformer
