#include "beamformer/cuda_offline_runner.hpp"

#include "beamformer/cuda_frame.hpp"
#include "beamformer/quantization.hpp"
#include "beamformer/temporal_integration.hpp"

#include "beamformer/cuda_stage_api.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
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
    CudaStream() { check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate offline runner"); }
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
    CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate offline runner"); }
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
               "cudaEventElapsedTime offline runner");
    return milliseconds;
}

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(const std::size_t count) : count_(count) {
        if (count_ == 0 || count_ > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::invalid_argument("offline CUDA buffer has an invalid element count");
        }
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
                   "cudaMalloc offline frame buffer");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

    void copy_from(const std::vector<T>& host, const char* operation) {
        if (host.size() != count_) {
            throw std::invalid_argument("offline host/device buffer sizes differ");
        }
        check_cuda(cudaMemcpy(data_, host.data(), bytes(), cudaMemcpyHostToDevice), operation);
    }

    void copy_to(std::vector<T>& host, const char* operation) const {
        if (host.size() != count_) {
            throw std::invalid_argument("offline device/host buffer sizes differ");
        }
        check_cuda(cudaMemcpy(host.data(), data_, bytes(), cudaMemcpyDeviceToHost), operation);
    }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

template <typename Value>
class PinnedBuffer {
  public:
    explicit PinnedBuffer(const std::size_t count) : count_(count) {
        if (count_ == 0 || count_ > std::numeric_limits<std::size_t>::max() / sizeof(Value)) {
            throw std::invalid_argument("offline pinned buffer has an invalid count");
        }
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&data_), bytes()),
                   "cudaMallocHost offline pinned buffer");
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
            throw std::invalid_argument("offline pinned input size does not match");
        }
        std::copy(source.begin(), source.end(), data_);
    }

    void copy_to(std::vector<Value>& destination, const std::size_t count) const {
        if (count > count_ || destination.size() != count) {
            throw std::invalid_argument("offline pinned output size does not match");
        }
        std::copy(data_, data_ + count, destination.begin());
    }

  private:
    Value* data_ = nullptr;
    std::size_t count_ = 0;
};

std::size_t weight_count(const Dimensions& dims, const CudaBeamformerKernel kernel) {
    if (kernel == CudaBeamformerKernel::Direct) {
        return dims.n_beams * dims.n_freq * dims.n_ant;
    }
    return tiled_weight_count(dims);
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

struct CudaOfflineFrameRunner::Impl {
    Impl(const Dimensions& requested_capacity, const CudaBeamformerKernel selected_kernel,
         const std::optional<TemporalIntegrationConfig>& selected_integration,
         const CudaBeamformerOutput selected_output)
        : capacity(requested_capacity),
          kernel(selected_kernel),
          temporal_integration(selected_integration),
          output(selected_output),
          output_capacity(output_dimensions(capacity, temporal_integration)),
          workspace(capacity, kernel, temporal_integration, output),
          device_voltage(packed_voltage_bytes(capacity)),
          device_weights(weight_count(capacity, kernel)),
          host_voltage(packed_voltage_bytes(capacity)),
          host_weights(weight_count(capacity, kernel)) {
        device_float_output = std::make_unique<DeviceBuffer<float>>(
            output_capacity.n_time * output_capacity.n_freq * output_capacity.n_beams);
        if (output == CudaBeamformerOutput::Float32) {
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

    Dimensions capacity;
    CudaBeamformerKernel kernel;
    std::optional<TemporalIntegrationConfig> temporal_integration;
    CudaBeamformerOutput output;
    Dimensions output_capacity;
    CudaBeamformerWorkspace workspace;

    // Persistent buffers and stream for steady-state pinned execution
    DeviceBuffer<std::uint8_t> device_voltage;
    DeviceBuffer<ComplexFloat> device_weights;
    PinnedBuffer<std::uint8_t> host_voltage;
    PinnedBuffer<ComplexFloat> host_weights;
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

CudaOfflineFrameRunner::CudaOfflineFrameRunner(
    const Dimensions& capacity, const CudaBeamformerKernel kernel,
    const std::optional<TemporalIntegrationConfig> temporal_integration,
    const CudaBeamformerOutput output)
    : impl_(std::make_unique<Impl>(capacity, kernel, temporal_integration, output)) {}

CudaOfflineFrameRunner::~CudaOfflineFrameRunner() = default;

CudaBeamformerKernel CudaOfflineFrameRunner::kernel() const {
    return impl_->kernel;
}

CudaBeamformerOutput CudaOfflineFrameRunner::output() const {
    return impl_->output;
}

bool CudaOfflineFrameRunner::has_temporal_integration() const {
    return impl_->temporal_integration.has_value();
}

CudaOfflineFrameResult CudaOfflineFrameRunner::run(
    const PackedVoltage& packed, const Weights& weights, const Dimensions& dims,
    const std::uint64_t frame_id, const ShardDescriptor& shard) {
    validate_dimensions(dims);
    validate_shard_descriptor(shard);
    if (dims.n_freq != impl_->capacity.n_freq || dims.n_ant != impl_->capacity.n_ant
        || dims.n_time > impl_->capacity.n_time || dims.n_beams > impl_->capacity.n_beams) {
        throw std::invalid_argument("offline frame dimensions exceed runner capacity");
    }

    const std::size_t packed_count = packed_voltage_bytes(dims);
    if (packed.size() != packed_count) {
        throw std::invalid_argument("offline packed voltage size does not match dimensions");
    }
    const std::size_t expected_weights = weight_count(dims, impl_->kernel);
    if (weights.size() != expected_weights) {
        throw std::invalid_argument("offline weights size does not match selected layout");
    }

    const Dimensions output_dims = output_dimensions(dims, impl_->temporal_integration);
    CudaOfflineFrameResult result;
    result.output_dims = output_dims;
    result.timings.setup_ms = impl_->workspace.setup_ms();

    DeviceBuffer<std::uint8_t> device_voltage(packed.size());
    DeviceBuffer<ComplexFloat> device_weights(weights.size());

    const auto transfer_start = Clock::now();
    device_voltage.copy_from(packed, "cudaMemcpy offline voltage host to device");
    device_weights.copy_from(weights, "cudaMemcpy offline weights host to device");
    const auto transfer_end = Clock::now();
    result.timings.host_to_device_ms = elapsed_ms(transfer_start, transfer_end);

    const auto voltage_frame = make_packed_voltage_frame_view(
        device_voltage.data(), dims, frame_id, shard);
    const auto weights_frame = impl_->kernel == CudaBeamformerKernel::Direct
                                   ? make_weights_frame_view(
                                         device_weights.data(), dims, frame_id, shard)
                                   : make_tiled_weights_frame_view(
                                         device_weights.data(), dims, frame_id, shard);

    DeviceBuffer<float>* device_float_output = nullptr;
    DeviceBuffer<std::int8_t>* device_int8_output = nullptr;
    DeviceBuffer<Int8QuantizationParameters>* device_parameters = nullptr;
    std::unique_ptr<DeviceBuffer<float>> float_output;
    std::unique_ptr<DeviceBuffer<std::int8_t>> int8_output;
    std::unique_ptr<DeviceBuffer<Int8QuantizationParameters>> parameters;

    CudaFrameView output_frame;
    std::unique_ptr<CudaFrameView> parameter_frame;
    if (impl_->output == CudaBeamformerOutput::Float32) {
        result.float32_output.resize(output_dims.n_time * output_dims.n_freq
                                     * output_dims.n_beams);
        float_output = std::make_unique<DeviceBuffer<float>>(result.float32_output.size());
        device_float_output = float_output.get();
        output_frame = make_intensity_frame_view(
            device_float_output->data(), output_dims, frame_id, shard);
    } else {
        if (!impl_->temporal_integration) {
            throw std::logic_error("offline int8 output requires temporal integration");
        }
        result.quantized_output.codes.resize(quantized_intensity_bytes(output_dims));
        result.quantized_output.parameters.resize(
            quantization_parameter_count(output_dims));
        int8_output = std::make_unique<DeviceBuffer<std::int8_t>>(
            result.quantized_output.codes.size());
        parameters = std::make_unique<DeviceBuffer<Int8QuantizationParameters>>(
            result.quantized_output.parameters.size());
        device_int8_output = int8_output.get();
        device_parameters = parameters.get();
        output_frame = make_quantized_intensity_frame_view(
            device_int8_output->data(), output_dims, frame_id, shard);
        parameter_frame = std::make_unique<CudaFrameView>(
            make_quantization_parameters_frame_view(
                device_parameters->data(), output_dims, frame_id, shard));
    }

    const auto device_timings = impl_->workspace.run_device_frame(
        voltage_frame, weights_frame, output_frame, parameter_frame.get());
    result.timings.kernel_ms = device_timings.kernel_ms;
    result.timings.temporal_integration_ms = device_timings.temporal_integration_ms;
    result.timings.quantization_ms = device_timings.quantization_ms;
    result.timings.device_to_device_ms = device_timings.device_to_device_ms;

    const auto result_start = Clock::now();
    if (impl_->output == CudaBeamformerOutput::Float32) {
        device_float_output->copy_to(
            result.float32_output, "cudaMemcpy offline float output device to host");
    } else {
        device_int8_output->copy_to(
            result.quantized_output.codes, "cudaMemcpy offline int8 output device to host");
        device_parameters->copy_to(
            result.quantized_output.parameters,
            "cudaMemcpy offline quantization parameters device to host");
    }
    const auto result_end = Clock::now();
    result.timings.device_to_host_ms = elapsed_ms(result_start, result_end);
    return result;
}

void CudaOfflineFrameRunner::preload_weights(const Weights& weights, const Dimensions& dims) {
    const std::size_t expected_weights = weight_count(dims, impl_->kernel);
    if (weights.size() != expected_weights) {
        throw std::invalid_argument("preload_weights size does not match selected layout");
    }
    impl_->host_weights.copy_from(weights, expected_weights);
    check_cuda(cudaMemcpy(impl_->device_weights.data(), impl_->host_weights.data(),
                          expected_weights * sizeof(ComplexFloat), cudaMemcpyHostToDevice),
               "cudaMemcpy preload weights offline frame");
}

std::uint8_t* CudaOfflineFrameRunner::pinned_host_voltage_data() {
    return impl_->host_voltage.data();
}

CudaOfflineFrameResult CudaOfflineFrameRunner::run_pinned(
    const Dimensions& dims, const std::uint64_t frame_id, const ShardDescriptor& shard) {
    validate_dimensions(dims);
    validate_shard_descriptor(shard);
    if (dims.n_freq != impl_->capacity.n_freq || dims.n_ant != impl_->capacity.n_ant
        || dims.n_time > impl_->capacity.n_time || dims.n_beams > impl_->capacity.n_beams) {
        throw std::invalid_argument("offline frame dimensions exceed runner capacity");
    }

    const std::size_t packed_count = packed_voltage_bytes(dims);
    const Dimensions output_dims = output_dimensions(dims, impl_->temporal_integration);
    CudaOfflineFrameResult result;
    result.output_dims = output_dims;
    result.timings.setup_ms = impl_->workspace.setup_ms();
    if (impl_->output == CudaBeamformerOutput::Float32) {
        result.float32_output.resize(output_dims.n_time * output_dims.n_freq * output_dims.n_beams);
    } else {
        result.quantized_output.codes.resize(quantized_intensity_bytes(output_dims));
        result.quantized_output.parameters.resize(quantization_parameter_count(output_dims));
    }

    const auto stream = impl_->stream.get();
    check_cuda(cudaEventRecord(impl_->start.get(), stream), "cudaEventRecord offline start");
    check_cuda(cudaMemcpyAsync(impl_->device_voltage.data(), impl_->host_voltage.data(),
                               packed_count, cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync offline voltage host to device");
    check_cuda(cudaEventRecord(impl_->h2d_end.get(), stream), "cudaEventRecord offline H2D end");

    if (impl_->temporal_integration) {
        launch_packed_integrated_beamformer(
            impl_->kernel, stream, impl_->device_voltage.data(),
            impl_->device_weights.data(),
            impl_->device_float_output->data(),
            dims, *impl_->temporal_integration);
    } else {
        launch_packed_beamformer(impl_->kernel, stream, impl_->device_voltage.data(),
                                 impl_->device_weights.data(),
                                 impl_->device_float_output->data(), dims);
    }
    check_cuda(cudaEventRecord(impl_->compute_end.get(), stream), "cudaEventRecord offline compute end");

    if (impl_->output == CudaBeamformerOutput::QuantizedInt8) {
        launch_quantize_integrated_intensity(
            stream, impl_->device_float_output->data(), impl_->device_int8_output->data(),
            impl_->device_parameters->data(), output_dims);
        check_cuda(cudaEventRecord(impl_->quantization_end.get(), stream),
                   "cudaEventRecord offline quantization end");
        check_cuda(cudaMemcpyAsync(impl_->host_int8_output->data(),
                                   impl_->device_int8_output->data(),
                                   result.quantized_output.codes.size(),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync offline int8 output device to host");
        check_cuda(cudaMemcpyAsync(impl_->host_parameters->data(),
                                   impl_->device_parameters->data(),
                                   result.quantized_output.parameters.size() * sizeof(Int8QuantizationParameters),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync offline parameters device to host");
    } else {
        check_cuda(cudaMemcpyAsync(impl_->host_float_output->data(),
                                   impl_->device_float_output->data(),
                                   result.float32_output.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync offline float output device to host");
    }
    check_cuda(cudaEventRecord(impl_->d2h_end.get(), stream), "cudaEventRecord offline D2H end");
    check_cuda(cudaEventSynchronize(impl_->d2h_end.get()), "cudaEventSynchronize offline output");

    result.timings.host_to_device_ms = event_elapsed_ms(impl_->start, impl_->h2d_end);
    result.timings.kernel_ms = event_elapsed_ms(impl_->h2d_end, impl_->compute_end);
    if (impl_->output == CudaBeamformerOutput::QuantizedInt8) {
        result.timings.quantization_ms = event_elapsed_ms(impl_->compute_end, impl_->quantization_end);
        result.timings.device_to_host_ms = event_elapsed_ms(impl_->quantization_end, impl_->d2h_end);
        impl_->host_int8_output->copy_to(result.quantized_output.codes, result.quantized_output.codes.size());
        impl_->host_parameters->copy_to(result.quantized_output.parameters, result.quantized_output.parameters.size());
    } else {
        result.timings.device_to_host_ms = event_elapsed_ms(impl_->compute_end, impl_->d2h_end);
        impl_->host_float_output->copy_to(result.float32_output, result.float32_output.size());
    }
    return result;
}

} // namespace beamformer
