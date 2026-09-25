#include "beamformer/cadence.hpp"
#include "beamformer/config.hpp"
#include "beamformer/cuda_offline_runner.hpp"
#include "beamformer/cuda_two_shard_runner.hpp"
#include "beamformer/geometry.hpp"
#include "beamformer/synthetic_data.hpp"
#include "beamformer/temporal_integration.hpp"
#include "beamformer/weights.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::size_t n_time = 15360;
    std::size_t n_freq = beamformer::default_frequency_channels; // 336
    std::size_t n_ant = 64;
    std::size_t n_beams = 64;
    std::size_t integration_spectra = 320;
    beamformer::CudaBeamformerKernel kernel = beamformer::CudaBeamformerKernel::Tiled;
    std::string mode = "all";           // "all", "one-shard", "two-shard"
    std::string output_format = "all";  // "all", "int8", "float32"
    std::size_t warmups = 3;
    std::size_t repetitions = 10;
    std::uint32_t seed = 1;
    double frame_deadline_ms = 0.0;     // 0.0 => auto-compute from n_time (51.2 ms for 15360)
    std::filesystem::path output_prefix = "results/production_cadence";
    bool dry_run = false;
    bool verbose = false;
};

const char* kernel_name(const beamformer::CudaBeamformerKernel kernel) {
    return kernel == beamformer::CudaBeamformerKernel::Direct ? "direct" : "tiled";
}

const char* output_name(const beamformer::CudaBeamformerOutput output) {
    return output == beamformer::CudaBeamformerOutput::Float32 ? "float32" : "int8";
}

double elapsed_ms(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::size_t parse_size(const char* value, const char* option) {
    const std::string text(value);
    std::size_t used = 0;
    const auto parsed = std::stoull(text, &used);
    if (used != text.size()) {
        throw std::invalid_argument(std::string("invalid integer for ") + option);
    }
    return static_cast<std::size_t>(parsed);
}

double parse_double(const char* value, const char* option) {
    const std::string text(value);
    std::size_t used = 0;
    const auto parsed = std::stod(text, &used);
    if (used != text.size()) {
        throw std::invalid_argument(std::string("invalid number for ") + option);
    }
    return parsed;
}

const char* require_value(const int argc, char** argv, int& index) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value after ") + argv[index]);
    }
    return argv[++index];
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "Production-Size Kotekan Cadence Benchmark (Ticket 9)\n"
        << "Measures whether the packed-input direct CUDA beamformer pipeline sustains\n"
        << "the Kotekan frame cadence (51.2 ms for 15360 spectra) on one GPU.\n\n"
        << "Options:\n"
        << "  --n-time N                 Time samples per frame (default: 15360)\n"
        << "  --n-freq N                 Local frequency channels per shard (default: 336)\n"
        << "  --n-ant N                  Antenna elements (default: 64)\n"
        << "  --n-beams N                Synthesized beams (default: 64)\n"
        << "  --integration-spectra N    Spectra per integration window (default: 320)\n"
        << "  --kernel NAME              Kernel selector: tiled or direct (default: tiled)\n"
        << "  --mode NAME                Pipeline mode: all, one-shard, two-shard (default: all)\n"
        << "  --output-format NAME       Output format: all, int8, float32 (default: all)\n"
        << "  --warmups N                Warmup runs outside timing (default: 3)\n"
        << "  --repetitions N            Timed repetitions (default: 10)\n"
        << "  --seed N                   RNG seed for synthetic noise (default: 1)\n"
        << "  --frame-deadline-ms X      Override Kotekan frame deadline (default: 51.2 ms)\n"
        << "  --output-prefix PATH       Prefix for output files (default: results/production_cadence)\n"
        << "  --dry-run                  Calculate allocations and print config without GPU execution\n"
        << "  --verbose, -v              Print per-repetition progress\n"
        << "  --help                     Display this help message\n";
}

Options parse_options(const int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--n-time") {
            opt.n_time = parse_size(require_value(argc, argv, i), "--n-time");
        } else if (arg == "--n-freq") {
            opt.n_freq = parse_size(require_value(argc, argv, i), "--n-freq");
        } else if (arg == "--n-ant") {
            opt.n_ant = parse_size(require_value(argc, argv, i), "--n-ant");
        } else if (arg == "--n-beams") {
            opt.n_beams = parse_size(require_value(argc, argv, i), "--n-beams");
        } else if (arg == "--integration-spectra") {
            opt.integration_spectra = parse_size(require_value(argc, argv, i), "--integration-spectra");
        } else if (arg == "--kernel") {
            const std::string name(require_value(argc, argv, i));
            if (name == "direct") {
                opt.kernel = beamformer::CudaBeamformerKernel::Direct;
            } else if (name == "tiled") {
                opt.kernel = beamformer::CudaBeamformerKernel::Tiled;
            } else {
                throw std::invalid_argument("kernel must be direct or tiled");
            }
        } else if (arg == "--mode") {
            opt.mode = require_value(argc, argv, i);
            if (opt.mode != "all" && opt.mode != "one-shard" && opt.mode != "two-shard") {
                throw std::invalid_argument("mode must be all, one-shard, or two-shard");
            }
        } else if (arg == "--output-format") {
            opt.output_format = require_value(argc, argv, i);
            if (opt.output_format != "all" && opt.output_format != "int8" && opt.output_format != "float32") {
                throw std::invalid_argument("output-format must be all, int8, or float32");
            }
        } else if (arg == "--warmups") {
            opt.warmups = parse_size(require_value(argc, argv, i), "--warmups");
        } else if (arg == "--repetitions") {
            opt.repetitions = parse_size(require_value(argc, argv, i), "--repetitions");
        } else if (arg == "--seed") {
            opt.seed = static_cast<std::uint32_t>(parse_size(require_value(argc, argv, i), "--seed"));
        } else if (arg == "--frame-deadline-ms") {
            opt.frame_deadline_ms = parse_double(require_value(argc, argv, i), "--frame-deadline-ms");
        } else if (arg == "--output-prefix") {
            opt.output_prefix = require_value(argc, argv, i);
        } else if (arg == "--dry-run") {
            opt.dry_run = true;
        } else if (arg == "--verbose" || arg == "-v") {
            opt.verbose = true;
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (opt.frame_deadline_ms <= 0.0) {
        opt.frame_deadline_ms = beamformer::frame_deadline_ms(opt.n_time);
    }
    if (opt.repetitions == 0) {
        throw std::invalid_argument("--repetitions must be positive");
    }
    if (opt.integration_spectra != 10 && opt.integration_spectra != 320) {
        throw std::invalid_argument("--integration-spectra must be 10 or 320");
    }
    return opt;
}

struct HardwareInfo {
    std::string device_name;
    int driver_version = 0;
    int runtime_version = 0;
    int major = 0;
    int minor = 0;
    int multiprocessor_count = 0;
    int clock_rate_khz = 0;
    int memory_clock_rate_khz = 0;
    int memory_bus_width_bits = 0;
    std::size_t total_global_mem_bytes = 0;
};

HardwareInfo query_hardware() {
    HardwareInfo info;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        info.device_name = prop.name;
        info.major = prop.major;
        info.minor = prop.minor;
        info.multiprocessor_count = prop.multiProcessorCount;
        info.memory_bus_width_bits = prop.memoryBusWidth;
        info.total_global_mem_bytes = prop.totalGlobalMem;
        cudaDeviceGetAttribute(&info.clock_rate_khz, cudaDevAttrClockRate, 0);
        cudaDeviceGetAttribute(&info.memory_clock_rate_khz, cudaDevAttrMemoryClockRate, 0);
    }
    cudaDriverGetVersion(&info.driver_version);
    cudaRuntimeGetVersion(&info.runtime_version);
    return info;
}

struct RunTiming {
    std::string run_type; // "one_shard" or "two_shard"
    std::string output_format;
    std::string kernel;
    std::size_t repeat = 0;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double quantization_ms = 0.0;
    double d2h_ms = 0.0;
    double wall_ms = 0.0;
    double gmac_per_sec = 0.0;
    double tflop_per_sec = 0.0;
    double h2d_gb_per_sec = 0.0;
    double d2h_gb_per_sec = 0.0;
};

struct CaseSummary {
    std::string run_type;
    std::string output_format;
    std::string kernel;
    std::size_t shards_processed = 1;
    std::size_t repetitions = 0;
    beamformer::Statistics wall_stats;
    beamformer::Statistics kernel_stats;
    beamformer::Statistics h2d_stats;
    beamformer::Statistics quant_stats;
    beamformer::Statistics d2h_stats;
    double throughput_gmac_s = 0.0;
    double throughput_tflops = 0.0;
    std::size_t payload_bytes_total = 0;
    std::size_t output_bytes_total = 0;
    std::size_t host_pinned_bytes = 0;
    std::size_t device_bytes = 0;
    beamformer::CadenceVerdict cadence_verdict;
};

std::array<beamformer::TiledWeights, beamformer::frequency_shard_count>
make_benchmark_tiled_weights(const beamformer::Dimensions& dims) {
    const auto descriptors = beamformer::default_shard_descriptors();
    const auto positions = beamformer::default_positions(dims.n_ant);
    const auto directions = beamformer::default_beam_grid(dims.n_beams);
    std::array<beamformer::TiledWeights, beamformer::frequency_shard_count> weights;
    for (std::size_t s = 0; s < beamformer::frequency_shard_count; ++s) {
        const float start_hz = beamformer::default_frequency_start_hz
            + static_cast<float>(descriptors[s].absolute_frequency_start)
                  * beamformer::default_channel_width_hz;
        weights[s] = beamformer::generate_tiled_weights(
            dims, positions,
            beamformer::channelized_frequencies(dims.n_freq, start_hz), directions);
    }
    return weights;
}

std::array<beamformer::Weights, beamformer::frequency_shard_count>
make_benchmark_direct_weights(const beamformer::Dimensions& dims) {
    const auto descriptors = beamformer::default_shard_descriptors();
    const auto positions = beamformer::default_positions(dims.n_ant);
    const auto directions = beamformer::default_beam_grid(dims.n_beams);
    std::array<beamformer::Weights, beamformer::frequency_shard_count> weights;
    for (std::size_t s = 0; s < beamformer::frequency_shard_count; ++s) {
        const float start_hz = beamformer::default_frequency_start_hz
            + static_cast<float>(descriptors[s].absolute_frequency_start)
                  * beamformer::default_channel_width_hz;
        weights[s] = beamformer::generate_weights(
            dims, positions,
            beamformer::channelized_frequencies(dims.n_freq, start_hz), directions);
    }
    return weights;
}

CaseSummary benchmark_one_shard(
    const Options& opt, const beamformer::Dimensions& dims,
    const beamformer::TemporalIntegrationConfig& integration,
    const beamformer::CudaBeamformerOutput output_fmt,
    const beamformer::PackedShard& shard,
    const beamformer::Weights& weights,
    std::vector<RunTiming>& all_timings) {

    const std::string out_name = output_name(output_fmt);
    const std::string kern_name = kernel_name(opt.kernel);
    const std::size_t payload_bytes = beamformer::packed_voltage_bytes(dims);
    const std::size_t out_times = beamformer::integrated_time_count(dims.n_time, integration);
    const beamformer::Dimensions out_dims{out_times, dims.n_freq, dims.n_ant, dims.n_beams};
    const std::size_t out_bytes = (output_fmt == beamformer::CudaBeamformerOutput::Float32)
        ? beamformer::intensity_bytes(out_dims)
        : (beamformer::quantized_intensity_bytes(out_dims)
           + beamformer::quantization_parameter_count(out_dims) * sizeof(beamformer::Int8QuantizationParameters));
    const double cmac = beamformer::calculate_cmac_count(dims, 1);
    const double flops = beamformer::calculate_estimated_flops(dims, 1);

    beamformer::CudaOfflineFrameRunner runner(dims, opt.kernel, integration, output_fmt);
    runner.preload_weights(weights, dims);
    std::copy(shard.payload.begin(), shard.payload.end(), runner.pinned_host_voltage_data());

    // Warmups
    for (std::size_t w = 0; w < opt.warmups; ++w) {
        static_cast<void>(runner.run_pinned(dims, w + 1, shard.descriptor));
    }

    std::vector<double> wall_vec, kern_vec, h2d_vec, quant_vec, d2h_vec;
    wall_vec.reserve(opt.repetitions);
    kern_vec.reserve(opt.repetitions);
    h2d_vec.reserve(opt.repetitions);
    quant_vec.reserve(opt.repetitions);
    d2h_vec.reserve(opt.repetitions);

    for (std::size_t rep = 0; rep < opt.repetitions; ++rep) {
        const auto t_start = Clock::now();
        const auto res = runner.run_pinned(dims, rep + 100, shard.descriptor);
        const auto t_end = Clock::now();
        const double wall_ms = elapsed_ms(t_start, t_end);

        wall_vec.push_back(wall_ms);
        kern_vec.push_back(res.timings.kernel_ms);
        h2d_vec.push_back(res.timings.host_to_device_ms);
        quant_vec.push_back(res.timings.quantization_ms);
        d2h_vec.push_back(res.timings.device_to_host_ms);

        RunTiming rt;
        rt.run_type = "one_shard";
        rt.output_format = out_name;
        rt.kernel = kern_name;
        rt.repeat = rep + 1;
        rt.h2d_ms = res.timings.host_to_device_ms;
        rt.kernel_ms = res.timings.kernel_ms;
        rt.quantization_ms = res.timings.quantization_ms;
        rt.d2h_ms = res.timings.device_to_host_ms;
        rt.wall_ms = wall_ms;
        rt.gmac_per_sec = (cmac / 1e9) / (res.timings.kernel_ms * 1e-3);
        rt.tflop_per_sec = (flops / 1e12) / (res.timings.kernel_ms * 1e-3);
        rt.h2d_gb_per_sec = (static_cast<double>(payload_bytes) / 1e9) / (res.timings.host_to_device_ms * 1e-3);
        rt.d2h_gb_per_sec = (static_cast<double>(out_bytes) / 1e9) / (res.timings.device_to_host_ms * 1e-3);
        all_timings.push_back(rt);

        if (opt.verbose) {
            std::cout << "  [1-shard " << out_name << "] rep " << (rep + 1) << "/" << opt.repetitions
                      << ": wall=" << wall_ms << " ms (h2d=" << rt.h2d_ms << " kern=" << rt.kernel_ms
                      << " d2h=" << rt.d2h_ms << ")\n";
        }
    }

    CaseSummary summary;
    summary.run_type = "one_shard";
    summary.output_format = out_name;
    summary.kernel = kern_name;
    summary.shards_processed = 1;
    summary.repetitions = opt.repetitions;
    summary.wall_stats = beamformer::compute_statistics(wall_vec);
    summary.kernel_stats = beamformer::compute_statistics(kern_vec);
    summary.h2d_stats = beamformer::compute_statistics(h2d_vec);
    summary.quant_stats = beamformer::compute_statistics(quant_vec);
    summary.d2h_stats = beamformer::compute_statistics(d2h_vec);
    summary.throughput_gmac_s = (cmac / 1e9) / (summary.kernel_stats.median * 1e-3);
    summary.throughput_tflops = (flops / 1e12) / (summary.kernel_stats.median * 1e-3);
    summary.payload_bytes_total = payload_bytes;
    summary.output_bytes_total = out_bytes;
    summary.host_pinned_bytes = payload_bytes + weights.size() * sizeof(beamformer::ComplexFloat) + out_bytes;
    summary.device_bytes = summary.host_pinned_bytes;
    summary.cadence_verdict = beamformer::evaluate_cadence(summary.wall_stats.median, opt.frame_deadline_ms);

    return summary;
}

CaseSummary benchmark_two_shards(
    const Options& opt, const beamformer::Dimensions& dims,
    const beamformer::TemporalIntegrationConfig& integration,
    const beamformer::CudaBeamformerOutput output_fmt,
    const beamformer::PackedShardSet& shards,
    const std::array<beamformer::Weights, beamformer::frequency_shard_count>& weights,
    std::vector<RunTiming>& all_timings) {

    const std::string out_name = output_name(output_fmt);
    const std::string kern_name = kernel_name(opt.kernel);
    const std::size_t payload_bytes_single = beamformer::packed_voltage_bytes(dims);
    const std::size_t payload_bytes_total = payload_bytes_single * beamformer::frequency_shard_count;
    const std::size_t out_times = beamformer::integrated_time_count(dims.n_time, integration);
    const beamformer::Dimensions out_dims{out_times, dims.n_freq, dims.n_ant, dims.n_beams};
    const std::size_t out_bytes_single = (output_fmt == beamformer::CudaBeamformerOutput::Float32)
        ? beamformer::intensity_bytes(out_dims)
        : (beamformer::quantized_intensity_bytes(out_dims)
           + beamformer::quantization_parameter_count(out_dims) * sizeof(beamformer::Int8QuantizationParameters));
    const std::size_t out_bytes_total = out_bytes_single * beamformer::frequency_shard_count;
    const double cmac_total = beamformer::calculate_cmac_count(dims, beamformer::frequency_shard_count);
    const double flops_total = beamformer::calculate_estimated_flops(dims, beamformer::frequency_shard_count);

    beamformer::CudaOfflineTwoShardRunner runner(dims, opt.kernel, integration, output_fmt);
    runner.preload_weights(weights, dims);
    for (std::size_t s = 0; s < beamformer::frequency_shard_count; ++s) {
        std::copy(shards[s].payload.begin(), shards[s].payload.end(), runner.pinned_host_voltage_data(s));
    }
    const std::array<beamformer::ShardDescriptor, beamformer::frequency_shard_count> descriptors{
        shards[0].descriptor, shards[1].descriptor
    };

    // Warmups
    for (std::size_t w = 0; w < opt.warmups; ++w) {
        static_cast<void>(runner.run_pinned(dims, w + 1, descriptors));
    }

    std::vector<double> wall_vec, kern_vec, h2d_vec, quant_vec, d2h_vec;
    wall_vec.reserve(opt.repetitions);
    kern_vec.reserve(opt.repetitions);
    h2d_vec.reserve(opt.repetitions);
    quant_vec.reserve(opt.repetitions);
    d2h_vec.reserve(opt.repetitions);

    for (std::size_t rep = 0; rep < opt.repetitions; ++rep) {
        const auto res = runner.run_pinned(dims, rep + 200, descriptors);

        wall_vec.push_back(res.aggregate_wall_ms);
        kern_vec.push_back(res.aggregate_stage_max_timings.kernel_ms);
        h2d_vec.push_back(res.aggregate_stage_max_timings.host_to_device_ms);
        quant_vec.push_back(res.aggregate_stage_max_timings.quantization_ms);
        d2h_vec.push_back(res.aggregate_stage_max_timings.device_to_host_ms);

        RunTiming rt;
        rt.run_type = "two_shard";
        rt.output_format = out_name;
        rt.kernel = kern_name;
        rt.repeat = rep + 1;
        rt.h2d_ms = res.aggregate_stage_max_timings.host_to_device_ms;
        rt.kernel_ms = res.aggregate_stage_max_timings.kernel_ms;
        rt.quantization_ms = res.aggregate_stage_max_timings.quantization_ms;
        rt.d2h_ms = res.aggregate_stage_max_timings.device_to_host_ms;
        rt.wall_ms = res.aggregate_wall_ms;
        rt.gmac_per_sec = (cmac_total / 1e9) / (res.aggregate_stage_max_timings.kernel_ms * 1e-3);
        rt.tflop_per_sec = (flops_total / 1e12) / (res.aggregate_stage_max_timings.kernel_ms * 1e-3);
        rt.h2d_gb_per_sec = (static_cast<double>(payload_bytes_total) / 1e9)
                            / (res.aggregate_stage_max_timings.host_to_device_ms * 1e-3);
        rt.d2h_gb_per_sec = (static_cast<double>(out_bytes_total) / 1e9)
                            / (res.aggregate_stage_max_timings.device_to_host_ms * 1e-3);
        all_timings.push_back(rt);

        if (opt.verbose) {
            std::cout << "  [2-shard " << out_name << "] rep " << (rep + 1) << "/" << opt.repetitions
                      << ": wall=" << res.aggregate_wall_ms << " ms (h2d=" << rt.h2d_ms
                      << " kern=" << rt.kernel_ms << " d2h=" << rt.d2h_ms << ")\n";
        }
    }

    CaseSummary summary;
    summary.run_type = "two_shard";
    summary.output_format = out_name;
    summary.kernel = kern_name;
    summary.shards_processed = beamformer::frequency_shard_count;
    summary.repetitions = opt.repetitions;
    summary.wall_stats = beamformer::compute_statistics(wall_vec);
    summary.kernel_stats = beamformer::compute_statistics(kern_vec);
    summary.h2d_stats = beamformer::compute_statistics(h2d_vec);
    summary.quant_stats = beamformer::compute_statistics(quant_vec);
    summary.d2h_stats = beamformer::compute_statistics(d2h_vec);
    summary.throughput_gmac_s = (cmac_total / 1e9) / (summary.kernel_stats.median * 1e-3);
    summary.throughput_tflops = (flops_total / 1e12) / (summary.kernel_stats.median * 1e-3);
    summary.payload_bytes_total = payload_bytes_total;
    summary.output_bytes_total = out_bytes_total;
    const std::size_t weights_bytes_total = weights[0].size() * sizeof(beamformer::ComplexFloat) * 2;
    summary.host_pinned_bytes = payload_bytes_total + weights_bytes_total + out_bytes_total;
    summary.device_bytes = summary.host_pinned_bytes;
    summary.cadence_verdict = beamformer::evaluate_cadence(summary.wall_stats.median, opt.frame_deadline_ms);

    return summary;
}

void write_raw_timings_csv(const std::filesystem::path& path, const std::vector<RunTiming>& timings) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        throw std::runtime_error("failed to open " + path.string());
    }
    file << "run_type,output_format,kernel,repeat,h2d_ms,kernel_ms,quantization_ms,d2h_ms,wall_ms,"
         << "gmac_per_sec,tflop_per_sec,h2d_gb_per_sec,d2h_gb_per_sec\n";
    for (const auto& t : timings) {
        file << t.run_type << "," << t.output_format << "," << t.kernel << "," << t.repeat << ","
             << std::fixed << std::setprecision(4)
             << t.h2d_ms << "," << t.kernel_ms << "," << t.quantization_ms << "," << t.d2h_ms << ","
             << t.wall_ms << "," << t.gmac_per_sec << "," << t.tflop_per_sec << ","
             << t.h2d_gb_per_sec << "," << t.d2h_gb_per_sec << "\n";
    }
}

void write_summary_csv(const std::filesystem::path& path, const std::vector<CaseSummary>& summaries, const double deadline_ms) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        throw std::runtime_error("failed to open " + path.string());
    }
    file << "run_type,output_format,kernel,shards,repetitions,deadline_ms,wall_median_ms,wall_mean_ms,"
         << "wall_min_ms,wall_max_ms,wall_p99_ms,kernel_median_ms,h2d_median_ms,d2h_median_ms,"
         << "throughput_tflops,margin_ms,duty_cycle_pct,realtime_factor,cadence_pass\n";
    for (const auto& s : summaries) {
        file << s.run_type << "," << s.output_format << "," << s.kernel << "," << s.shards_processed << ","
             << s.repetitions << "," << std::fixed << std::setprecision(3)
             << deadline_ms << "," << s.wall_stats.median << "," << s.wall_stats.mean << ","
             << s.wall_stats.min << "," << s.wall_stats.max << "," << s.wall_stats.p99 << ","
             << s.kernel_stats.median << "," << s.h2d_stats.median << "," << s.d2h_stats.median << ","
             << s.throughput_tflops << "," << s.cadence_verdict.margin_ms << ","
             << s.cadence_verdict.duty_cycle_pct << "," << s.cadence_verdict.realtime_factor << ","
             << (s.cadence_verdict.passed ? "PASS" : "FAIL") << "\n";
    }
}

void write_summary_json(const std::filesystem::path& path, const Options& opt,
                        const HardwareInfo& hw, const std::vector<CaseSummary>& summaries) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        throw std::runtime_error("failed to open " + path.string());
    }
    file << "{\n"
         << "  \"benchmark\": \"production_cadence\",\n"
         << "  \"dimensions\": {\n"
         << "    \"n_time\": " << opt.n_time << ",\n"
         << "    \"n_freq_local\": " << opt.n_freq << ",\n"
         << "    \"n_ant\": " << opt.n_ant << ",\n"
         << "    \"n_beams\": " << opt.n_beams << ",\n"
         << "    \"integration_spectra\": " << opt.integration_spectra << ",\n"
         << "    \"kernel\": \"" << kernel_name(opt.kernel) << "\"\n"
         << "  },\n"
         << "  \"cadence_deadline_ms\": " << std::fixed << std::setprecision(3) << opt.frame_deadline_ms << ",\n"
         << "  \"hardware\": {\n"
         << "    \"gpu_device_name\": \"" << hw.device_name << "\",\n"
         << "    \"cuda_driver_version\": " << hw.driver_version << ",\n"
         << "    \"cuda_runtime_version\": " << hw.runtime_version << ",\n"
         << "    \"compute_capability\": \"" << hw.major << "." << hw.minor << "\",\n"
         << "    \"multiprocessor_count\": " << hw.multiprocessor_count << ",\n"
         << "    \"clock_rate_mhz\": " << (hw.clock_rate_khz / 1000.0) << ",\n"
         << "    \"total_global_mem_mib\": " << (hw.total_global_mem_bytes / (1024.0 * 1024.0)) << "\n"
         << "  },\n"
         << "  \"cases\": [\n";

    for (std::size_t i = 0; i < summaries.size(); ++i) {
        const auto& s = summaries[i];
        file << "    {\n"
             << "      \"run_type\": \"" << s.run_type << "\",\n"
             << "      \"output_format\": \"" << s.output_format << "\",\n"
             << "      \"kernel\": \"" << s.kernel << "\",\n"
             << "      \"shards_processed\": " << s.shards_processed << ",\n"
             << "      \"repetitions\": " << s.repetitions << ",\n"
             << "      \"wall_time_ms\": {\n"
             << "        \"median\": " << s.wall_stats.median << ",\n"
             << "        \"mean\": " << s.wall_stats.mean << ",\n"
             << "        \"min\": " << s.wall_stats.min << ",\n"
             << "        \"max\": " << s.wall_stats.max << ",\n"
             << "        \"p95\": " << s.wall_stats.p95 << ",\n"
             << "        \"p99\": " << s.wall_stats.p99 << ",\n"
             << "        \"stddev\": " << s.wall_stats.stddev << "\n"
             << "      },\n"
             << "      \"kernel_time_ms\": {\n"
             << "        \"median\": " << s.kernel_stats.median << ",\n"
             << "        \"mean\": " << s.kernel_stats.mean << ",\n"
             << "        \"min\": " << s.kernel_stats.min << ",\n"
             << "        \"max\": " << s.kernel_stats.max << "\n"
             << "      },\n"
             << "      \"h2d_time_ms\": {\"median\": " << s.h2d_stats.median << "},\n"
             << "      \"quantization_time_ms\": {\"median\": " << s.quant_stats.median << "},\n"
             << "      \"d2h_time_ms\": {\"median\": " << s.d2h_stats.median << "},\n"
             << "      \"throughput\": {\n"
             << "        \"gmac_per_sec\": " << s.throughput_gmac_s << ",\n"
             << "        \"tflop_per_sec\": " << s.throughput_tflops << "\n"
             << "      },\n"
             << "      \"memory_bytes\": {\n"
             << "        \"payload_total\": " << s.payload_bytes_total << ",\n"
             << "        \"output_total\": " << s.output_bytes_total << ",\n"
             << "        \"host_pinned_estimate\": " << s.host_pinned_bytes << ",\n"
             << "        \"device_estimate\": " << s.device_bytes << "\n"
             << "      },\n"
             << "      \"cadence\": {\n"
             << "        \"deadline_ms\": " << opt.frame_deadline_ms << ",\n"
             << "        \"passed\": " << (s.cadence_verdict.passed ? "true" : "false") << ",\n"
             << "        \"margin_ms\": " << s.cadence_verdict.margin_ms << ",\n"
             << "        \"duty_cycle_pct\": " << s.cadence_verdict.duty_cycle_pct << ",\n"
             << "        \"realtime_factor\": " << s.cadence_verdict.realtime_factor << "\n"
             << "      }\n"
             << "    }" << (i + 1 < summaries.size() ? "," : "") << "\n";
    }
    file << "  ]\n}\n";
}

void print_summary_table(const std::vector<CaseSummary>& summaries, const double deadline_ms) {
    std::cout << "\n"
              << "========================================================================================================\n"
              << "                         KOTEKAN FRAME CADENCE BENCHMARK SUMMARY (Ticket 9)\n"
              << "========================================================================================================\n"
              << std::left << std::setw(12) << "Pipeline"
              << std::setw(9) << "Format"
              << std::setw(8) << "Kernel"
              << std::right << std::setw(11) << "Wall (ms)"
              << std::setw(11) << "Kernel (ms)"
              << std::setw(10) << "H2D (ms)"
              << std::setw(9) << "D2H (ms)"
              << std::setw(10) << "TFLOP/s"
              << std::setw(11) << "Margin(ms)"
              << std::setw(10) << "Cadence"
              << std::setw(9) << "Speedup"
              << "\n"
              << "--------------------------------------------------------------------------------------------------------\n";

    for (const auto& s : summaries) {
        std::cout << std::left << std::setw(12) << s.run_type
                  << std::setw(9) << s.output_format
                  << std::setw(8) << s.kernel
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(11) << s.wall_stats.median
                  << std::setw(11) << s.kernel_stats.median
                  << std::setw(10) << s.h2d_stats.median
                  << std::setw(9) << s.d2h_stats.median
                  << std::setw(10) << s.throughput_tflops
                  << std::setw(11) << s.cadence_verdict.margin_ms
                  << std::setw(10) << (s.cadence_verdict.passed ? " PASS " : "*FAIL*")
                  << std::setw(8) << std::setprecision(1) << s.cadence_verdict.realtime_factor << "x"
                  << "\n";
    }
    std::cout << "========================================================================================================\n"
              << " Frame Deadline: " << std::fixed << std::setprecision(2) << deadline_ms << " ms | Margin = Deadline - Wall Time | Speedup = Real-time factor\n\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto opt = parse_options(argc, argv);
        const auto hw = query_hardware();

        const beamformer::Dimensions dims{opt.n_time, opt.n_freq, opt.n_ant, opt.n_beams};
        beamformer::validate_dimensions(dims);
        const beamformer::TemporalIntegrationConfig integration{opt.integration_spectra};

        const std::size_t payload_1s = beamformer::packed_voltage_bytes(dims);
        const std::size_t payload_2s = payload_1s * beamformer::frequency_shard_count;
        const std::size_t out_times = beamformer::integrated_time_count(dims.n_time, integration);
        const beamformer::Dimensions out_dims{out_times, dims.n_freq, dims.n_ant, dims.n_beams};

        std::cout << "=== CHARTS/Kotekan-Aligned Production Cadence Benchmark ===\n"
                  << "GPU: " << hw.device_name << " (SMs: " << hw.multiprocessor_count
                  << ", VRAM: " << (hw.total_global_mem_bytes / (1024 * 1024)) << " MiB)\n"
                  << "Driver Version: " << hw.driver_version << " | CUDA Runtime: " << hw.runtime_version << "\n"
                  << "Dimensions: T=" << dims.n_time << ", F_local=" << dims.n_freq
                  << ", E=" << dims.n_ant << ", B=" << dims.n_beams << "\n"
                  << "Temporal Integration: " << opt.integration_spectra << " spectra -> "
                  << out_times << " output intervals\n"
                  << "Kernel: " << kernel_name(opt.kernel) << " | Warmups: " << opt.warmups
                  << " | Repetitions: " << opt.repetitions << "\n"
                  << "Kotekan Frame Cadence Deadline: " << opt.frame_deadline_ms << " ms\n"
                  << "Per-Shard Payload: " << (payload_1s / (1024.0 * 1024.0)) << " MiB ("
                  << payload_1s << " bytes)\n"
                  << "Two-Shard Payload: " << (payload_2s / (1024.0 * 1024.0)) << " MiB ("
                  << payload_2s << " bytes)\n";

        if (opt.dry_run) {
            std::cout << "\n[Dry Run] Configuration and memory verified successfully. Exiting without execution.\n";
            return 0;
        }

        std::cout << "\nGenerating deterministic synthetic two-shard inputs and weights...\n";
        const auto shards = beamformer::make_two_shard_noise(dims, opt.seed);
        const auto tiled_weights = make_benchmark_tiled_weights(dims);
        const auto direct_weights = make_benchmark_direct_weights(dims);

        std::vector<RunTiming> all_timings;
        std::vector<CaseSummary> summaries;

        const bool run_1s = (opt.mode == "all" || opt.mode == "one-shard");
        const bool run_2s = (opt.mode == "all" || opt.mode == "two-shard");
        const bool run_f32 = (opt.output_format == "all" || opt.output_format == "float32");
        const bool run_int8 = (opt.output_format == "all" || opt.output_format == "int8");

        // Helper to select proper weights based on kernel
        const auto& active_weights = (opt.kernel == beamformer::CudaBeamformerKernel::Tiled)
            ? tiled_weights
            : direct_weights;

        if (run_1s) {
            if (run_f32) {
                std::cout << "\nRunning One-Shard Integrated Float32 Benchmark...\n";
                summaries.push_back(benchmark_one_shard(
                    opt, dims, integration, beamformer::CudaBeamformerOutput::Float32,
                    shards[0], active_weights[0], all_timings));
            }
            if (run_int8) {
                std::cout << "\nRunning One-Shard Integrated Quantized-Int8 Benchmark...\n";
                summaries.push_back(benchmark_one_shard(
                    opt, dims, integration, beamformer::CudaBeamformerOutput::QuantizedInt8,
                    shards[0], active_weights[0], all_timings));
            }
        }

        if (run_2s) {
            if (run_f32) {
                std::cout << "\nRunning Two-Shard Integrated Float32 Benchmark (Concurrent Dual-Stream)...\n";
                summaries.push_back(benchmark_two_shards(
                    opt, dims, integration, beamformer::CudaBeamformerOutput::Float32,
                    shards, active_weights, all_timings));
            }
            if (run_int8) {
                std::cout << "\nRunning Two-Shard Integrated Quantized-Int8 Benchmark (Concurrent Dual-Stream)...\n";
                summaries.push_back(benchmark_two_shards(
                    opt, dims, integration, beamformer::CudaBeamformerOutput::QuantizedInt8,
                    shards, active_weights, all_timings));
            }
        }

        // Print final table
        print_summary_table(summaries, opt.frame_deadline_ms);

        // Write output files
        const auto raw_csv = opt.output_prefix.string() + "_timings.csv";
        const auto sum_csv = opt.output_prefix.string() + "_summary.csv";
        const auto sum_json = opt.output_prefix.string() + "_summary.json";

        write_raw_timings_csv(raw_csv, all_timings);
        write_summary_csv(sum_csv, summaries, opt.frame_deadline_ms);
        write_summary_json(sum_json, opt, hw, summaries);

        std::cout << "Results successfully written to:\n"
                  << "  - Raw timings CSV: " << raw_csv << "\n"
                  << "  - Summary CSV:     " << sum_csv << "\n"
                  << "  - Summary JSON:    " << sum_json << "\n\n";

        // Check overall acceptance gate:
        bool all_passed = true;
        for (const auto& s : summaries) {
            if (!s.cadence_verdict.passed) {
                all_passed = false;
            }
        }

        if (all_passed) {
            std::cout << "[ACCEPTANCE GATE: PASS] The CUDA beamformer pipeline sustains the Kotekan frame cadence ("
                      << opt.frame_deadline_ms << " ms) across all evaluated modes on this GPU.\n";
            return 0;
        } else {
            std::cerr << "[ACCEPTANCE GATE: FAIL] One or more modes exceeded the frame cadence deadline ("
                      << opt.frame_deadline_ms << " ms).\n";
            return 1;
        }

    } catch (const std::exception& err) {
        std::cerr << "benchmark_production_cadence error: " << err.what() << '\n';
        return 1;
    }
}
