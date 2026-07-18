#!/usr/bin/env python3

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "tools" / "plot_benchmark.py"
SPEC = importlib.util.spec_from_file_location("plot_benchmark", SCRIPT)
plot_benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plot_benchmark)


def timing_row(repeat: int, cpu_ms: float, kernel_ms: float,
               pipeline_ms: float) -> dict[str, float]:
    return {
        "n_ant": 32.0,
        "n_freq": 672.0,
        "n_beams": 4.0,
        "n_time": 16.0,
        "repeat": float(repeat),
        "n_outputs": 43_008.0,
        "n_cmac": 1_376_256.0,
        "estimated_flop": 11_139_072.0,
        "cpu_ms": cpu_ms,
        "gpu_kernel_ms": kernel_ms,
        "gpu_pipeline_wall_ms": pipeline_ms,
    }


class PlotBenchmarkTest(unittest.TestCase):
    def test_summary_uses_medians_and_ratio_of_medians(self):
        records = [
            timing_row(0, 10.0, 2.0, 5.0),
            timing_row(1, 14.0, 4.0, 7.0),
        ]
        summary = plot_benchmark.summarize_timings(records)
        self.assertEqual(len(summary), 1)
        record = summary[0]
        self.assertEqual(record["cpu_ms_median"], 12.0)
        self.assertEqual(record["gpu_kernel_ms_median"], 3.0)
        self.assertEqual(record["gpu_pipeline_wall_ms_median"], 6.0)
        self.assertEqual(record["speedup_kernel"], 4.0)
        self.assertEqual(record["speedup_pipeline"], 2.0)
        self.assertAlmostEqual(record["cpu_cmac_per_s"], 1_376_256.0 / 0.012)

    def test_matrix_preserves_beam_and_time_order(self):
        records = [
            {"n_ant": 32.0, "n_beams": 4.0, "n_time": 16.0,
             "speedup_pipeline": 3.0},
            {"n_ant": 32.0, "n_beams": 1.0, "n_time": 1.0,
             "speedup_pipeline": 0.5},
            {"n_ant": 32.0, "n_beams": 4.0, "n_time": 1.0,
             "speedup_pipeline": 1.5},
            {"n_ant": 32.0, "n_beams": 1.0, "n_time": 16.0,
             "speedup_pipeline": 1.0},
        ]
        beams, times, matrix = plot_benchmark.matrix_for(
            records, 32, "speedup_pipeline")
        np.testing.assert_array_equal(beams, [1, 4])
        np.testing.assert_array_equal(times, [1, 16])
        np.testing.assert_allclose(matrix, [[0.5, 1.0], [1.5, 3.0]])

    def test_csv_loader_and_summary_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "timings.csv"
            destination = Path(directory) / "summary.csv"
            source.write_text(
                "n_ant,n_freq,n_beams,n_time,repeat,n_outputs,n_cmac,"
                "estimated_flop,cpu_ms,gpu_kernel_ms,gpu_pipeline_wall_ms\n"
                "32,672,4,16,0,43008,1376256,11139072,10,2,5\n"
            )
            records = plot_benchmark.load_numeric_csv(source)
            summary = plot_benchmark.summarize_timings(records)
            plot_benchmark.write_summary(destination, summary)
            self.assertTrue(destination.exists())
            self.assertIn("speedup_pipeline", destination.read_text().splitlines()[0])


if __name__ == "__main__":
    unittest.main()
