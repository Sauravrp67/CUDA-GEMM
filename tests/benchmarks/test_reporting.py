import json

from benchmarks.harness.cases import MATMUL_CASE_SPEC
from benchmarks.harness.reporting import (
    build_report_metadata,
    legacy_compare_report,
    legacy_sweep_report,
)


def _sample_records() -> list[dict]:
    return [
        {
            "operation": "matmul",
            "benchmark": "matmul",
            "kernel": "naive_matmul",
            "M": 64,
            "K": 256,
            "N": 512,
            "dtype": "float32",
            "median_ms": 8.0,
            "mean_ms": 8.1,
            "std_ms": 0.1,
            "min_ms": 7.9,
            "max_ms": 8.2,
            "cv_pct": 1.23,
            "num_iters": 10,
            "flops": 16777216,
            "algo_bytes": 786432,
            "achieved_tflops": 0.002097,
            "achieved_bw": 0.098304,
            "arithmetic_intensity": 21.333333,
            "sol_compute_pct": 0.015533,
            "sol_bw_pct": 0.0512,
        },
        {
            "operation": "matmul",
            "benchmark": "matmul",
            "kernel": "torch_matmul",
            "M": 64,
            "K": 256,
            "N": 512,
            "dtype": "float32",
            "median_ms": 2.0,
            "mean_ms": 2.1,
            "std_ms": 0.1,
            "min_ms": 1.9,
            "max_ms": 2.2,
            "cv_pct": 4.76,
            "num_iters": 10,
            "flops": 16777216,
            "algo_bytes": 786432,
            "achieved_tflops": 0.008389,
            "achieved_bw": 0.393216,
            "arithmetic_intensity": 21.333333,
            "sol_compute_pct": 0.062141,
            "sol_bw_pct": 0.2048,
        },
        {
            "operation": "matmul",
            "benchmark": "matmul",
            "kernel": "naive_matmul",
            "M": 128,
            "K": 256,
            "N": 512,
            "dtype": "float32",
            "median_ms": 16.0,
            "mean_ms": 16.2,
            "std_ms": 0.2,
            "min_ms": 15.8,
            "max_ms": 16.4,
            "cv_pct": 1.23,
            "num_iters": 10,
            "flops": 33554432,
            "algo_bytes": 917504,
            "achieved_tflops": 0.002097,
            "achieved_bw": 0.057344,
            "arithmetic_intensity": 36.571429,
            "sol_compute_pct": 0.015533,
            "sol_bw_pct": 0.029867,
        },
        {
            "operation": "matmul",
            "benchmark": "matmul",
            "kernel": "torch_matmul",
            "M": 128,
            "K": 256,
            "N": 512,
            "dtype": "float32",
            "median_ms": 4.0,
            "mean_ms": 4.1,
            "std_ms": 0.1,
            "min_ms": 3.9,
            "max_ms": 4.2,
            "cv_pct": 2.43,
            "num_iters": 10,
            "flops": 33554432,
            "algo_bytes": 917504,
            "achieved_tflops": 0.008389,
            "achieved_bw": 0.229376,
            "arithmetic_intensity": 36.571429,
            "sol_compute_pct": 0.062141,
            "sol_bw_pct": 0.119467,
        },
    ]


def test_legacy_sweep_report_writes_expected_json_and_final_table(tmp_path, capsys):
    metadata = build_report_metadata(
        case_spec=MATMUL_CASE_SPEC,
        report_style="sweep",
        dtype="float32",
        sweep_axis="M",
        x_values=[64, 128],
        kernels=["naive_matmul", "torch_matmul"],
        fixed_params={"K": 256, "N": 512},
        cold_l2=True,
        use_cuda_graph=False,
    )

    outputs = legacy_sweep_report(
        case_spec=MATMUL_CASE_SPEC,
        metadata=metadata,
        records=_sample_records(),
        plot_dir=tmp_path,
        timing_dir=tmp_path,
    )
    captured = capsys.readouterr().out

    assert len(outputs) == 3
    json_path, txt_path, plot_path = outputs
    payload = json.loads(json_path.read_text())

    assert payload["operation"] == "matmul"
    assert payload["report_style"] == "sweep"
    assert payload["kernels"] == ["naive_matmul", "torch_matmul"]
    assert txt_path.name.endswith("_final.txt")
    assert "Final sweep summary for matmul at M=128" in txt_path.read_text()
    assert "vs PyTorch" in txt_path.read_text()
    assert "SUMMARY TABLE" in captured
    assert plot_path.suffix == ".png"


def test_legacy_compare_report_builds_metric_specific_filenames(tmp_path, capsys):
    metadata = build_report_metadata(
        case_spec=MATMUL_CASE_SPEC,
        report_style="compare",
        dtype="float32",
        sweep_axis="M",
        x_values=[64, 128],
        kernels=["naive_matmul", "torch_matmul"],
        fixed_params={"K": 256, "N": 512},
        cold_l2=True,
        use_cuda_graph=False,
    )

    outputs = legacy_compare_report(
        case_spec=MATMUL_CASE_SPEC,
        metadata=metadata,
        records=_sample_records(),
        plot_dir=tmp_path,
        timing_dir=tmp_path,
        metrics=("latency", "tflops"),
        annotate=False,
    )
    captured = capsys.readouterr().out

    names = [path.name for path in outputs]
    assert any(name.startswith("compare_matmul_M_") and name.endswith(".json") for name in names)
    assert any(name.startswith("compare_matmul_M_") and name.endswith("_final.txt") for name in names)
    assert any(name.startswith("compare_latency_sweepM_K256_N512_none_") for name in names)
    assert any(name.startswith("compare_tflops_sweepM_K256_N512_none_") for name in names)
    txt_path = next(path for path in outputs if path.suffix == ".txt")
    assert "Bandwidth (GB/s)" in txt_path.read_text()
    assert "SUMMARY" in captured
