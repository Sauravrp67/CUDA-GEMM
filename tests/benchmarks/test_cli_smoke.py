from pathlib import Path

import pytest

from benchmarks.kernels.primitives.matmul import main as matmul_main


@pytest.mark.requires_cuda
def test_matmul_cli_both_report_styles_emit_outputs(tmp_path: Path):
    result = matmul_main(
        [
            "--kernels",
            "naive_matmul",
            "tiled_matmul",
            "coarsened_tiled_matmul",
            "reg2d_tiled_matmul",
            "reg1d_tiled_matmul",
            "vec_reg2d_tiled_matmul",
            "warp_tiled_matmul",
            "torch_matmul",
            "--report-style",
            "both",
            "--M",
            "16",
            "32",
            "--K",
            "32",
            "--N",
            "64",
            "--warmup-ms",
            "1",
            "--timed-ms",
            "1",
            "--out-dir",
            str(tmp_path),
            "--no-l2-flush",
        ]
    )

    outputs = [Path(path) for path in result["outputs"]]
    run_id = result["metadata"].run_id
    assert any(path.name.startswith("matmul_benchmark_sweepM_") for path in outputs)
    assert any(path.name.startswith("compare_latency_sweepM_K32_N64_none_") for path in outputs)
    assert any(path.suffix == ".json" for path in outputs)
    assert any(path.suffix == ".txt" for path in outputs)
    assert any(path.parent == tmp_path / "timing" / "matmul" / run_id for path in outputs if path.suffix == ".json")
    assert any(path.parent == tmp_path / "timing" / "matmul" / run_id for path in outputs if path.suffix == ".txt")
    assert all(path.parent == tmp_path / "plots" / "matmul" / run_id for path in outputs if path.suffix == ".png")
