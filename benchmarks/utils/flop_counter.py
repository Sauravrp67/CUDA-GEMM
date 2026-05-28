from dataclasses import dataclass
from typing import Literal

BYTES_PER_DTYPE: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}


@dataclass(frozen=True)
class KernelStats:
    flops: int
    bytes_minimum: int
    bytes_no_reuse: int

    @property
    def ai_minimum(self) -> float:
        return self.flops / self.bytes_minimum

    @property
    def ai_no_reuse(self) -> float:
        return self.flops / self.bytes_no_reuse

    def regime(self, ridge_point: float) -> str:
        return "MEMORY-BOUND" if self.ai_no_reuse < ridge_point else "COMPUTE-BOUND"


def gemm_stats(
    M: int,
    K: int,
    N: int,
    dtype: Literal["float32", "float16", "bfloat16"] = "float32",
) -> KernelStats:
    bpe = BYTES_PER_DTYPE[dtype]
    flops = 2 * M * K * N
    bytes_moved = (M * K + K * N + M * N) * bpe
    return KernelStats(
        flops=flops,
        bytes_minimum=bytes_moved,
        bytes_no_reuse=bytes_moved,
    )
