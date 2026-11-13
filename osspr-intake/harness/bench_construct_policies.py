"""Baseline for pymdp.control.construct_policies.

Workload: exhaustive policies for one controllable factor with 5 actions and
horizon 6 (15625 policies). This is the public policy-matrix constructor used
when a caller materializes the full action tree before planning.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4",
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _timing import emit, measure
from pymdp.control import construct_policies


def main() -> None:
    def call():
        return construct_policies([9], [5], policy_len=6)

    stats = measure(call, repeats=7, warmup=2, sync_output=True)
    policies = stats["result"]
    emit(
        "pymdp.control.construct_policies",
        stats,
        float(policies.shape[0] * 1000 + policies[0, 0, 0] + policies[-1, -1, 0]),
    )


if __name__ == "__main__":
    main()
