"""Baseline for pymdp.control.update_posterior_policies.

Workload: one exhaustive policy-posterior update for a three-factor model
(location 12 x context 9 x cue 6), controls (4, 3, 1), horizon 3
(1728 policies). Utility, state information gain, and parameter information
gain are all enabled. Arrays are passed into the jitted call so the compile
does not constant-fold the score.
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

import jax
import jax.numpy as jnp
from jax import random

from _timing import emit, measure
from pymdp.control import construct_policies, update_posterior_policies
from pymdp.maths import log_stable

jax.config.update("jax_enable_x64", False)


def _norm_cols(key, shape):
    raw = random.uniform(key, shape, minval=0.05, maxval=1.0)
    return raw / raw.sum(axis=0, keepdims=True)


def build_model():
    keys = random.split(random.PRNGKey(1), 8)
    A = [
        _norm_cols(keys[0], (24, 12, 9)),
        _norm_cols(keys[1], (10, 12, 6)),
        _norm_cols(keys[2], (8, 9, 6)),
    ]
    B = [
        _norm_cols(keys[3], (12, 12, 4)),
        _norm_cols(keys[4], (9, 9, 3)),
        _norm_cols(keys[5], (6, 6, 1)),
    ]
    C = [log_stable(jnp.ones(n) / n) for n in (24, 10, 8)]
    qs = [jnp.ones(n) / n for n in (12, 9, 6)]
    policies = construct_policies([12, 9, 6], [4, 3, 1], policy_len=3)
    E = jnp.ones(policies.shape[0]) / policies.shape[0]
    pA = [tensor * 5.0 + 1.0 for tensor in A]
    pB = [tensor * 5.0 + 1.0 for tensor in B]
    return policies, qs, A, B, C, E, pA, pB


def main() -> None:
    policies, qs, A, B, C, E, pA, pB = build_model()
    A_dependencies = [[0, 1], [0, 2], [1, 2]]
    B_dependencies = [[0], [1], [2]]

    @jax.jit
    def run(policies, qs, A, B, C, E, pA, pB):
        return update_posterior_policies(
            policies,
            qs,
            A,
            B,
            C,
            E,
            pA,
            pB,
            A_dependencies,
            B_dependencies,
            gamma=16.0,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
        )

    stats = measure(
        lambda: run(policies, qs, A, B, C, E, pA, pB),
        repeats=7,
        warmup=3,
        sync_output=True,
    )
    q_pi, _neg_efe = stats["result"]
    emit("pymdp.control.update_posterior_policies", stats, float(jnp.sum(q_pi)))


if __name__ == "__main__":
    main()
