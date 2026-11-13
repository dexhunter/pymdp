"""Baseline for pymdp.maths.calc_vfe.

Workload: sequence free energy of one 4096-step trajectory for a two-factor
model (20 and 16 states), two observation modalities, and Dirichlet parameter
KL terms. Arrays are arguments of the jitted call so the score is not
constant-folded at compile time.
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
from pymdp.maths import calc_vfe

jax.config.update("jax_enable_x64", False)


def _norm_cols(key, shape):
    raw = random.uniform(key, shape, minval=0.05, maxval=1.0)
    return raw / raw.sum(axis=0, keepdims=True)


def main() -> None:
    timesteps = 4096
    keys = random.split(random.PRNGKey(2), 8)
    A = [_norm_cols(keys[0], (32, 20, 16)), _norm_cols(keys[1], (10, 20))]
    B = [_norm_cols(keys[2], (20, 20, 5)), _norm_cols(keys[3], (16, 16, 1))]
    qs = [
        jax.nn.softmax(random.normal(keys[4], (timesteps, 20)), axis=-1),
        jax.nn.softmax(random.normal(keys[5], (timesteps, 16)), axis=-1),
    ]
    obs = [
        jax.nn.softmax(random.normal(keys[6], (timesteps, 32)), axis=-1),
        jax.nn.softmax(random.normal(keys[7], (timesteps, 10)), axis=-1),
    ]
    prior = [jnp.ones(20) / 20, jnp.ones(16) / 16]
    actions = jnp.stack(
        [
            random.randint(keys[4], (timesteps - 1,), 0, 5),
            jnp.zeros((timesteps - 1,), dtype=jnp.int32),
        ],
        axis=-1,
    )
    qA = [tensor * 6.0 + 1.0 for tensor in A]
    pA = [jnp.ones_like(tensor) for tensor in A]
    qB = [tensor * 6.0 + 1.0 for tensor in B]
    pB = [jnp.ones_like(tensor) for tensor in B]

    @jax.jit
    def run(qs, prior, obs, A, B, actions, qA, pA, qB, pB):
        return calc_vfe(
            qs,
            prior,
            obs=obs,
            A=A,
            B=B,
            past_actions=actions,
            A_dependencies=[[0, 1], [0]],
            B_dependencies=[[0], [1]],
            qA=qA,
            pA=pA,
            qB=qB,
            pB=pB,
            distr_obs=True,
        )

    stats = measure(
        lambda: run(qs, prior, obs, A, B, actions, qA, pA, qB, pB),
        repeats=7,
        warmup=2,
        sync_output=True,
    )
    _vfe_t, vfe = stats["result"]
    emit("pymdp.maths.calc_vfe", stats, float(vfe))


if __name__ == "__main__":
    main()
