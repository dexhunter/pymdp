"""Baseline for pymdp.legacy.control.update_posterior_policies_factorized.

Workload: one decision step of the NumPy agent on a 4x4 location factor plus a
4-level context factor, five actions, planning horizon 4 (625 policies). Utility
and state information gain are both on, which is the default infer_policies path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _timing import emit, measure

from pymdp.legacy import utils as legacy_utils
from pymdp.legacy.control import update_posterior_policies_factorized
from pymdp.legacy.maths import spm_log_obj_array


def build_model():
    rng = np.random.default_rng(0)
    num_states = [16, 4]
    num_obs = [16, 3]
    num_controls = [5, 1]
    A = legacy_utils.obj_array(2)
    A[0] = rng.random((num_obs[0], num_states[0])) + 0.05
    A[0] /= A[0].sum(axis=0, keepdims=True)
    A[1] = rng.random((num_obs[1], num_states[0], num_states[1])) + 0.05
    A[1] /= A[1].sum(axis=0, keepdims=True)
    B = legacy_utils.obj_array(2)
    for factor, (states, controls) in enumerate(zip(num_states, num_controls)):
        raw = rng.random((states, states, controls)) + 0.05
        B[factor] = raw / raw.sum(axis=0, keepdims=True)
    C = spm_log_obj_array(legacy_utils.obj_array_uniform(num_obs))
    C[0] = np.log(np.array([0.02] * 15 + [0.70]))
    qs = legacy_utils.obj_array_uniform(num_states)
    policies = []
    for actions in np.ndindex(*([num_controls[0]] * 4)):
        policy = np.zeros((4, 2), dtype=int)
        policy[:, 0] = actions
        policies.append(policy)
    return qs, A, B, C, policies


def main() -> None:
    qs, A, B, C, policies = build_model()

    def call():
        return update_posterior_policies_factorized(
            qs,
            A,
            B,
            C,
            [[0], [0, 1]],
            [[0], [1]],
            policies,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=False,
            gamma=16.0,
        )

    stats = measure(call, repeats=7, warmup=1, sync_output=False)
    q_pi, _neg_efe = stats["result"]
    emit(
        "pymdp.legacy.control.update_posterior_policies_factorized",
        stats,
        float(np.sum(q_pi)),
    )


if __name__ == "__main__":
    main()
