"""Orientation-safe bridge from M0/M1 to the three-lag NPI estimand."""

from __future__ import annotations

import numpy as np


def discrete_rate_jacobian(
    state: np.ndarray,
    coupling_target_source: np.ndarray,
    tau_seconds: float,
    a0: float,
    b0: float,
    dt_seconds: float,
    nonlinear: bool,
) -> np.ndarray:
    """Jacobian of one Euler step, with output rows as targets and columns as sources."""

    x = np.asarray(state, float)
    C = np.asarray(coupling_target_source, float)
    if x.ndim != 1 or C.shape != (len(x), len(x)) or tau_seconds <= 0 or dt_seconds <= 0:
        raise ValueError("invalid state/coupling/time contract")
    local_and_coupling = a0 * np.eye(len(x)) + C
    if nonlinear:
        drive = a0 * x + b0 + C @ x
        phi = 1.0 / (1.0 + np.exp(-drive))
        derivative = np.diag(phi * (1.0 - phi)) @ local_and_coupling
    else:
        derivative = local_and_coupling
    return np.eye(len(x)) + (dt_seconds / tau_seconds) * (-np.eye(len(x)) + derivative)


def markov_jacobian_as_npi_lags(one_step_jacobian: np.ndarray) -> np.ndarray:
    """Embed a first-order model in K[oldest,middle,latest,target,source]."""

    jacobian = np.asarray(one_step_jacobian, float)
    if jacobian.ndim != 2 or jacobian.shape[0] != jacobian.shape[1] or not np.isfinite(jacobian).all():
        raise ValueError("one-step Jacobian must be finite and square")
    K = np.zeros((3, *jacobian.shape), dtype=float)
    K[2] = jacobian
    return K


def aggregate_npi_seeds(seed_jacobians: np.ndarray, expected_seeds: int = 5) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(seed_jacobians, float)
    if values.ndim != 4 or values.shape[0] != expected_seeds or values.shape[1] != 3:
        raise ValueError(f"NPI Jacobians must be [{expected_seeds},3,target,source]")
    if values.shape[2] != values.shape[3] or not np.isfinite(values).all():
        raise ValueError("NPI Jacobians must be finite and square on target/source axes")
    return values.mean(axis=0), values.std(axis=0, ddof=1)

