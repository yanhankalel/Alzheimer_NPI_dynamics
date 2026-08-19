"""Synthetic recovery utilities for the frozen low-rank coupling family."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coupling_v1 import CouplingParameters, validate_fixed_modes


@dataclass(frozen=True)
class RecoveryMetrics:
    relative_parameter_error: float
    prediction_rmse: float
    parameter_correlation: float


@dataclass(frozen=True)
class RecoveryThresholds:
    max_relative_parameter_error: float
    max_prediction_rmse: float
    min_parameter_correlation: float


def recover_mode_amplitudes(
    coupling_samples: np.ndarray,
    baseline_target_source: np.ndarray,
    modes: np.ndarray,
) -> np.ndarray:
    """Project coupling residuals onto fixed modes without assuming orthogonality."""

    samples = np.asarray(coupling_samples, float)
    baseline = np.asarray(baseline_target_source, float)
    modes = validate_fixed_modes(modes)
    if samples.ndim != 3 or samples.shape[1:] != baseline.shape or baseline.shape != modes.shape[1:]:
        raise ValueError("coupling samples, baseline, and modes do not align")
    design = modes.reshape(len(modes), -1).T
    residual = (samples - baseline).reshape(len(samples), -1)
    amplitudes, _, rank, _ = np.linalg.lstsq(design, residual.T, rcond=None)
    if rank < len(modes):
        raise ValueError("fixed coupling modes are linearly dependent")
    return amplitudes.T


def recover_disease_coefficients(
    amplitudes: np.ndarray,
    stage_features: np.ndarray,
    amyloid_residual: np.ndarray,
    tau_residual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover beta_d, beta_A and beta_T from an identified synthetic design."""

    y = np.asarray(amplitudes, float)
    g = np.asarray(stage_features, float)
    rA = np.asarray(amyloid_residual, float)
    rT = np.asarray(tau_residual, float)
    if y.ndim != 2 or g.ndim != 2 or rA.shape != (len(y),) or rT.shape != (len(y),):
        raise ValueError("synthetic recovery arrays do not align")
    design = np.column_stack((g, rA, rT))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("synthetic disease design is rank deficient")
    predicted = design @ coefficients
    n_stage = g.shape[1]
    return coefficients[:n_stage].T, coefficients[n_stage], coefficients[n_stage + 1], predicted


def recovery_metrics(
    truth: CouplingParameters,
    recovered: CouplingParameters,
    observed_amplitudes: np.ndarray,
    predicted_amplitudes: np.ndarray,
) -> RecoveryMetrics:
    true_vector = np.concatenate((np.ravel(truth.beta_d), truth.beta_a, truth.beta_t))
    recovered_vector = np.concatenate((np.ravel(recovered.beta_d), recovered.beta_a, recovered.beta_t))
    denominator = max(float(np.linalg.norm(true_vector)), np.finfo(float).eps)
    relative_error = float(np.linalg.norm(recovered_vector - true_vector) / denominator)
    rmse = float(np.sqrt(np.mean((np.asarray(observed_amplitudes) - np.asarray(predicted_amplitudes)) ** 2)))
    if np.std(true_vector) == 0 or np.std(recovered_vector) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(true_vector, recovered_vector)[0, 1])
    return RecoveryMetrics(relative_error, rmse, correlation)


def passes_recovery(metrics: RecoveryMetrics, thresholds: RecoveryThresholds) -> bool:
    """Apply only explicitly supplied thresholds; no defaults are permitted."""

    return bool(
        np.isfinite((metrics.relative_parameter_error, metrics.prediction_rmse, metrics.parameter_correlation)).all()
        and metrics.relative_parameter_error <= thresholds.max_relative_parameter_error
        and metrics.prediction_rmse <= thresholds.max_prediction_rmse
        and metrics.parameter_correlation >= thresholds.min_parameter_correlation
    )
