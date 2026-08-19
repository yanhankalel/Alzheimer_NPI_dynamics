"""Low-rank disease-conditioned effective coupling with [target, source] axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class CouplingParameters:
    H0: np.ndarray
    modes: np.ndarray
    beta_d: np.ndarray
    beta_a: np.ndarray
    beta_t: np.ndarray


def validate_mapping(P: np.ndarray, n_rois: int = 90, n_networks: int = 9) -> np.ndarray:
    P = np.asarray(P, float)
    if P.shape != (n_rois, n_networks) or not np.isfinite(P).all():
        raise ValueError(f"P must have shape {(n_rois, n_networks)}")
    if np.any(P < 0) or not np.allclose(P.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("P must be a nonnegative fractional mapping with each row summing to one")
    if np.linalg.matrix_rank(P) < n_networks:
        raise ValueError("P must contain all network columns")
    return P


def normalize_modes(modes: np.ndarray) -> np.ndarray:
    """Zero diagonals, unit-normalize, and fix mode signs deterministically."""

    result = np.asarray(modes, float).copy()
    if result.ndim != 3 or result.shape[1] != result.shape[2] or not np.isfinite(result).all():
        raise ValueError("modes must have shape [mode,target,source]")
    diagonal = np.arange(result.shape[1])
    for index in range(result.shape[0]):
        result[index, diagonal, diagonal] = 0.0
        norm = np.linalg.norm(result[index])
        if norm <= 0:
            raise ValueError(f"mode {index} is zero after diagonal removal")
        result[index] /= norm
        flat = result[index].ravel()
        pivot = int(np.argmax(np.abs(flat)))
        if flat[pivot] < 0:
            result[index] *= -1.0
    return result


def validate_fixed_modes(modes: np.ndarray) -> np.ndarray:
    result = np.asarray(modes, float)
    if result.ndim != 3 or result.shape[1] != result.shape[2] or not np.isfinite(result).all():
        raise ValueError("modes must have shape [mode,target,source]")
    diagonal = np.arange(result.shape[1])
    if not np.allclose(result[:, diagonal, diagonal], 0.0, atol=1e-12):
        raise ValueError("fixed modes must have zero diagonals")
    if not np.allclose(np.linalg.norm(result, axis=(1, 2)), 1.0, atol=1e-10):
        raise ValueError("fixed modes must have unit Frobenius norm")
    for index, mode in enumerate(result):
        pivot = int(np.argmax(np.abs(mode)))
        if mode.ravel()[pivot] < 0:
            raise ValueError(f"fixed mode {index} does not use the deterministic sign convention")
    return result


def build_effective_coupling(
    P: np.ndarray,
    parameters: CouplingParameters,
    d: float,
    rA: float,
    rT: float,
    stage_design: Callable[[float], np.ndarray],
) -> np.ndarray:
    """Build C[target,source] without an unconstrained ROI-by-ROI matrix."""

    P = validate_mapping(P, P.shape[0], P.shape[1])
    H0 = np.asarray(parameters.H0, float)
    modes = validate_fixed_modes(parameters.modes)
    n_modes = modes.shape[0]
    if H0.shape != (P.shape[1], P.shape[1]):
        raise ValueError("H0 must be [network,target, network,source]")
    g = np.atleast_1d(np.asarray(stage_design(float(d)), float))
    beta_d = np.asarray(parameters.beta_d, float)
    beta_a = np.asarray(parameters.beta_a, float)
    beta_t = np.asarray(parameters.beta_t, float)
    if beta_d.shape != (n_modes, g.size) or beta_a.shape != (n_modes,) or beta_t.shape != (n_modes,):
        raise ValueError("disease coefficient dimensions do not match the fixed modes")
    if not all(np.isfinite(x).all() for x in (H0, g, beta_d, beta_a, beta_t)):
        raise ValueError("coupling inputs must be finite")
    amplitudes = beta_d @ g + beta_a * float(rA) + beta_t * float(rT)
    C = P @ H0 @ P.T + np.einsum("l,lij->ij", amplitudes, modes)
    C = np.asarray(C, float)
    np.fill_diagonal(C, 0.0)
    return C


def extract_training_modes(
    ec: np.ndarray,
    subject_ids: Sequence[str],
    training_subjects: Sequence[str],
    P: np.ndarray,
    n_modes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate H0 and residual modes using training subjects only.

    Sessions are first averaged within subject, so subjects contribute equally.
    The returned matrices follow [target,source] orientation.
    """

    values = np.asarray(ec, float)
    if values.ndim != 3 or values.shape[1] != values.shape[2] or values.shape[0] != len(subject_ids):
        raise ValueError("ec must be [session,target,source] and align with subject_ids")
    P = validate_mapping(P, values.shape[1], P.shape[1])
    train = set(training_subjects)
    available = set(subject_ids)
    if not train or not train.issubset(available):
        raise ValueError("training_subjects must be a nonempty subset of subject_ids")
    subject_means = []
    for subject in sorted(train):
        indices = [i for i, candidate in enumerate(subject_ids) if candidate == subject]
        subject_means.append(values[indices].mean(axis=0))
    subject_means = np.asarray(subject_means)
    mean_ec = subject_means.mean(axis=0)
    pinv = np.linalg.pinv(P)
    H0 = pinv @ mean_ec @ pinv.T
    baseline = P @ H0 @ P.T
    residual = subject_means - baseline
    diagonal = np.arange(values.shape[1])
    residual[:, diagonal, diagonal] = 0.0
    centered = residual - residual.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered.reshape(len(centered), -1), full_matrices=False)
    rank = int(np.sum(singular_values > np.finfo(float).eps * max(centered.shape) * singular_values[0])) if singular_values.size else 0
    if n_modes < 1 or n_modes > rank:
        raise ValueError(f"requested {n_modes} modes but training residual rank is {rank}")
    modes = normalize_modes(vt[:n_modes].reshape(n_modes, values.shape[1], values.shape[2]))
    return H0, modes
