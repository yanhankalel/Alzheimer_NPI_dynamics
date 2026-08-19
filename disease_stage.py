"""Interval-censored latent disease-stage inference.

The event time is inferred on a one-dimensional grid.  The fMRI cohort is an
application cohort only; curve fitting must use a separately supplied reference
cohort that excludes every fMRI subject.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MonotoneCurve:
    name: str
    stage_years: np.ndarray
    mean: np.ndarray
    increasing: bool

    def __post_init__(self) -> None:
        x = np.asarray(self.stage_years, float)
        y = np.asarray(self.mean, float)
        if x.ndim != 1 or y.shape != x.shape or x.size < 2:
            raise ValueError(f"{self.name}: curve arrays must be equal one-dimensional shapes")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
            raise ValueError(f"{self.name}: curve grid must be finite and strictly increasing")
        direction = np.diff(y)
        if self.increasing and np.any(direction < -1e-12):
            raise ValueError(f"{self.name}: expected an increasing curve")
        if not self.increasing and np.any(direction > 1e-12):
            raise ValueError(f"{self.name}: expected a decreasing curve")

    def predict(self, stage_years: np.ndarray) -> np.ndarray:
        return np.interp(stage_years, self.stage_years, self.mean)


@dataclass(frozen=True)
class MarkerObservation:
    marker: str
    time_years: float
    value: float
    sd: float

    def __post_init__(self) -> None:
        if not np.isfinite((self.time_years, self.value, self.sd)).all() or self.sd <= 0:
            raise ValueError("marker observation must be finite with sd > 0")


@dataclass(frozen=True)
class ConversionInterval:
    """Event-time bounds; either side may be absent for one-sided censoring."""

    last_negative_year: float | None = None
    first_positive_year: float | None = None

    def __post_init__(self) -> None:
        lo, hi = self.last_negative_year, self.first_positive_year
        if lo is not None and not np.isfinite(lo):
            raise ValueError("last-negative time must be finite")
        if hi is not None and not np.isfinite(hi):
            raise ValueError("first-positive time must be finite")
        if lo is not None and hi is not None and lo >= hi:
            raise ValueError("conversion interval must satisfy last_negative < first_positive")


@dataclass(frozen=True)
class StagePosterior:
    event_time_mean: float
    event_time_sd: float
    event_time_q025: float
    event_time_q50: float
    event_time_q975: float
    grid: np.ndarray
    probability: np.ndarray
    fit_status: str

    def at_scan(self, scan_time_years: float) -> dict[str, float | str]:
        return {
            "d_mean": float(scan_time_years - self.event_time_mean),
            "d_sd": self.event_time_sd,
            "fit_status": self.fit_status,
        }


@dataclass(frozen=True)
class MarkerResidual:
    raw_mean: float
    raw_sd_from_stage: float
    standardized_mean: float
    status: str


def _quantile(grid: np.ndarray, probability: np.ndarray, q: float) -> float:
    return float(np.interp(q, np.cumsum(probability), grid))


def infer_event_time(
    event_time_grid: np.ndarray,
    curves: Mapping[str, MonotoneCurve],
    observations: Sequence[MarkerObservation],
    conversion: ConversionInterval | None,
    prior_mean: float,
    prior_sd: float,
) -> StagePosterior:
    """Infer the latent event time from markers and censoring constraints."""

    grid = np.asarray(event_time_grid, float)
    if grid.ndim != 1 or grid.size < 3 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("event_time_grid must be finite, one-dimensional, and increasing")
    if not np.isfinite((prior_mean, prior_sd)).all() or prior_sd <= 0:
        raise ValueError("prior must be finite with prior_sd > 0")

    logp = -0.5 * ((grid - prior_mean) / prior_sd) ** 2
    used_markers = 0
    for observation in observations:
        if observation.marker not in curves:
            raise KeyError(f"missing frozen curve for marker {observation.marker!r}")
        stage = observation.time_years - grid
        expected = curves[observation.marker].predict(stage)
        logp -= 0.5 * ((observation.value - expected) / observation.sd) ** 2
        used_markers += 1

    if conversion is not None:
        allowed = np.ones(grid.shape, dtype=bool)
        if conversion.last_negative_year is not None:
            allowed &= grid > conversion.last_negative_year
        if conversion.first_positive_year is not None:
            allowed &= grid <= conversion.first_positive_year
        logp = np.where(allowed, logp, -np.inf)
    if not np.isfinite(logp).any():
        raise ValueError("event-time grid has no support inside the censoring interval")

    logp -= np.max(logp)
    probability = np.exp(logp)
    probability /= probability.sum()
    mean = float(np.sum(grid * probability))
    variance = float(np.sum((grid - mean) ** 2 * probability))
    if conversion is not None and used_markers:
        status = "POSTERIOR_MARKERS_AND_CENSORING"
    elif conversion is not None:
        status = "POSTERIOR_CENSORING_ONLY"
    elif used_markers:
        status = "POSTERIOR_MARKERS_ONLY"
    else:
        status = "POSTERIOR_PRIOR_ONLY"
    return StagePosterior(
        event_time_mean=mean,
        event_time_sd=float(np.sqrt(max(variance, 0.0))),
        event_time_q025=_quantile(grid, probability, 0.025),
        event_time_q50=_quantile(grid, probability, 0.5),
        event_time_q975=_quantile(grid, probability, 0.975),
        grid=grid,
        probability=probability,
        fit_status=status,
    )


def validate_reference_exclusion(reference_subjects: Sequence[str], fmri_subjects: Sequence[str]) -> None:
    def canonical(subject: str) -> str:
        value = str(subject).strip().upper().replace("SUB-ADNI", "").replace("ADNI", "")
        compact = value.replace("_", "")
        match = re.fullmatch(r"(\d{3})S(\d{4})", compact)
        return f"{match.group(1)}_S_{match.group(2)}" if match else value

    overlap = sorted({canonical(subject) for subject in reference_subjects}.intersection(
        canonical(subject) for subject in fmri_subjects
    ))
    if overlap:
        raise ValueError(f"disease-curve reference cohort leaks {len(overlap)} fMRI subjects; first={overlap[0]}")


def posterior_marker_residual(
    curve: MonotoneCurve,
    observation: MarkerObservation,
    posterior: StagePosterior,
) -> MarkerResidual:
    """Return rA/rT-style residual while propagating event-time uncertainty."""

    if observation.marker != curve.name:
        raise ValueError("observation marker does not match the frozen curve")
    predicted = curve.predict(observation.time_years - posterior.grid)
    expected = float(np.sum(predicted * posterior.probability))
    prediction_variance = float(np.sum((predicted - expected) ** 2 * posterior.probability))
    raw = float(observation.value - expected)
    return MarkerResidual(
        raw_mean=raw,
        raw_sd_from_stage=float(np.sqrt(max(prediction_variance, 0.0))),
        standardized_mean=raw / observation.sd,
        status=f"RESIDUAL_FROM_{posterior.fit_status}",
    )
