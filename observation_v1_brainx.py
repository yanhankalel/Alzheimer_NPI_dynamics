"""Synthetic candidate BOLD observer and BrainX-native summary metrics.

The 607-ms sampling contract is frozen, but the HRF family is not.  Therefore
this module is available for synthetic engineering/recovery only until an HRF
protocol file clears the real-data blocker.
"""

from __future__ import annotations

import brainmass
import brainunit as u
import braintools
import numpy as np


def candidate_hrf_bold(
    neural_time_by_roi: np.ndarray,
    neural_dt_seconds: float,
    tr_seconds: float = 0.607,
    internal_average_seconds: float = 0.001,
) -> np.ndarray:
    neural = np.asarray(neural_time_by_roi, float)
    if neural.ndim != 2 or neural.shape[0] < 2 or not np.isfinite(neural).all():
        raise ValueError("neural trajectory must be finite [time,roi]")
    if neural_dt_seconds <= 0 or tr_seconds <= 0 or internal_average_seconds <= 0:
        raise ValueError("observation time steps must be positive")
    observer = brainmass.HRFBold(
        period=tr_seconds * u.second,
        downsample_period=internal_average_seconds * u.second,
        kernel=brainmass.GammaHRFKernel(),
    )
    bold = observer(neural, dt=neural_dt_seconds * u.second)
    result = np.asarray(u.get_magnitude(bold), float)
    if result.ndim != 2 or result.shape[1] != neural.shape[1] or not np.isfinite(result).all():
        raise ValueError(f"invalid HRF output shape/data: {result.shape}")
    return result


def bold_summaries(
    bold_time_by_roi: np.ndarray,
    tr_seconds: float = 0.607,
    fcd_window_tr: int = 20,
    fcd_step_tr: int = 5,
) -> dict[str, np.ndarray]:
    bold = np.asarray(bold_time_by_roi, float)
    if bold.ndim != 2 or bold.shape[0] < fcd_window_tr or not np.isfinite(bold).all():
        raise ValueError("BOLD must be finite [time,roi] and cover one FCD window")
    fc = np.asarray(braintools.metric.functional_connectivity(bold), float)
    fcd = np.asarray(
        braintools.metric.functional_connectivity_dynamics(
            bold,
            window_size=int(fcd_window_tr),
            step_size=int(fcd_step_tr),
        ),
        float,
    )
    frequencies, first_psd = braintools.metric.power_spectral_density(bold[:, 0], tr_seconds * 1000.0)
    power = [np.asarray(first_psd, float)]
    for roi in range(1, bold.shape[1]):
        roi_frequencies, roi_power = braintools.metric.power_spectral_density(
            bold[:, roi], tr_seconds * 1000.0
        )
        if not np.array_equal(np.asarray(roi_frequencies), np.asarray(frequencies)):
            raise ValueError("PSD frequency grids differ across ROIs")
        power.append(np.asarray(roi_power, float))
    result = {
        "fc": fc,
        "fcd": fcd,
        "psd_frequencies_per_ms": np.asarray(frequencies, float),
        "psd_power_frequency_by_roi": np.stack(power, axis=1),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("non-finite BOLD summary")
    return result

