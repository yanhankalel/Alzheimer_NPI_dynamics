"""BrainX implementation of M0 and M1 effective rate dynamics.

This module is imported only by the dedicated BrainX interpreter.  Effective
coupling is not passed to ``brainmass.Network`` because it is not structural
connectivity.  ``brainmass.additive_coupling`` is used directly with the frozen
``[target,source]`` orientation, while ``brainmass.Simulator`` owns the compiled
time loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import brainmass
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class RateModelConfig:
    tau_seconds: float
    a0: float
    b0: float
    sigma_per_sqrt_second: float
    nonlinear: bool

    def __post_init__(self) -> None:
        values = (self.tau_seconds, self.a0, self.b0, self.sigma_per_sqrt_second)
        if not np.isfinite(values).all() or self.tau_seconds <= 0 or self.sigma_per_sqrt_second < 0:
            raise ValueError("invalid global rate-model configuration")


class EffectiveRateStep(brainstate.nn.Module):
    """One global-parameter state variable per ROI."""

    def __init__(self, coupling_target_source: np.ndarray, config: RateModelConfig):
        super().__init__()
        coupling = np.asarray(coupling_target_source, float)
        if coupling.ndim != 2 or coupling.shape[0] != coupling.shape[1]:
            raise ValueError("coupling must be square [target,source]")
        if not np.isfinite(coupling).all() or not np.allclose(np.diag(coupling), 0.0):
            raise ValueError("coupling must be finite with a zero diagonal")
        self.coupling = jnp.asarray(coupling)
        self.config = config
        self.x = brainstate.HiddenState(jnp.zeros(coupling.shape[0]))

    def update(self):
        x = self.x.value
        source_by_target = jnp.broadcast_to(x, self.coupling.shape)
        coupled = brainmass.additive_coupling(source_by_target, self.coupling, 1.0)
        drive = self.config.a0 * x + self.config.b0 + coupled
        target = jax_sigmoid(drive) if self.config.nonlinear else drive
        drift_per_second = (-x + target) / self.config.tau_seconds
        dt_seconds = u.get_magnitude(brainstate.environ.get_dt().in_unit(u.second))
        if self.config.sigma_per_sqrt_second:
            diffusion = (
                self.config.sigma_per_sqrt_second
                * jnp.sqrt(dt_seconds)
                * brainstate.random.randn(*x.shape)
            )
        else:
            diffusion = 0.0
        self.x.value = x + dt_seconds * drift_per_second + diffusion
        return self.x.value


def jax_sigmoid(value):
    return 1.0 / (1.0 + jnp.exp(-value))


def simulate_rate_model(
    coupling_target_source: np.ndarray,
    config: RateModelConfig,
    duration_seconds: float,
    dt_seconds: float,
    seed: int,
    initial_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run M0/M1 through the BrainMass simulator with exact seed replay."""

    if duration_seconds <= 0 or dt_seconds <= 0:
        raise ValueError("duration and dt must be positive")
    brainstate.random.seed(int(seed))
    model = EffectiveRateStep(coupling_target_source, config)
    if initial_state is not None:
        initial = np.asarray(initial_state, float)
        if initial.shape != model.x.value.shape or not np.isfinite(initial).all():
            raise ValueError("initial_state shape/data mismatch")
        model.x.value = jnp.asarray(initial)
    result = brainmass.Simulator(model, dt=dt_seconds * u.second).run(
        duration_seconds * u.second,
        monitors=["x"],
        init_states=False,
    )
    return np.asarray(u.get_magnitude(result["ts"]), float), np.asarray(u.get_magnitude(result["x"]), float)

