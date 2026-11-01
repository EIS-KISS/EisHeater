

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic_ai import RunContext

from .eisgen import eisgenerator
from .fitting import eisfit, report, run_fit


@dataclass
class SpectrumDeps:
    omega: np.ndarray  # original full-resolution omega, for slicing fit windows
    re: np.ndarray
    im: np.ndarray
    res_omega: np.ndarray  # 20 log-spaced points, the simulate grid
    res_re: np.ndarray
    res_im: np.ndarray
    n_original: int = 0
    omega_range: tuple[float, float] = (0.0, 0.0)
    extra: dict = field(default_factory=dict)


def _fit_window(
    deps: SpectrumDeps, omega_min: float | None, omega_max: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    if omega_min is None and omega_max is None:
        return deps.omega, deps.re, deps.im, ""
    mask = np.ones(deps.omega.size, dtype=bool)
    if omega_min is not None:
        mask &= deps.omega >= omega_min
    if omega_max is not None:
        mask &= deps.omega <= omega_max
    if int(mask.sum()) < 2:
        raise ValueError(f"omega window [{omega_min}, {omega_max}] selects fewer than 2 points")
    w, r, i = deps.omega[mask], deps.re[mask], deps.im[mask]
    note = f"Fitting a slice: {int(mask.sum())} points, omega {w.min():.4e} to {w.max():.4e} rad/s"
    return w, r, i, note


def fit_report(
    omega: np.ndarray,
    re: np.ndarray,
    im: np.ndarray,
    model_str: str,
    note: str = "",
) -> str:
    try:
        result = run_fit(omega, re, im, model_str)
    except eisfit.ModelParseError as e:
        return f"could not parse the circuit {model_str!r}: {e}"
    except (eisfit.FitError, ValueError) as e:
        return f"eisfit could not fit {model_str!r}: {e}"
    return report(result, note)


def register_tools(agent) -> None:
    @agent.tool
    def fit_spectrum(
        ctx: RunContext[SpectrumDeps],
        model_str: str,
        omega_min: float | None = None,
        omega_max: float | None = None,
    ) -> str:
        """Fit a candidate circuit to the spectrum with eisfit.

        model_str is a KISS-encoded circuit, e.g. "r{50}-p{1e-4, 0.8}". Each parameter can be given
        in three ways:
        - a single value, e.g. r{100}: the starting point of the fit
        - a range, e.g. r{1~1000}: bounds the fit to that interval. Use this to keep a parameter in
          a physically sensible range
        - omitted, e.g. r: starts in the middle of that element's default range.
        Optionally restrict the fit to a frequency window: give omega_min and/or omega_max (rad/s)
        and only the measured points inside [omega_min, omega_max] are fitted. Use this to fit a
        single feature (e.g. one arc) in isolation. The window must select at least two points.
        Returns the full eisfit report: the fitted parameters, the fit quality (residuals, chi^2),
        and a verdict on whether the circuit actually fits the data. Use it to check how well a
        candidate explains the spectrum."""
        deps = ctx.deps
        try:
            omega, re, im, note = _fit_window(deps, omega_min, omega_max)
        except ValueError as e:
            return f"could not fit: {e}"
        return fit_report(omega, re, im, model_str, note)

    @agent.tool
    def simulate_spectrum(ctx: RunContext[SpectrumDeps], model_str: str) -> str:
        """Generate a spectrum from a circuit with eisgenerator and return its values.

        model_str is a KISS-encoded circuit with concrete parameter values, e.g.
        "r{50}-p{1e-4, 0.8}". The circuit is evaluated on the same 20 log-spaced frequency points
        as the resampled measured spectrum, so you can compare the generated curve point for point
        against the measured one. If a parameter is given as a range, its first value is used.
        Returns the concrete model string used and, for each of the 20 points, the generated
        omega, Re(Z), Im(Z), |Z|, and phase, plus the residual against the measured spectrum:
        dRe = Re(sim)-Re(meas) and dIm = Im(sim)-Im(meas). Small residuals mean the circuit
        matches the data at that frequency."""
        deps = ctx.deps
        try:
            model = eisgenerator.Model(model_str)
        except Exception as e:
            return f"could not build the circuit {model_str!r}: {e}"
        concrete = model.getModelStrWithParam(0)
        try:
            points = model.executeSweep(deps.res_omega.tolist(), 0)
        except Exception as e:
            return f"could not evaluate the circuit {model_str!r}: {e}"
        omega = np.array([p.omega for p in points])
        re = np.array([p.im.real for p in points])
        im = np.array([p.im.imag for p in points])
        # Residuals vs the measured 20-point spectrum on the same grid: sim minus measured.
        d_re = re - deps.res_re
        d_im = im - deps.res_im
        mag = np.hypot(re, im)
        phase = np.degrees(np.arctan2(im, re))
        rows = [
            f"{w:.6e}, {r:.6e}, {i:.6e}, {m:.6e}, {p:.4f}, {dr:.6e}, {di:.6e}"
            for w, r, i, m, p, dr, di in zip(omega, re, im, mag, phase, d_re, d_im)
        ]
        table = (
            "omega [rad/s], Re(Z) [ohm], Im(Z) [ohm], |Z| [ohm], phase [deg], "
            "dRe = Re(sim)-Re(meas) [ohm], dIm = Im(sim)-Im(meas) [ohm]\n" + "\n".join(rows)
        )
        header = f"generated spectrum for {concrete}\n"
        return header + table
