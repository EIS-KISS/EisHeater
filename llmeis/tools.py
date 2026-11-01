"""The agent's tools and the eisfit interaction they share with the pipeline.

Two tools are exposed to the model:

  * fit_spectrum(model_str, [omega_min, omega_max]): fit the proposed circuit to the ORIGINAL
                                full-resolution spectrum (or a frequency slice of it) with the
                                proposed circuit and hand the fit report back.
  * simulate_spectrum(model_str): generate a spectrum from the circuit with eisgeneratorpy on
                                the 20-point resampled grid and hand the values back.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field

import numpy as np
from pydantic_ai import RunContext

from .eisgen import eisgenerator
from .spectra import Spectrum, write_csv

# eisfit is a CPU fit with random restarts; give it room but bound it so one bad model can't stall
# the whole run.
_EISFIT_TIMEOUT = 180.0


@dataclass
class SpectrumDeps:
    csv_path: str  # path to the original EISF csv, what eisfit reads
    omega: np.ndarray  # original full-resolution omega, for slicing fit windows
    re: np.ndarray
    im: np.ndarray
    res_omega: np.ndarray  # 20 log-spaced points, the simulate grid
    res_re: np.ndarray
    res_im: np.ndarray
    n_original: int = 0
    omega_range: tuple[float, float] = (0.0, 0.0)
    extra: dict = field(default_factory=dict)


def run_eisfit(csv_path: str, model_str: str, timeout: float = _EISFIT_TIMEOUT) -> str:
    """Run eisfit on a spectrum with a model string, returning its full stdout+stderr text.
    range (r{1~1000}) bounds the fit to that interval, and one that is not given starts in the
    middle of that element's default range."""
    cmd = ["eisfit", "-f", csv_path, "-m", model_str]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"eisfit timed out after {timeout:.0f}s fitting {model_str!r}"
    out = proc.stdout or ""
    err = proc.stderr or ""
    return (out + ("\n" + err if err else "")).strip()


def _fit_csv_path(deps: SpectrumDeps, omega_min: float | None, omega_max: float | None) -> tuple[str, str]:
    """The csv to fit: the full spectrum, or a temp csv sliced to [omega_min, omega_max].

    Returns (csv_path, note); note is empty for the full spectrum.
    if omega_min is None and omega_max is None:
        return deps.csv_path, ""
    mask = np.ones(deps.omega.size, dtype=bool)
    if omega_min is not None:
        mask &= deps.omega >= omega_min
    if omega_max is not None:
        mask &= deps.omega <= omega_max
    if int(mask.sum()) < 2:
        raise ValueError(f"omega window [{omega_min}, {omega_max}] selects fewer than 2 points")
    w, r, i = deps.omega[mask], deps.re[mask], deps.im[mask]
    slice_path = os.path.join(os.path.dirname(deps.csv_path), "spectrum_slice.csv")
    write_csv(Spectrum(name="slice", omega=w, re=r, im=i), slice_path)
    note = f"Fitting a slice: {int(mask.sum())} points, omega {w.min():.4e} to {w.max():.4e} rad/s"
    return slice_path, note


# A parameter-table row: index, name, start, fitted. The fitted column may be "nan".
_PARAM_RE = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+(-?[\d.]+(?:[eE][+-]?\d+)?)\s+(-?[\d.]+(?:[eE][+-]?\d+)?|nan)"
)
# A fit-quality row: an indented "key   value" pair, value being the trailing token(s).
_FITQUAL_RE = re.compile(r"^\s{2,}(\S(?:.*\S)?)\s{2,}(\S.*?)\s*$")
    try:
        return float(tok)
    except ValueError:
        return None


def parse_eisfit_output(text: str) -> dict:
    """Parse eisfit's report into structured fields.

    Returns a dict with:
      parameters   - {name: fitted value} for every parameter row
      fitted_model - the "Fitted model:" string, or ""
      fit_quality  - {metric: value-string} from the "Fit quality" block
      verdict      - the "Verdict:" line, or ""
      raw          - the full report text
    Missing sections yield empty values rather than raising, so a failed fit still parses."""
    parameters: dict[str, float] = {}
    fit_quality: dict[str, str] = {}
    fitted_model = ""
    verdict = ""
    in_quality = False

    for line in text.splitlines():
        if line.startswith("Fitted model:"):
            fitted_model = line.split(":", 1)[1].strip()
            in_quality = False
            continue
        if line.startswith("Verdict:"):
            verdict = line.split(":", 1)[1].strip()
            in_quality = False
            continue
        if line.strip() == "Fit quality":
            in_quality = True
            continue
        if in_quality:
            if not line.strip():
                continue
            if not line.startswith(" "):
                in_quality = False
                continue
            m = _FITQUAL_RE.match(line)
            if m:
                fit_quality[m.group(1).strip()] = m.group(2).strip()
            else:
                # Long keys leave only a single space before the value; take the trailing token.
                parts = line.strip().rsplit(None, 1)
                if len(parts) == 2:
                    fit_quality[parts[0]] = parts[1]
            continue
        pm = _PARAM_RE.match(line)
        if pm:
            name = pm.group(2)
            fitted = _to_float(pm.group(4))
            if fitted is not None:
                parameters[name] = fitted

    return {
        "parameters": parameters,
        "fitted_model": fitted_model,
        "fit_quality": fit_quality,
        "verdict": verdict,
        "raw": text,
    }


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
            csv_path, note = _fit_csv_path(deps, omega_min, omega_max)
        except ValueError as e:
            return f"could not fit: {e}"
        report = run_eisfit(csv_path, model_str)
        if note:
            report = f"{note}\n{report}"
        return report

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
