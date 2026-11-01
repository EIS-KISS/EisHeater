rom __future__ import annotations

import math

import numpy as np

import eisfit

__all__ = ["as_dict", "eisfit", "fit_to_dict", "report", "run_fit"]

eisfit.setLogLevel(eisfit.LogLevel.ERROR)

DEFAULT_RESTARTS = 16
DEFAULT_MAX_EVALUATIONS = 2000


def run_fit(
    omega: np.ndarray,
    re: np.ndarray,
    im: np.ndarray,
    model_str: str,
    *,
    restarts: int = DEFAULT_RESTARTS,
    max_evaluations: int = DEFAULT_MAX_EVALUATIONS,
) -> eisfit.FitResult:
    values = np.asarray(re, dtype=np.float64) + 1j * np.asarray(im, dtype=np.float64)
    return eisfit.fit(
        values,
        model_str,
        np.asarray(omega, dtype=np.float64),
        restarts=restarts,
        maxEvaluations=max_evaluations,
    )


def report(result: eisfit.FitResult, note: str = "") -> str:
    text = result.summary()
    return f"{note}\n{text}" if note else text


def _finite(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def as_dict(result: eisfit.FitResult) -> dict:
    names = list(result.parameterNames)
    parameters = {name: _finite(value) for name, value in zip(names, result.parameters)}
    sigma = {name: _finite(value) for name, value in zip(names, result.uncertainties)}
    noise_floor = float(result.noiseFloor)
    return {
        "parameters": parameters,
        "uncertainties": sigma,
        "fitted_model": result.fittedModel,
        "fit_quality": {
            "chi^2 per point": f"{result.chiSquared:.4e}",
            "reduced chi^2": f"{result.reducedChiSquared:.4e}",
            "rms relative residual": f"{result.rmsRelativeError * 100:.3f} %",
            "largest point residual": f"{result.maxRelativeError * 100:.3f} %",
            "noise floor of data": (
                f"{noise_floor * 100:.3f} %" if noise_floor > 0 else "not estimable"
            ),
            "residual autocorrelation": f"{result.residualCorrelation:.2f}",
            "condition number": f"{result.conditionNumber:.4e}",
            "optimiser": (
                f"{'converged' if result.converged else 'out of budget'}, "
                f"{int(result.evaluations)} simulations, "
                f"start {int(result.bestStart)} of {int(result.starts)}"
            ),
        },
        "verdict": result.verdict.label,
        "success": bool(result.success),
        "at_bound": [bool(b) for b in result.atBound],
        "warnings": list(result.warnings()),
    }


def fit_to_dict(
    omega: np.ndarray,
    re: np.ndarray,
    im: np.ndarray,
    model_str: str,
) -> dict:
    try:
        result = run_fit(omega, re, im, model_str)
    except (eisfit.ModelParseError, eisfit.FitError, ValueError) as e:
        return {"parameters": {}, "fitted_model": "", "fit_quality": {}, "verdict": "", "error": str(e)}
    return as_dict(result)
