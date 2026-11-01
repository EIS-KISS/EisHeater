from __future__ import annotations

import io
import os
import tarfile
from dataclasses import dataclass
import kisstype
import numpy as np
import kisstype


@dataclass
class Spectrum:

    name: str
    omega: np.ndarray  # angular frequency, rad/s, ascending
    re: np.ndarray  # Re(Z), ohm
    im: np.ndarray  # Im(Z), ohm
    header_model: str = ""  # ECM string from the EISF header, not shown to the agent
    description: str = ""

    @property
    def n_points(self) -> int:
        return int(self.omega.size)

    @property
    def omega_range(self) -> tuple[float, float]:
        return float(self.omega.min()), float(self.omega.max())

    @property
    def impedance(self) -> np.ndarray:
        """The measured points as complex impedances, the order the fitter takes them in."""
        return self.re + 1j * self.im


def parse_model(lines: list[str]) -> str:
    model = ""
    for i, line in enumerate(lines):
        if line.startswith("EISF"):
            if i + 1 < len(lines):
                model = lines[i + 1]
            break
    else:
        if len(lines) > 1:
            model = lines[1]
    return model.strip().strip('"')


def _from_kisstype(name: str, text: str) -> Spectrum | None:
    loaded = kisstype.Spectra.loadFromStream(io.StringIO(text))
    points = loaded.data
    if not points:
        return None
    omega = np.fromiter((point.omega for point in points), dtype=np.float64, count=len(points))
    re, im = kisstype.eisToValarrays(points)
    re = np.asarray(re, dtype=np.float64)
    im = np.asarray(im, dtype=np.float64)
    # kisstype hands the points back in file order; everything downstream (the resampling's
    # interpolation above all) wants them sorted by frequency.
    order = np.argsort(omega, kind="stable")
    return Spectrum(
        name=name,
        omega=omega[order],
        re=re[order],
        im=im[order],
        header_model=loaded.model,
        description=loaded.header,
    )


def parse_eisf(name: str, text: str) -> Spectrum | None:
    return _from_kisstype(name, text)


def resample_log(spec: Spectrum, n: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w0, w1 = spec.omega_range
    if w0 <= 0 or w1 <= w0:
        raise ValueError(f"spectrum {spec.name} has a degenerate omega range {w0}..{w1}")
    grid = np.logspace(np.log10(w0), np.log10(w1), n)
    re = np.interp(grid, spec.omega, spec.re)
    im = np.interp(grid, spec.omega, spec.im)
    return grid, re, im


def format_data(omega: np.ndarray, re: np.ndarray, im: np.ndarray) -> str:
    rows = [f"{w:.6e}, {r:.6e}, {i:.6e}" for w, r, i in zip(omega, re, im)]
    return "omega [rad/s], Re(Z) [ohm], Im(Z) [ohm]\n" + "\n".join(rows)


def format_data_polar(omega: np.ndarray, re: np.ndarray, im: np.ndarray) -> str:
    mag = np.hypot(re, im)
    phase = np.degrees(np.arctan2(im, re))
    rows = [
        f"{w:.6e}, {r:.6e}, {i:.6e}, {m:.6e}, {p:.4f}"
        for w, r, i, m, p in zip(omega, re, im, mag, phase)
    ]
    return "omega [rad/s], Re(Z) [ohm], Im(Z) [ohm], |Z| [ohm], phase [deg]\n" + "\n".join(rows)


def iter_spectra(paths: list[str]):
    for path in paths:
        if path.endswith(".csv"):
            yield from _iter_csv(path)
        else:
            yield from _iter_tar(path)


def _iter_csv(path: str):
    with open(path, "r", errors="replace") as f:
        yield os.path.basename(path), f.read()


def _iter_tar(tarpath: str):
    with tarfile.open(tarpath) as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            yield member.name, fobj.read().decode(errors="replace")
