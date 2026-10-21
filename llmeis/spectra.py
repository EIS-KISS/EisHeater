"""EISF spectrum parsing, log resampling, and dataset iteration (tar archives and/or csv files).

The spectrum files are the EISF csv format used across the kiss toolchain:

    EISF, 1.1.0
    "l{...}-p{...}r{...}-..."
    None
    /path/to/source.eis3
    omega, real, im
    2.513953e-02, 6.978860e+03, -9.020950e+03
    ...

The second line is the ECM string the spectrum was generated from. It is parsed and kept on the
Spectrum, but the agent is never shown it: the whole point is to recover the circuit from the data.
"""

from __future__ import annotations

import os
import tarfile
from dataclasses import dataclass

import numpy as np


@dataclass
class Spectrum:
    """One EIS spectrum: name, the measured data, and the (hidden) header model."""

    name: str
    omega: np.ndarray  # angular frequency, rad/s, ascending
    re: np.ndarray  # Re(Z), ohm
    im: np.ndarray  # Im(Z), ohm
    header_model: str = ""  # ECM string from the EISF header, not shown to the agent
    description: str = ""  # free text line under the model, if any

    @property
    def n_points(self) -> int:
        return int(self.omega.size)

    @property
    def omega_range(self) -> tuple[float, float]:
        return float(self.omega.min()), float(self.omega.max())


def parse_model(lines: list[str]) -> str:
    """The ECM string from an EISF header: the line after the EISF marker, quotes stripped."""
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


def parse_spectra(lines: list[str]):
    """(omega, re, im) from the data block after the "omega, real, im" header, omega ascending."""
    hdr = None
    for i, line in enumerate(lines):
        if line.startswith("omega"):
            hdr = i
            break
    if hdr is None:
        return None
    rows = []
    for line in lines[hdr + 1:]:
        line = line.strip()
        if not line:
            continue
        fields = line.split(",")
        if len(fields) < 3:
            continue
        try:
            rows.append((float(fields[0]), float(fields[1]), float(fields[2])))
        except ValueError:
            break
    if not rows:
        return None
    arr = np.array(rows, dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return arr[:, 0], arr[:, 1], arr[:, 2]


def parse_eisf(name: str, text: str) -> Spectrum | None:
    """Parse one EISF csv into a Spectrum, or None if it holds no usable data block."""
    lines = text.splitlines()
    if not lines:
        return None
    data = parse_spectra(lines)
    if data is None:
        return None
    omega, re, im = data
    # The description is the non-empty line between the model and the data header, if any.
    description = ""
    for i, line in enumerate(lines):
        if line.startswith("EISF") and i + 3 < len(lines) and not lines[i + 3].startswith("omega"):
            candidate = lines[i + 3].strip()
            if candidate and candidate != "None":
                description = candidate
            break
    return Spectrum(
        name=name,
        omega=omega,
        re=re,
        im=im,
        header_model=parse_model(lines),
        description=description,
    )


def resample_log(spec: Spectrum, n: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a spectrum onto n points log-spaced in omega across its full range.

    Linear interpolation of Re and Im onto the new grid; the endpoints are the spectrum's own
    min/max omega, so the resampled curve spans exactly the measured range."""
    w0, w1 = spec.omega_range
    if w0 <= 0 or w1 <= w0:
        raise ValueError(f"spectrum {spec.name} has a degenerate omega range {w0}..{w1}")
    grid = np.logspace(np.log10(w0), np.log10(w1), n)
    re = np.interp(grid, spec.omega, spec.re)
    im = np.interp(grid, spec.omega, spec.im)
    return grid, re, im


def format_data(omega: np.ndarray, re: np.ndarray, im: np.ndarray) -> str:
    """A spectrum as a compact text table: one "omega, re, im" row per point."""
    rows = [f"{w:.6e}, {r:.6e}, {i:.6e}" for w, r, i in zip(omega, re, im)]
    return "omega [rad/s], Re(Z) [ohm], Im(Z) [ohm]\n" + "\n".join(rows)


def format_data_polar(omega: np.ndarray, re: np.ndarray, im: np.ndarray) -> str:
    """A spectrum as a text table with omega, Re, Im, |Z|, and phase (degrees).

    The |Z| and phase columns are the polar form of the impedance, the same quantities the Bode
    plot shows, so the model can read magnitudes and phase directly from the numbers."""
    mag = np.hypot(re, im)
    phase = np.degrees(np.arctan2(im, re))
    rows = [
        f"{w:.6e}, {r:.6e}, {i:.6e}, {m:.6e}, {p:.4f}"
        for w, r, i, m, p in zip(omega, re, im, mag, phase)
    ]
    return "omega [rad/s], Re(Z) [ohm], Im(Z) [ohm], |Z| [ohm], phase [deg]\n" + "\n".join(rows)


def iter_spectra(paths: list[str]):
    """Yield (name, text) for every spectrum in the given dataset paths, in the order given.

    Each path is either a dataset tar (every .csv member, in tar order) or a single EISF csv file.
    Non-csv tar members (an optional meta.json, directories, ...) are skipped."""
    for path in paths:
        if path.endswith(".csv"):
            yield from _iter_csv(path)
        else:
            yield from _iter_tar(path)


def _iter_csv(path: str):
    """Yield the single (name, text) of one EISF csv file; the name is the file's basename."""
    with open(path, "r", errors="replace") as f:
        yield os.path.basename(path), f.read()


def _iter_tar(tarpath: str):
    """Yield (name, text) for every .csv member of a dataset tar, in tar order."""
    with tarfile.open(tarpath) as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.endswith(".csv"):
                continue
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            yield member.name, fobj.read().decode(errors="replace")


def write_csv(spec: Spectrum, path: str) -> None:
    """Write a spectrum back out in EISF csv format (used to hand the original to eisfit)."""
    lines = ["EISF, 1.1.0", f'"{spec.header_model}"', "None", spec.description, "omega, real, im"]
    lines += [f"{w:.6e}, {r:.6e}, {i:.6e}" for w, r, i in zip(spec.omega, spec.re, spec.im)]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
