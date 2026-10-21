"""Nyquist and Bode plots in the style of meannyquist.py.

Black and white only, TrueType outlines, series told apart by line style not hue. The same
rcParams and line conventions meannyquist.py uses for the thesis plots, applied to a single
spectrum: a Nyquist curve (Re against -Im) and a Bode pair (|Z| and phase against frequency).
"""

from __future__ import annotations

import io
import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Black and white only, on the conventions of meannyquist.py: TrueType outlines so the PDFs stay
# scalable and searchable in the thesis, and an inverted gray colormap for anything that ever shades.
plt.rcParams["image.cmap"] = "gray_r"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# matplotlib is not thread-safe: serialize figure creation/render/close so concurrent spectra'
# plots don't race on the global figure state.
_PLOT_LOCK = threading.Lock()


def _save_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return buf.getvalue()


def plot_nyquist(omega: np.ndarray, re: np.ndarray, im: np.ndarray, title: str = "") -> bytes:
    """Nyquist plot, Re(Z) against -Im(Z), as a PNG.

    The curve is drawn from high to low frequency (the conventional direction) so the arrow of
    time reads left to right along the arc."""
    with _PLOT_LOCK:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        order = np.argsort(omega)[::-1]
        ax.plot(re[order], -im[order], color="black", linestyle="-", linewidth=1.4)
        ax.set_xlabel("Re(Z) [$\\Omega$]")
        ax.set_ylabel("-Im(Z) [$\\Omega$]")
        if title:
            ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        return _save_png(fig)


def plot_bode(omega: np.ndarray, re: np.ndarray, im: np.ndarray, title: str = "") -> bytes:
    """Bode plot: |Z| (log-log) over phase (log-x, linear-y), as a PNG.

    Frequency is shown in Hz (omega / 2 pi) on the axis, the conventional Bode convention, while
    the data stays in rad/s internally."""
    with _PLOT_LOCK:
        f = omega / (2.0 * np.pi)
        mag = np.hypot(re, im)
        phase = np.degrees(np.arctan2(im, re))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.6, 5.2), sharex=True)
        ax1.loglog(f, mag, color="black", linestyle="-", linewidth=1.4)
        ax1.set_ylabel("|Z| [$\\Omega$]")
        ax1.grid(True, alpha=0.3, which="both")
        if title:
            ax1.set_title(title)

        ax2.semilogx(f, phase, color="black", linestyle="-", linewidth=1.4)
        ax2.set_xlabel("frequency [Hz]")
        ax2.set_ylabel("phase [deg]")
        ax2.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        return _save_png(fig)
