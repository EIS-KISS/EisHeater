from __future__ import annotations

import io
import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["image.cmap"] = "gray_r"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

_PLOT_LOCK = threading.Lock()


def _save_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return buf.getvalue()


def plot_nyquist(omega: np.ndarray, re: np.ndarray, im: np.ndarray, title: str = "") -> bytes:
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
