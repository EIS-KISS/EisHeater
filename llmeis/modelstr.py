"""KISS encoding: the model-string format, its documentation for the LLM, and validation.

The model strings this project uses are the KISS encoding of the eisgenerator toolchain (see
phdthesis/chapters/background/eis_boukamp_coding.tex). A circuit is a string of element symbols
combined by an implicit parallel operator (adjacency) and an explicit series operator ("-"), with
parentheses for grouping. Parameters are embedded in braces after each element.

eisgenerator does not implement the Gerischer element "g", so model strings containing it are
rejected here even though the encoding itself allows it.
"""

from __future__ import annotations

# The element reference, in the form the LLM needs to read it. Kept in sync with the eisgenerator
# model page; "g" is listed only to say it is unsupported.
KISS_ENCODING_DOC = """\
Model strings use the KISS encoding. A circuit is built from element symbols combined by two
operators:

  * series: write a "-" between two elements, e.g. "r-c" is a resistor in series with a capacitor.
  * parallel: write two elements directly adjacent, e.g. "rc" is a resistor in parallel with a
    capacitor. The implicit parallel operator has HIGHER precedence than "-".
  * parentheses group sub-circuits, e.g. "r(r-c)" is a resistor in parallel with a series
    resistor-capacitor branch.

Elements (case insensitive) and their parameters:

  r  resistor        R  resistance [ohm]                 Z = R
  c  capacitor       C  capacitance [F]                  Z = 1 / (j omega C)
  l  inductor        L  inductance [H]                   Z = j omega L
  w  Warburg         A  coefficient [ohm s^-1/2]         Z = A (1 - j) / sqrt(omega)
  p  constant phase  Q  admittance coefficient [S s^a],  Z = 1 / (Q (j omega)^alpha)
                     alpha  exponent [-]  (0 < alpha <= 1)
  t  transmission    R [ohm/m], Q [S s^a/m], alpha [-],  Z = Z0 tanh(gamma l)
     line, closed    l  length [m]
  o  transmission    R [ohm/m], Q [S s^a/m], alpha [-],  Z = Z0 / tanh(gamma l)
     line, open      l  length [m]

The Gerischer element "g" is NOT supported and must not appear in a model string.

Parameters are written in braces after the element. An element with several parameters lists them
comma separated:

  r{100}                 a 100 ohm resistor
  p{1e-5, 0.8}           a CPE with Q = 1e-5 S s^a and alpha = 0.8
  r{100}-p{1e-5, 0.8}    100 ohm in series with that CPE

A parameter may also be a range, "a~b" (linear) or "a~bL" (logarithmic in base 10), with an
optional step count after ":", e.g. r{1~1e3L:10}. Ranges are only meaningful for parameter sweeps;
for a concrete circuit every parameter must be a single number.

Frequency convention: omega is the angular frequency in rad/s. The e^(+j omega t) convention is
used, so a capacitor has negative imaginary impedance, Im(Z) = -1/(omega C). A depressed
semicircle (real battery behaviour) needs a CPE "p", not an ideal "c".

Worked example: "l{1e-6}-r{50}-p{1e-4, 0.9}(r{1000}-w{100}-c{1e-2})" is a small inductor in
series with a 50 ohm resistor, in series with a parallel block containing a CPE and a series
resistor-Warburg-capacitor branch."""


def class_of(model: str) -> str:
    """Intern an ECM string to its class by dropping every bracketed parameter list.

    Same rule meannyquist.py / the C++ loaders use: "r{10000}p{6.5e-07, 0.5}" becomes "rp"."""
    out = []
    depth = 0
    for c in model:
        if c in "{[<":
            depth += 1
        if depth == 0:
            out.append(c)
        if c in "}]>":
            depth -= 1
    return "".join(out)


def has_parameters(model: str) -> bool:
    """True if the model string carries at least one braced parameter list."""
    return "{" in model and "}" in model


def validate_model_str(model: str, eisgen) -> tuple[bool, str]:
    """Check a model string is usable: parseable by eisgenerator, no "g", has parameters.

    Returns (ok, message). The message is empty on success, otherwise a human-readable reason the
    LLM can act on."""
    m = model.strip().strip('"')
    if not m:
        return False, "the model string is empty"
    # "g" is a standalone element symbol; reject it wherever it appears outside braces.
    depth = 0
    for c in m:
        if c in "{[(":
            depth += 1
        elif c in "}])":
            depth -= 1
        elif depth == 0 and c in "gG":
            return False, 'the model string contains the Gerischer element "g", which eisgenerator does not support'
    if not has_parameters(m):
        return False, (
            "the model string has no parameters; every element needs a braced value, "
            'e.g. "r{50}-p{1e-4, 0.8}"'
        )
    try:
        eisgen.Model(m)
    except Exception as e:  # ParseError and friends
        return False, f"eisgenerator could not parse the model string: {e}"
    return True, ""
