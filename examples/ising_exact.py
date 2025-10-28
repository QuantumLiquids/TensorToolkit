#!/usr/bin/env python3

"""
Generate analytic free energy CSV for 2D Ising (square lattice).
Serve for z2_ising_trg.cpp.

Usage:
    python ising_exact.py --betas "0.2,0.4,0.7,1.0" -o ising_exact.csv

Options:
    --betas STR    Comma-separated beta values, e.g. '0.2,0.4,0.7,1.0'
    -o STR         Output CSV path
    --J FLOAT      Coupling J (default: 1.0)
    --kB FLOAT     Boltzmann constant k_B (default: 1.0)
    --rtol FLOAT   Relative/absolute tolerance for integration (default: 1e-12)
"""

import argparse
import csv
import math
from typing import Callable

try:
    from scipy import integrate as _integrate  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import mpmath as _mp  # type: ignore
    _HAS_MPMATH = True
except Exception:
    _HAS_MPMATH = False


def free_energy_density(beta: float, J: float = 1.0, kB: float = 1.0, tol: float = 1e-12) -> float:
    """
    Exact free energy per site for the isotropic 2D Ising model (Onsager, isotropic J).
    Uses the authoritative double-integral form:

      -β f = ln 2 + (1/(8π^2)) ∫_0^{2π} ∫_0^{2π}
               ln[ cosh(2K) cosh(2K) - sinh(2K)(cos θx + cos θy) ] dθx dθy

    where K = β J / k_B.
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    K = beta * J / kB
    c = math.cosh(2.0 * K)
    s = math.sinh(2.0 * K)

    def g(theta_x: float, theta_y: float) -> float:
        val = c * c - s * (math.cos(theta_x) + math.cos(theta_y))
        # Numerical guard
        if val <= 0.0:
            val = 1e-300
        return math.log(val)

    if _HAS_SCIPY:
        val, _err = _integrate.dblquad(
            lambda ty, tx: g(tx, ty), 0.0, 2.0 * math.pi,
            lambda _tx: 0.0, lambda _tx: 2.0 * math.pi,
            epsabs=tol, epsrel=tol
        )
    elif _HAS_MPMATH:
        try:
            _mp.mp.dps = max(30, int(-math.log10(tol)) + 10)
        except Exception:
            _mp.mp.dps = 50
        c_mp = _mp.cosh(2 * K)
        s_mp = _mp.sinh(2 * K)
        def g_mp(tx, ty):
            v = c_mp * c_mp - s_mp * (_mp.cos(tx) + _mp.cos(ty))
            if v <= 0:
                v = _mp.mpf('1e-300')
            return _mp.log(v)
        val_mp = _mp.quad(lambda tx: _mp.quad(lambda ty: g_mp(tx, ty), [0, 2 * _mp.pi]), [0, 2 * _mp.pi])
        val = float(val_mp)
    else:
        raise SystemExit(
            "Missing integration backend: install SciPy (preferred) or mpmath.\n"
            "  pip install scipy\n"
            "or: pip install mpmath"
        )

    bracket = math.log(2.0) + val / (8.0 * math.pi * math.pi)
    f = - (kB / beta) * bracket
    return f


def parse_betas(betas_arg: str):
    parts = [p.strip() for p in betas_arg.split(',') if p.strip()]
    return [float(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="Generate analytic free energy CSV for 2D Ising (square lattice).")
    parser.add_argument("--betas", type=str, required=True, help="Comma-separated beta values, e.g. '0.2,0.4,0.7,1.0'")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--J", type=float, default=1.0, help="Coupling J (default: 1.0)")
    parser.add_argument("--kB", type=float, default=1.0, help="Boltzmann constant k_B (default: 1.0)")
    parser.add_argument("--rtol", type=float, default=1e-12, help="Relative/absolute tolerance for integration")
    args = parser.parse_args()

    betas = parse_betas(args.betas)
    rows = [(beta, free_energy_density(beta, J=args.J, kB=args.kB, tol=args.rtol)) for beta in betas]

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beta", "f_exact"])
        for beta, fval in rows:
            w.writerow([f"{beta:.16g}", f"{fval:.16g}"])


if __name__ == "__main__":
    main()


