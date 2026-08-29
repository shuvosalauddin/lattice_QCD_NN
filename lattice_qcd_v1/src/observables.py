"""
Physical observables for U(1) lattice gauge theory: plaquettes / Wilson
loops, the gauge-covariant kinetic term of a charged scalar, plus a slot
for topological charge.
"""

import torch

from .lattice import wrap


def calc_plaquettes_open(lx, ly):
    """
    Elementary plaquette (1x1 Wilson loop) phase, open boundary:
        theta_x(n) + theta_y(n+x) - theta_x(n+y) - theta_y(n)

    Named _open because it uses open-boundary slicing; do not confuse with
    a periodic-BC plaquette calc (which would use torch.roll instead).
    """
    p = lx[:-1, :-1] + ly[1:, :-1] - lx[:-1, 1:] - ly[:-1, :-1]
    return wrap(p)


def kinetic_term_periodic(f, u_x, u_y):
    """
    Lattice-averaged gauge-covariant kinetic energy density of a charge-1
    scalar field f, periodic boundary conditions:

        (1/N) * sum_x  Re[ f(x)^* U_x(x) f(x+x^) + f(x)^* U_y(x) f(x+y^) ]

    Each term f(x)^* U_mu(x) f(x+mu) is the standard gauge-invariant
    "hopping" term used to couple a charged matter field to a U(1) gauge
    field on the lattice (the same structure that appears in lattice
    scalar QED / lattice Higgs actions). It is manifestly invariant under
    a local gauge transform: f(x) -> g(x) f(x) and
    U_mu(x) -> g(x) U_mu(x) g(x+mu)^*, so
        f(x)^* U_mu(x) f(x+mu)
          -> g(x)^* f(x)^* * g(x) U_mu(x) g(x+mu)^* * g(x+mu) f(x+mu)
           = f(x)^* U_mu(x) f(x+mu)   (all the g factors cancel).

    Used as a physically meaningful regression target for the
    equivariant-vs-plain-CNN comparison in experiments/equivariance_demo.py.

    f: (B, C, L, L) complex. u_x, u_y: (B, 1, L, L) complex, |u| = 1.
    Returns: (B,) real tensor, one scalar per batch element (summed/averaged
    over channels and lattice sites).
    """
    f_fwd_x = torch.roll(f, shifts=-1, dims=2)
    f_fwd_y = torch.roll(f, shifts=-1, dims=3)
    term_x = torch.conj(f) * u_x * f_fwd_x
    term_y = torch.conj(f) * u_y * f_fwd_y
    density = term_x.real + term_y.real
    return density.mean(dim=(1, 2, 3))


# TODO: topological_charge(lx, ly) -- sum of wrapped plaquette phases / 2*pi,
# once needed for a specific experiment.
