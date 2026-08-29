"""
Physical observables for U(1) lattice gauge theory: plaquettes / Wilson
loops, plus a slot for topological charge.
"""

import torch

from .lattice import wrap


def calc_plaquettes_open(lx, ly):
    """
    Elementary plaquette (1x1 Wilson loop) phase, open boundary:
        theta_x(n) + theta_y(n+x) - theta_x(n+y) - theta_y(n)

    Named _open because it uses open-boundary slicing; do not confuse with
    calc_plaquettes_periodic below.
    """
    p = lx[:-1, :-1] + ly[1:, :-1] - lx[:-1, 1:] - ly[:-1, :-1]
    return wrap(p)


def calc_plaquettes_periodic(lx, ly):
    """
    Elementary plaquette phase, periodic boundary:
        theta_x(n) + theta_y(n+x) - theta_x(n+y) - theta_y(n)

    Uses torch.roll so every site has a well-defined plaquette (no dropped
    boundary rows/cols, unlike calc_plaquettes_open). Do not mix with the
    open-BC links/gauge transforms in lattice.py.
    """
    p = lx + torch.roll(ly, shifts=-1, dims=0) - torch.roll(lx, shifts=-1, dims=1) - ly
    return wrap(p)


# TODO: topological_charge(lx, ly) -- sum of wrapped plaquette phases / 2*pi,
# once needed for a specific experiment. Use calc_plaquettes_periodic for
# a well-defined (boundary-free) total winding number.