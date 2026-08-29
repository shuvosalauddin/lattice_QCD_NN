"""
Physical observables for U(1) lattice gauge theory: plaquettes / Wilson
loops, topological charge, and the Wilson gauge action.
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


def calc_topological_charge(lx, ly):
    """
    Computes the topological winding number Q of the 2D U(1) lattice.
    Uses periodic boundary conditions to ensure a well-defined, boundary-free 
    total winding number.
    
    Q = (1 / 2*pi) * Sum(wrapped_plaquettes)
    """
    p = calc_plaquettes_periodic(lx, ly)
    Q = torch.sum(p) / (2 * torch.pi)
    # The physical topological charge must be an integer
    return torch.round(Q)


def calc_wilson_action(plaquettes, beta=1.0):
    """
    Computes the standard U(1) Wilson gauge action.
    
    S = beta * Sum(1 - cos(P_munu))
    
    This acts as the fundamental physics-informed loss function when training 
    generative models or solving for field configurations.
    """
    return beta * torch.sum(1.0 - torch.cos(plaquettes))