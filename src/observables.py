"""
Physical observables for U(1) lattice gauge theory: plaquettes / Wilson
loops, plus a slot for topological charge.
"""

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


# TODO: topological_charge(lx, ly) -- sum of wrapped plaquette phases / 2*pi,
# once needed for a specific experiment.
