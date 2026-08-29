"""
Lattice geometries, local gauge transforms, and link-variable operations
for U(1) lattice gauge theory (open and periodic boundary conditions).
"""

import torch


def wrap(theta):
    """Wrap an angle tensor to (-pi, pi]."""
    return torch.remainder(theta + torch.pi, 2 * torch.pi) - torch.pi


def apply_local_gauge_open(lx, ly, alpha):
    """
    Open-boundary local U(1) gauge transform of link angles:
        theta_mu(n) -> theta_mu(n) + alpha(n) - alpha(n+mu)

    Boundary links that never enter an interior plaquette are left as-is,
    matching observables.calc_plaquettes_open. Written functionally
    (torch.cat, not in-place slice assignment) so it composes safely with
    autograd if you wrap these tensors in requires_grad=True.
    """
    lx_new = torch.cat([lx[:-1, :] + alpha[:-1, :] - alpha[1:, :], lx[-1:, :]], dim=0)
    ly_new = torch.cat([ly[:, :-1] + alpha[:, :-1] - alpha[:, 1:], ly[:, -1:]], dim=1)
    return wrap(lx_new), wrap(ly_new)


def gauge_transform_periodic(f, u_x, u_y, alpha, charge=1):
    """
    Local U(1) gauge transform (periodic BC) of a charge-`charge` field + links:
        f(n)    -> g(n)^charge * f(n)
        U_mu(n) -> g(n) * U_mu(n) * g(n+mu)^dagger   (links always transport charge 1)
    Defaults to charge=1 (matter field) to match models.py. Used by the
    gauge-equivariant CNN in models.py, which assumes periodic boundaries
    via torch.roll — do not mix with the open-BC helpers above.
    """
    g = torch.polar(torch.ones_like(alpha), alpha)
    g_fwd_x = torch.roll(g, shifts=-1, dims=-2)
    g_fwd_y = torch.roll(g, shifts=-1, dims=-1)
    return g**charge * f, g * u_x * torch.conj(g_fwd_x), g * u_y * torch.conj(g_fwd_y)