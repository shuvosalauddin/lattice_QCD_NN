"""
Quantitative equivariance/invariance checks for every layer in src/models.py.
Run with: python tests/test_equivariance.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lattice import gauge_transform_periodic
from src.models import (
    GaugeEquivariantConv2D,
    GaugeBilinearConv2D,
    ModEquivariantActivation,
    GaugeInvariantReadout,
    LatticeGaugeCNN,
    PlainCNN,
)

torch.manual_seed(0)
L = 8
B, C_in, C_hidden = 2, 3, 6


def random_lattice():
    f = torch.randn(B, C_in, L, L, dtype=torch.cfloat)
    links_x = (torch.rand(L, L) * 2 - 1) * torch.pi
    links_y = (torch.rand(L, L) * 2 - 1) * torch.pi
    u_x = torch.polar(torch.ones(1, 1, L, L), links_x.unsqueeze(0).unsqueeze(0)).expand(B, -1, -1, -1)
    u_y = torch.polar(torch.ones(1, 1, L, L), links_y.unsqueeze(0).unsqueeze(0)).expand(B, -1, -1, -1)
    alpha = (torch.rand(L, L) * 2 - 1) * torch.pi
    alpha = alpha.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
    return f, u_x, u_y, alpha


def rel_err(a, b):
    return (torch.norm(a - b) / torch.norm(a).clamp_min(1e-12)).item()


def test_conv_charge1_equivariance():
    f, u_x, u_y, alpha = random_lattice()
    layer = GaugeEquivariantConv2D(C_in, C_hidden)
    g = torch.polar(torch.ones_like(alpha), alpha)

    out0 = layer(f, u_x, u_y)
    f_t, u_x_t, u_y_t = gauge_transform_periodic(f, u_x, u_y, alpha)
    out_t = layer(f_t, u_x_t, u_y_t)

    err = rel_err(out_t, g * out0)
    print(f"[conv]    charge-1 equivariance out(g.x) == g.out(x): rel err = {err:.2e}")
    assert err < 1e-5


def test_bilinear_charge2_equivariance():
    f, u_x, u_y, alpha = random_lattice()
    layer = GaugeBilinearConv2D(C_in, C_in, C_hidden)
    g = torch.polar(torch.ones_like(alpha), alpha)

    out0 = layer(f, f, u_x, u_y)
    f_t, u_x_t, u_y_t = gauge_transform_periodic(f, u_x, u_y, alpha)
    out_t = layer(f_t, f_t, u_x_t, u_y_t)

    # bilinear layer combines two charge-1 fields -> charge-2 output
    err = rel_err(out_t, (g ** 2) * out0)
    print(f"[bilinear] charge-2 equivariance out(g.x) == g^2.out(x): rel err = {err:.2e}")
    assert err < 1e-5


def test_activation_preserves_any_charge():
    f, u_x, u_y, alpha = random_lattice()
    act = ModEquivariantActivation(C_in)
    g = torch.polar(torch.ones_like(alpha), alpha)

    for charge, label in [(1, "charge-1"), (2, "charge-2")]:
        z = f if charge == 1 else f * f
        out0 = act(z)
        out_t = act((g ** charge) * z)
        err = rel_err(out_t, (g ** charge) * out0)
        print(f"[act]     {label} equivariance preserved: rel err = {err:.2e}")
        assert err < 1e-5


def test_full_model_invariance():
    f, u_x, u_y, alpha = random_lattice()
    model = LatticeGaugeCNN(C_in, C_hidden, n_layers=3, out_features=1)

    out0 = model(f, u_x, u_y)
    f_t, u_x_t, u_y_t = gauge_transform_periodic(f, u_x, u_y, alpha)
    out_t = model(f_t, u_x_t, u_y_t)

    err = rel_err(out_t, out0)
    print(f"[model]   end-to-end gauge INVARIANCE (readout unchanged): rel err = {err:.2e}")
    assert err < 1e-4


def test_plain_cnn_is_not_invariant():
    f, u_x, u_y, alpha = random_lattice()
    x = torch.cat([f.real, f.imag, u_x.real, u_x.imag, u_y.real, u_y.imag], dim=1)
    model = PlainCNN(in_channels=x.shape[1], hidden_channels=C_hidden, n_layers=3, out_features=1)

    out0 = model(x)
    f_t, u_x_t, u_y_t = gauge_transform_periodic(f, u_x, u_y, alpha)
    x_t = torch.cat([f_t.real, f_t.imag, u_x_t.real, u_x_t.imag, u_y_t.real, u_y_t.imag], dim=1)
    out_t = model(x_t)

    err = rel_err(out_t, out0)
    print(f"[control] plain CNN is NOT invariant under the same transform: rel err = {err:.2e}")
    assert err > 1e-2, "plain CNN should NOT be invariant -- if this fails, the control is broken"


if __name__ == "__main__":
    test_conv_charge1_equivariance()
    test_bilinear_charge2_equivariance()
    test_activation_preserves_any_charge()
    test_full_model_invariance()
    test_plain_cnn_is_not_invariant()
    print("\nAll checks passed.")
