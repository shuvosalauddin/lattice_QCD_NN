"""
Gauge-equivariant CNN ("L-CNN" style, cf. Favoni, Ipp, Muller & Schuh 2020)
for U(1) lattice gauge theory, periodic boundary conditions.
"""

import torch
import torch.nn as nn


def _init(out_c, in_c):
    return torch.randn(out_c, in_c, dtype=torch.cfloat) * (1.0 / in_c) ** 0.5


class GaugeEquivariantConv2D(nn.Module):
    """
    One U(1) lattice gauge-equivariant conv layer (periodic BC), using the
    full 4-neighbour stencil. Forward neighbours are transported with the
    site's own link U_mu(n); backward neighbours with U_mu(n-mu)^dagger.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.w_center = nn.Parameter(_init(out_channels, in_channels))
        self.w_fwd_x = nn.Parameter(_init(out_channels, in_channels))
        self.w_bwd_x = nn.Parameter(_init(out_channels, in_channels))
        self.w_fwd_y = nn.Parameter(_init(out_channels, in_channels))
        self.w_bwd_y = nn.Parameter(_init(out_channels, in_channels))

    def forward(self, f, u_x, u_y):
        # f: (B, C_in, L, L) complex. u_x, u_y: (B, 1, L, L) complex, |u| = 1.
        f_fwd_x = u_x * torch.roll(f, shifts=-1, dims=2)
        f_bwd_x = torch.conj(torch.roll(u_x, shifts=1, dims=2)) * torch.roll(f, shifts=1, dims=2)
        f_fwd_y = u_y * torch.roll(f, shifts=-1, dims=3)
        f_bwd_y = torch.conj(torch.roll(u_y, shifts=1, dims=3)) * torch.roll(f, shifts=1, dims=3)

        out = torch.einsum('oi,bixy->boxy', self.w_center, f)
        out = out + torch.einsum('oi,bixy->boxy', self.w_fwd_x, f_fwd_x)
        out = out + torch.einsum('oi,bixy->boxy', self.w_bwd_x, f_bwd_x)
        out = out + torch.einsum('oi,bixy->boxy', self.w_fwd_y, f_fwd_y)
        out = out + torch.einsum('oi,bixy->boxy', self.w_bwd_y, f_bwd_y)
        return out


class ModEquivariantActivation(nn.Module):
    """
    Gauge-equivariant nonlinearity (lattice analogue of complex modReLU):
    acts on the magnitude only and keeps the phase, so it commutes with
    any per-site U(1) phase multiplication.
    """

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, z):
        mag = z.abs()
        scale = torch.relu(mag + self.bias.view(1, -1, 1, 1)) / (mag + self.eps)
        return z * scale


class GaugeInvariantReadout(nn.Module):
    """
    |z|^2 per channel is invariant under *local* U(1) gauge transforms
    (since |g(n)| = 1 pointwise), so a real-valued linear layer can mix
    those magnitudes freely into any output (regression targets, class
    logits, an energy density map, etc) without reintroducing gauge
    dependence.
    """

    def __init__(self, in_channels, out_features):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_features)

    def forward(self, z):
        inv = (z.real ** 2 + z.imag ** 2).permute(0, 2, 3, 1)   # (B, L, L, C), invariant
        return self.linear(inv).permute(0, 3, 1, 2)              # (B, out_features, L, L)


class LatticeGaugeCNN(nn.Module):
    """Stack of gauge-equivariant conv+activation layers, invariant head."""

    def __init__(self, in_channels, hidden_channels, n_layers, out_features):
        super().__init__()
        chans = [in_channels] + [hidden_channels] * n_layers
        self.layers = nn.ModuleList(
            [GaugeEquivariantConv2D(chans[i], chans[i + 1]) for i in range(n_layers)]
        )
        self.acts = nn.ModuleList(
            [ModEquivariantActivation(chans[i + 1]) for i in range(n_layers)]
        )
        self.readout = GaugeInvariantReadout(chans[-1], out_features)

    def forward(self, f, u_x, u_y):
        z = f
        for layer, act in zip(self.layers, self.acts):
            z = act(layer(z, u_x, u_y))
        return self.readout(z)