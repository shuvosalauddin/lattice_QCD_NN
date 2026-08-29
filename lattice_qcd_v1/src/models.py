"""
Gauge-equivariant CNN ("L-CNN" style, cf. Favoni, Ipp, Muller & Schuh 2020;
Holland, Ipp, Muller & Wenger 2024) for U(1) lattice gauge theory, periodic
boundary conditions.

CHARGE BOOKKEEPING (read this before adding new layers)
---------------------------------------------------------
Every complex field in this file carries an integer "charge" q under a
local U(1) gauge transform g(x): a charge-q field transforms as

    f(x)  ->  g(x)^q * f(x)

A matter field (input to the network) has charge 1. Parallel-transporting
a charge-q field along a link multiplies it by U^q, so charge-1 transport
uses the bare link; a charge-0 (already-invariant) field would need no
link factor at all.

Two operations are safe to mix charges with, one is NOT:
  - Complex-LINEAR combination (conv layers, activations that scale the
    magnitude) of same-charge inputs preserves that charge -- g(x) factors
    out of the sum. This is what GaugeEquivariantConv2D does.
  - Complex MULTIPLICATION of a charge-q1 and a charge-q2 field yields a
    charge-(q1+q2) field. This is what GaugeBilinearConv2D does, and it is
    exactly the "grow bigger Wilson loops" mechanism from the L-CNN papers.
  - Complex ADDITION of DIFFERENT-charge fields is NOT gauge-covariant:
    g(x)^q1 * a + g(x)^q2 * b != g(x)^k * (a + b) for any single k unless
    q1 == q2. Never add raw complex features of different charge together.

The one operation that is charge-agnostic and safe to combine anything
through is the invariant readout: |z|^2 is invariant for ANY integer
charge (|g(x)^q z| = |z| since |g(x)| = 1), so features of different
charge can be concatenated freely AFTER taking |z|^2, then linearly mixed
by an ordinary real-valued nn.Linear. This mirrors the "trace layer -> CNN"
pattern in the original L-CNN architecture (Fig. 6 of Favoni et al.).
"""

import torch
import torch.nn as nn


def _init(out_c, in_c, n_stencil=1):
    """
    Variance-preserving init. n_stencil is the number of INDEPENDENTLY
    initialized weight tensors that get summed together at forward time
    (e.g. GaugeEquivariantConv2D sums 5 of these: center + 4 neighbours).
    Since the terms are summed and are independent at init, their
    variances add -- so each term's own variance must be divided by
    n_stencil to keep the total output variance matched to the input's,
    or activation magnitude grows roughly sqrt(n_stencil) per layer and
    compounds into a blow-up after a few stacked layers.
    """
    fan_in = in_c * n_stencil
    return torch.randn(out_c, in_c, dtype=torch.cfloat) * (1.0 / fan_in) ** 0.5


def _init_bilinear(out_c, in_c_a, in_c_b, n_stencil):
    fan_in = in_c_a * in_c_b * n_stencil
    return torch.randn(out_c, in_c_a, in_c_b, n_stencil, dtype=torch.cfloat) * (1.0 / fan_in) ** 0.5


class GaugeEquivariantConv2D(nn.Module):
    """
    One U(1) lattice gauge-equivariant conv layer (periodic BC), using the
    full 4-neighbour stencil. Forward neighbours are transported with the
    site's own link U_mu(n); backward neighbours with U_mu(n-mu)^dagger.

    Charge: takes a charge-1 input, returns a charge-1 output.
    """

    _N_STENCIL = 5  # center + 4 neighbours, summed together at forward time

    def __init__(self, in_channels, out_channels):
        super().__init__()
        n = self._N_STENCIL
        self.w_center = nn.Parameter(_init(out_channels, in_channels, n))
        self.w_fwd_x = nn.Parameter(_init(out_channels, in_channels, n))
        self.w_bwd_x = nn.Parameter(_init(out_channels, in_channels, n))
        self.w_fwd_y = nn.Parameter(_init(out_channels, in_channels, n))
        self.w_bwd_y = nn.Parameter(_init(out_channels, in_channels, n))

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


class GaugeBilinearConv2D(nn.Module):
    """
    Bilinear convolution layer (cf. L-Bilin / L-CB in Favoni et al. 2020
    and Eq. 18 of Holland, Ipp, Muller & Wenger 2024): multiplies a local
    charge-1 field value by a *parallel-transported* neighbouring value of
    a (possibly different) charge-1 field, at the same site.

    This is the mechanism the plain GaugeEquivariantConv2D lacks: that
    layer only ever ADDS transported neighbours together, so stacking it
    N times only ever grows a real-linear combination of single-step
    transports -- it can never represent, say, a genuine two-link product
    (a 1x2 rectangle-type object). Multiplying two transported quantities
    at the same site is what actually traces out a bigger loop, which is
    the point emphasised in the L-CNN papers: bilinear layers let the
    network form Wilson loops of arbitrary shape from local building
    blocks, whereas a purely additive equivariant conv cannot.

    Charge: takes two charge-1 inputs, returns a charge-2 output. Do not
    add this layer's output to a charge-1 feature map directly (see the
    module docstring) -- keep it as a separate branch until the invariant
    readout.
    """

    _STENCIL = ('self', 'fwd_x', 'bwd_x', 'fwd_y', 'bwd_y')

    def __init__(self, in_channels_a, in_channels_b, out_channels):
        super().__init__()
        self.weight = nn.Parameter(
            _init_bilinear(out_channels, in_channels_a, in_channels_b, len(self._STENCIL))
        )

    @staticmethod
    def _transported_neighbours(w, u_x, u_y):
        """Same transport rule as GaugeEquivariantConv2D, returned as a list
        so it can be paired one-to-one with the bilinear weight's stencil axis."""
        w_fwd_x = u_x * torch.roll(w, shifts=-1, dims=2)
        w_bwd_x = torch.conj(torch.roll(u_x, shifts=1, dims=2)) * torch.roll(w, shifts=1, dims=2)
        w_fwd_y = u_y * torch.roll(w, shifts=-1, dims=3)
        w_bwd_y = torch.conj(torch.roll(u_y, shifts=1, dims=3)) * torch.roll(w, shifts=1, dims=3)
        return [w, w_fwd_x, w_bwd_x, w_fwd_y, w_bwd_y]

    def forward(self, w_a, w_b, u_x, u_y):
        # w_a: (B, Ca, L, L) complex, used untransported (local factor).
        # w_b: (B, Cb, L, L) complex, transported to each of the 5 stencil
        # positions before being multiplied into w_a.
        neighbours_b = self._transported_neighbours(w_b, u_x, u_y)
        out = 0.0
        for k, w_b_transported in enumerate(neighbours_b):
            out = out + torch.einsum(
                'oab,naxy,nbxy->noxy', self.weight[..., k], w_a, w_b_transported
            )
        return out


class ModEquivariantActivation(nn.Module):
    """
    Gauge-equivariant nonlinearity (lattice analogue of complex modReLU):
    acts on the magnitude only and keeps the phase, so it commutes with
    any per-site U(1) phase multiplication -- regardless of the field's
    charge, since it never inspects the phase at all.
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
    |z|^2 per channel is invariant under *local* U(1) gauge transforms for
    ANY integer charge q (since |g(x)^q| = 1 pointwise), so a real-valued
    linear layer can mix magnitudes from channels of different charge
    freely into any output (regression targets, class logits, an energy
    density map, etc) without reintroducing gauge dependence. This is the
    one place in the network where charge bookkeeping stops mattering.
    """

    def __init__(self, in_channels, out_features):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_features)

    def forward(self, z):
        inv = (z.real ** 2 + z.imag ** 2).permute(0, 2, 3, 1)   # (B, L, L, C), invariant
        return self.linear(inv).permute(0, 3, 1, 2)              # (B, out_features, L, L)


class LatticeGaugeCNN(nn.Module):
    """
    Two-branch gauge-equivariant CNN:
      - a charge-1 branch: a stack of GaugeEquivariantConv2D + activation,
        exactly as before (linear transports only);
      - a charge-2 branch: a single GaugeBilinearConv2D applied to the
        charge-1 branch's final features, giving the network access to
        genuine two-link products (bigger loops) that the charge-1 branch
        alone cannot represent.
    Both branches' invariant magnitudes are concatenated at the readout,
    which is always safe regardless of charge (see module docstring).
    """

    def __init__(self, in_channels, hidden_channels, n_layers, out_features):
        super().__init__()
        chans = [in_channels] + [hidden_channels] * n_layers
        self.layers = nn.ModuleList(
            [GaugeEquivariantConv2D(chans[i], chans[i + 1]) for i in range(n_layers)]
        )
        self.acts = nn.ModuleList(
            [ModEquivariantActivation(chans[i + 1]) for i in range(n_layers)]
        )
        self.bilin = GaugeBilinearConv2D(hidden_channels, hidden_channels, hidden_channels)
        self.bilin_act = ModEquivariantActivation(hidden_channels)
        self.readout = GaugeInvariantReadout(hidden_channels * 2, out_features)

    def forward(self, f, u_x, u_y):
        z1 = f
        for layer, act in zip(self.layers, self.acts):
            z1 = act(layer(z1, u_x, u_y))
        z2 = self.bilin_act(self.bilin(z1, z1, u_x, u_y))
        combined = torch.cat([z1, z2], dim=1)  # different charges (1 and 2) -- safe here
                                                # ONLY because the readout takes |.|^2 per
                                                # channel before any further mixing.
        return self.readout(combined)


class PlainCNN(nn.Module):
    """
    Deliberately NOT gauge-equivariant: a standard real-valued CNN that
    receives the real/imaginary parts of the field and the links as plain
    image channels with no built-in transport structure. Used as a
    control to demonstrate that an ordinary CNN's predictions are NOT
    invariant under a gauge transform of the input, unlike LatticeGaugeCNN.
    """

    def __init__(self, in_channels, hidden_channels, n_layers, out_features):
        super().__init__()
        layers = []
        c_in = in_channels
        for _ in range(n_layers):
            layers.append(nn.Conv2d(c_in, hidden_channels, kernel_size=3, padding=1, padding_mode='circular'))
            layers.append(nn.ReLU())
            c_in = hidden_channels
        self.conv = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden_channels, out_features, kernel_size=1)

    def forward(self, x):
        return self.head(self.conv(x))
