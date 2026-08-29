"""
U(1) lattice gauge theory: invariance/equivariance demos.

Renamed from 'U(1) 2D.py'. Logic unchanged from the original script;
lattice math now lives in src/lattice.py and src/observables.py, and the
CNN lives in src/models.py -- this file just wires them together and
produces plots.
"""

import os
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make `src` importable when running this file directly (python experiments/u1_2d_simulation.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lattice import apply_local_gauge_open, gauge_transform_periodic
from src.observables import calc_plaquettes_open
from src.models import GaugeEquivariantConv2D, LatticeGaugeCNN

torch.manual_seed(42)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

L = 10


# ==========================================
# EXPERIMENT 1: WILSON LOOP GAUGE INVARIANCE
# ==========================================
links_x = (torch.rand(L, L) * 2 - 1) * torch.pi
links_y = (torch.rand(L, L) * 2 - 1) * torch.pi
local_gauge = (torch.rand(L, L) * 2 - 1) * torch.pi

original_plaquettes = calc_plaquettes_open(links_x, links_y)
links_x_new, links_y_new = apply_local_gauge_open(links_x, links_y, local_gauge)
transformed_plaquettes = calc_plaquettes_open(links_x_new, links_y_new)

err1 = torch.norm(original_plaquettes - transformed_plaquettes).item()
print(f"[Exp 1] Wilson loop invariance error: {err1:.2e}  (float32 noise floor)")

fig1, axs1 = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle("U(1) Gauge Invariance: Links Change, Wilson Loops Don't", fontsize=14)
X, Y = np.meshgrid(np.arange(L), np.arange(L))


def plot_lattice(ax, lx, ly, plaquettes, title):
    im = ax.pcolormesh(np.arange(L), np.arange(L), plaquettes.numpy(),
                        cmap='viridis', vmin=-np.pi, vmax=np.pi, shading='flat', alpha=0.8)
    ax.set_xticks(np.arange(L))
    ax.set_yticks(np.arange(L))
    ax.grid(color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.quiver(X + 0.5, Y, np.cos(lx.numpy()), np.sin(lx.numpy()),
              color='white', pivot='mid', scale=25, headwidth=4, headlength=4, zorder=3)
    ax.quiver(X, Y + 0.5, np.cos(ly.numpy()), np.sin(ly.numpy()),
              color='red', pivot='mid', scale=25, headwidth=4, headlength=4, zorder=3)
    ax.set_title(title, pad=10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='both', length=0)
    ax.set_aspect('equal')
    ax.set_xlim(0, L - 1)
    ax.set_ylim(0, L - 1)
    return im


im1 = plot_lattice(axs1[0], links_x, links_y, original_plaquettes, "Original Gauge Field")
plot_lattice(axs1[1], links_x_new, links_y_new, transformed_plaquettes, "After Local Gauge Transform")
fig1.colorbar(im1, ax=axs1, orientation='vertical', fraction=0.02, pad=0.04).set_label('Wilson Loop Phase')
fig1.savefig(os.path.join(OUTPUT_DIR, "wilson_loop_invariance.jpg"), dpi=200, bbox_inches="tight")
plt.close(fig1)


# ==========================================
# EXPERIMENT 2: GAUGE-EQUIVARIANT CNN
# ==========================================
B, C_in, C_hidden, N_LAYERS, C_out = 2, 3, 8, 4, 1
model = LatticeGaugeCNN(C_in, C_hidden, N_LAYERS, C_out)

f = torch.randn(B, C_in, L, L, dtype=torch.cfloat)
u_x = torch.polar(torch.ones(1, 1, L, L), links_x.unsqueeze(0).unsqueeze(0)).expand(B, -1, -1, -1)
u_y = torch.polar(torch.ones(1, 1, L, L), links_y.unsqueeze(0).unsqueeze(0)).expand(B, -1, -1, -1)

out_original = model(f, u_x, u_y)

alpha = local_gauge.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
f_t, u_x_t, u_y_t = gauge_transform_periodic(f, u_x, u_y, alpha)
out_transformed = model(f_t, u_x_t, u_y_t)

# The readout is gauge-INVARIANT: output should be identical, not just phase-rotated.
err2 = torch.norm(out_original - out_transformed).item() / torch.norm(out_original).item()
print(f"[Exp 2] {N_LAYERS}-layer gauge-CNN readout invariance (relative error): {err2:.2e}")

# Sanity check on a single layer's pre-readout equivariance too:
one_layer = GaugeEquivariantConv2D(C_in, C_hidden)
z0 = one_layer(f, u_x, u_y)
z1 = one_layer(f_t, u_x_t, u_y_t)
g = torch.polar(torch.ones_like(alpha), alpha)
err3 = torch.norm(z1 - g * z0).item() / torch.norm(z0).item()
print(f"[Exp 2] single-layer field equivariance |out(g.x) - g.out(x)| (relative): {err3:.2e}")

# Visualize one channel of the invariant output map, before vs. after the
# gauge transform of the *inputs* -- these should look identical.
fig2, axs2 = plt.subplots(1, 2, figsize=(11, 5))
fig2.suptitle("Gauge-Invariant Readout: Identical Before/After a Gauge Transform of the Inputs", fontsize=12)
vmax = max(out_original[0, 0].abs().max().item(), out_transformed[0, 0].abs().max().item())
im_a = axs2[0].imshow(out_original[0, 0].detach().numpy(), cmap='magma', vmin=0, vmax=vmax)
axs2[0].set_title("Readout(original inputs)")
axs2[0].axis('off')
axs2[1].imshow(out_transformed[0, 0].detach().numpy(), cmap='magma', vmin=0, vmax=vmax)
axs2[1].set_title("Readout(gauge-transformed inputs)")
axs2[1].axis('off')
fig2.colorbar(im_a, ax=axs2, orientation='vertical', fraction=0.02, pad=0.04)
fig2.savefig(os.path.join(OUTPUT_DIR, "gauge_invariant_readout.jpg"), dpi=200, bbox_inches="tight")
plt.close(fig2)

print("Saved plots to", OUTPUT_DIR)
