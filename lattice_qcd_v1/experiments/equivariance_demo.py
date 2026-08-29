"""
Phase 1 experiment: reproduce the headline claim of Favoni, Ipp, Muller &
Schuh (2020) in miniature -- a plain CNN's predictions are sensitive to
gauge transformations of a *physically identical* configuration, while a
gauge-equivariant L-CNN's predictions are not, by construction.

Both networks are trained to regress the same physical target: the
lattice-averaged gauge-covariant kinetic term of a charge-1 scalar field
(see src/observables.kinetic_term_periodic), a genuinely gauge-invariant
quantity computed directly from the field and link configuration.

After training, we evaluate both models on a held-out configuration and
on a gauge-transformed copy of the *same* configuration (same physics,
different arbitrary gauge choice). A model that has learned the invariance
should give (near) identical predictions on both; a model that hasn't
built the symmetry in cannot guarantee this and, empirically, does not.
"""

import os
import sys

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lattice import gauge_transform_periodic
from src.observables import kinetic_term_periodic
from src.models import LatticeGaugeCNN, PlainCNN

torch.manual_seed(0)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "outputs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

L = 10
C_IN = 2
HIDDEN = 8
N_LAYERS = 3
N_TRAIN = 800
N_TEST = 100
EPOCHS = 300


def make_dataset(n_samples):
    f = torch.randn(n_samples, C_IN, L, L, dtype=torch.cfloat)
    links_x = (torch.rand(n_samples, L, L) * 2 - 1) * torch.pi
    links_y = (torch.rand(n_samples, L, L) * 2 - 1) * torch.pi
    u_x = torch.polar(torch.ones_like(links_x), links_x).unsqueeze(1)
    u_y = torch.polar(torch.ones_like(links_y), links_y).unsqueeze(1)
    target = kinetic_term_periodic(f, u_x, u_y)
    return f, u_x, u_y, target


def plain_cnn_input(f, u_x, u_y):
    return torch.cat([f.real, f.imag, u_x.real, u_x.imag, u_y.real, u_y.imag], dim=1)


def train(model, is_equivariant, f, u_x, u_y, target, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        if is_equivariant:
            out = model(f, u_x, u_y).mean(dim=(1, 2, 3))
        else:
            out = model(plain_cnn_input(f, u_x, u_y)).mean(dim=(1, 2, 3))
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def evaluate(model, is_equivariant, f, u_x, u_y):
    with torch.no_grad():
        if is_equivariant:
            return model(f, u_x, u_y).mean(dim=(1, 2, 3))
        return model(plain_cnn_input(f, u_x, u_y)).mean(dim=(1, 2, 3))


# ---------------- train both models on the same data ----------------
f_train, ux_train, uy_train, y_train = make_dataset(N_TRAIN)
f_test, ux_test, uy_test, y_test = make_dataset(N_TEST)

equivariant_model = LatticeGaugeCNN(C_IN, HIDDEN, N_LAYERS, out_features=1)
plain_model = PlainCNN(in_channels=2 * (C_IN + 1 + 1), hidden_channels=HIDDEN, n_layers=N_LAYERS, out_features=1)

print("Training equivariant L-CNN...")
losses_eq = train(equivariant_model, True, f_train, ux_train, uy_train, y_train, EPOCHS)
print("Training plain CNN...")
losses_plain = train(plain_model, False, f_train, ux_train, uy_train, y_train, EPOCHS)

pred_eq_test = evaluate(equivariant_model, True, f_test, ux_test, uy_test)
pred_plain_test = evaluate(plain_model, False, f_test, ux_test, uy_test)
mse_eq = ((pred_eq_test - y_test) ** 2).mean().item()
mse_plain = ((pred_plain_test - y_test) ** 2).mean().item()
print(f"[Regression] test MSE -- L-CNN: {mse_eq:.4e}   plain CNN: {mse_plain:.4e}")


# ---------------- the actual point: gauge-invariance test ----------------
alpha = (torch.rand(N_TEST, L, L) * 2 - 1) * torch.pi
alpha = alpha.unsqueeze(1)  # (N_TEST, 1, L, L), real -- gauge_transform_periodic builds g = exp(i*alpha) itself
f_test_t, ux_test_t, uy_test_t = gauge_transform_periodic(f_test, ux_test, uy_test, alpha)

pred_eq_orig = evaluate(equivariant_model, True, f_test, ux_test, uy_test)
pred_eq_transformed = evaluate(equivariant_model, True, f_test_t, ux_test_t, uy_test_t)
pred_plain_orig = evaluate(plain_model, False, f_test, ux_test, uy_test)
pred_plain_transformed = evaluate(plain_model, False, f_test_t, ux_test_t, uy_test_t)

eq_change = (pred_eq_transformed - pred_eq_orig).abs()
plain_change = (pred_plain_transformed - pred_plain_orig).abs()

print(f"\n[Gauge invariance check] same physical configs, different arbitrary gauge:")
print(f"  L-CNN   |pred(g.x) - pred(x)| : mean={eq_change.mean():.2e}  max={eq_change.max():.2e}")
print(f"  plain CNN |pred(g.x) - pred(x)|: mean={plain_change.mean():.2e}  max={plain_change.max():.2e}")
print(f"  ratio (plain / L-CNN)         : {(plain_change.mean() / eq_change.mean().clamp_min(1e-12)):.1f}x")


# ---------------- plots ----------------
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

axs[0].plot(losses_eq, label="L-CNN (equivariant)")
axs[0].plot(losses_plain, label="Plain CNN")
axs[0].set_xlabel("epoch")
axs[0].set_ylabel("training MSE loss")
axs[0].set_yscale("log")
axs[0].set_title("Regression training (both fit the same target)")
axs[0].legend()

axs[1].scatter(y_test.detach(), pred_eq_test.detach(), s=12, alpha=0.6, label="L-CNN")
axs[1].scatter(y_test.detach(), pred_plain_test.detach(), s=12, alpha=0.6, label="Plain CNN")
lims = [y_test.min().item(), y_test.max().item()]
axs[1].plot(lims, lims, 'k--', linewidth=1)
axs[1].set_xlabel("true kinetic term")
axs[1].set_ylabel("predicted")
axs[1].set_title("Test-set regression accuracy")
axs[1].legend()

# L-CNN's changes are ~1e-8 (float32 noise floor) vs plain CNN's ~1e-1 --
# an 8-orders-of-magnitude gap that a linear-scale histogram would render
# as an invisible bar, so use log-scale bins and annotate the L-CNN value
# directly instead of pretending it's a comparable-scale histogram.
plain_vals = plain_change.detach().numpy()
bins = torch.logspace(-9, 0, 25).numpy()
axs[2].hist(plain_vals, bins=bins, alpha=0.7, color="tab:orange", label="Plain CNN")
axs[2].axvline(eq_change.mean().item(), color="tab:blue", linewidth=2,
               label=f"L-CNN (mean = {eq_change.mean().item():.1e}, float32 noise floor)")
axs[2].set_xscale("log")
axs[2].set_xlabel("|pred(gauge-transformed) - pred(original)|  (log scale)")
axs[2].set_ylabel("count")
axs[2].set_title("Sensitivity to an arbitrary gauge choice\n(should be ~0 for a gauge-invariant quantity)")
axs[2].legend(fontsize=8)

fig.suptitle("Equivariant L-CNN vs. plain CNN: same regression task, only one respects gauge symmetry", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "equivariance_demo.jpg"), dpi=200, bbox_inches="tight")
plt.close(fig)

print("\nSaved plot to", os.path.join(OUTPUT_DIR, "equivariance_demo.jpg"))
