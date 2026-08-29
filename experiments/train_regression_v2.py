import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models import LgeConvNet

L = 16
B = 32
EPOCHS = 500

# ==========================================
# 1. PHYSICS DATA GENERATOR (Langevin Dynamics)
# ==========================================
def generate_langevin_u1_batch(beta_tensor, batch_size, L, steps=150, dt=0.02):
    """
    Simulates true U(1) thermal fluctuations using Langevin Dynamics.
    """
    # COLD START (Vacuum)
    lx = torch.zeros(batch_size, L, L, requires_grad=True)
    ly = torch.zeros(batch_size, L, L, requires_grad=True)
    
    beta = beta_tensor.view(batch_size, 1, 1)
    
    for step in range(steps):
        if lx.grad is not None: lx.grad.zero_()
        if ly.grad is not None: ly.grad.zero_()
        
        p = lx + torch.roll(ly, shifts=-1, dims=1) - torch.roll(lx, shifts=-1, dims=2) - ly
        action = (beta * (1.0 - torch.cos(p))).sum()
        action.backward()
        
        with torch.no_grad():
            # Proper Langevin: x_new = x - dt*∇S + √(2dt)*ξ (uncorrelated noise, unit variance)
            noise_x = torch.randn_like(lx)
            noise_y = torch.randn_like(ly)
            
            lx.data = lx - dt * lx.grad + (2 * dt)**0.5 * noise_x
            ly.data = ly - dt * ly.grad + (2 * dt)**0.5 * noise_y

    with torch.no_grad():
        p_final = lx + torch.roll(ly, shifts=-1, dims=1) - torch.roll(lx, shifts=-1, dims=2) - ly
        avg_plaquettes = torch.cos(p_final).mean(dim=(1, 2))
        
        lx = torch.remainder(lx + torch.pi, 2 * torch.pi) - torch.pi
        ly = torch.remainder(ly + torch.pi, 2 * torch.pi) - torch.pi
        
        u_x = torch.polar(torch.ones_like(lx), lx).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        u_y = torch.polar(torch.ones_like(ly), ly).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        
        # Use LOCAL plaquette values as input features (gauge-invariant)
        # CRITICAL FIX: spatial structure prevents mode collapse
        # Local 1 - cos(P) encodes action density at each site
        p_local = p_final  # [B, L, L] local plaquette phases
        f_spatial = 1.0 - 0.5 * torch.cos(p_local)  # [B, L, L] gauge-invariant scalars
        f = f_spatial.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, L, L, 1, 1]
        
    return f, u_x, u_y, avg_plaquettes

# ==========================================
# 2. MODEL, OPTIMIZER & LOSS INITIALIZATION
# ==========================================
# INCREASED CAPACITY: Was 16 hidden channels, now 32 (double)
# 3→4 layers provides more expressive power
model = LgeConvNet(in_channels=1, hidden_channels=32, n_layers=4, out_features=1, gauge_invariant=True)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
criterion = nn.MSELoss()

print("======================================================")
print(f"Training LGE-CNN for Action Parameter Regression")
print(f"Lattice: {L}x{L} | Batch Size: {B} | Epochs: {EPOCHS}")
print("======================================================\n")

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def sample_beta_physical(batch_size):
    """
    Sample beta from a distribution that emphasizes the phase transition region.
    U(1) has a phase transition around beta_c ≈ 0.67-0.9 (smooth crossover).
    This distribution weights physically interesting regimes higher.
    """
    # 60% from [0.3, 1.5] (phase transition), 40% from [1.5, 10.0] (weak coupling)
    n_phase = int(batch_size * 0.6)
    n_weak = batch_size - n_phase
    
    beta_phase = torch.empty(n_phase).uniform_(0.3, 1.5)
    beta_weak = torch.empty(n_weak).uniform_(1.5, 10.0)
    beta = torch.cat([beta_phase, beta_weak]).reshape(batch_size, 1, 1, 1)
    
    return beta[torch.randperm(batch_size)]  # Shuffle to break correlation

for epoch in range(EPOCHS):
    start_time = time.time()
    optimizer.zero_grad()
    
    true_beta = sample_beta_physical(B)
    true_flat = true_beta.view(-1)
    
    f, u_x, u_y, avg_plaq = generate_langevin_u1_batch(true_flat, B, L, steps=150, dt=0.02)
    
    predicted_beta = model(f, u_x, u_y)
    pred_flat = predicted_beta.mean(dim=(2, 3)).view(-1)
    
    loss = criterion(pred_flat, true_flat)
    loss.backward()
    optimizer.step()
    
    epoch_time = time.time() - start_time
    
    # Evaluate MAE (Mean Absolute Error) for a more readable accuracy metric
    mae = torch.abs(pred_flat - true_flat).mean().item()
    
    if epoch % 10 == 0:
        with torch.no_grad():
            # Sort the batch to examine the hot, warm, and cold lattice predictions
            sorted_idx = torch.argsort(true_flat)
            idx_low = sorted_idx[0].item()       # Hottest lattice (lowest beta)
            idx_mid = sorted_idx[B // 2].item()  # Median lattice
            idx_high = sorted_idx[-1].item()     # Coldest lattice (highest beta)
            
            # Verify gauge invariance: check that predictions are stable under small gauge transforms
            if epoch % 100 == 0 and epoch > 0:
                from src.lattice import gauge_transform_periodic
                alpha_test = (torch.rand(1, L, L) * 0.1 - 0.05)  # Small random gauge
                f_test_g, u_x_test_g, u_y_test_g = gauge_transform_periodic(f[0:1], u_x[0:1], u_y[0:1], alpha_test)
                pred_gauge_transformed = model(f_test_g, u_x_test_g, u_y_test_g).mean().item()
                pred_original = model(f[0:1], u_x[0:1], u_y[0:1]).mean().item()
                gauge_error = abs(pred_gauge_transformed - pred_original) / (abs(pred_original) + 1e-8)
                print(f"    [GAUGE] Invariance check (small gauge): {gauge_error:.2e} (should be ~1e-6)")

            print(f"--- Epoch {epoch:03d}/{EPOCHS} [{epoch_time:.2f}s] ---")
            print(f"    MSE Loss : {loss.item():.4f}")
            print(f"    Batch MAE: {mae:.4f} (Average prediction error)")
            print(f"    [HOT]  True Beta: {true_flat[idx_low]:.2f}  | Pred: {pred_flat[idx_low]:.2f}  | Plaquette: {avg_plaq[idx_low]:.3f}")
            print(f"    [WARM] True Beta: {true_flat[idx_mid]:.2f}  | Pred: {pred_flat[idx_mid]:.2f}  | Plaquette: {avg_plaq[idx_mid]:.3f}")
            print(f"    [COLD] True Beta: {true_flat[idx_high]:.2f} | Pred: {pred_flat[idx_high]:.2f} | Plaquette: {avg_plaq[idx_high]:.3f}")
            print("-" * 54)

print("\nTraining Complete.")

# ==========================================
# 4. EVALUATION & PLOTTING
# ==========================================
print("Generating final test configurations and evaluating...")
model.eval()
test_batch_size = 100

true_test_beta = sample_beta_physical(test_batch_size)
true_test_flat = true_test_beta.view(-1)
f_test, u_x_test, u_y_test, _ = generate_langevin_u1_batch(true_test_flat, test_batch_size, L, steps=200, dt=0.02)

with torch.no_grad():
    predicted_test_beta = model(f_test, u_x_test, u_y_test).mean(dim=(2, 3)).view(-1)

plt.figure(figsize=(8, 6))
plt.scatter(true_test_flat.numpy(), predicted_test_beta.numpy(), alpha=0.7, color='blue', edgecolors='k')
plt.plot([1, 10], [1, 10], color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')

plt.title(f"LGE-CNN Action Parameter Regression (L={L}x{L})")
plt.xlabel("True Beta")
plt.ylabel("Predicted Beta")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plot_path = os.path.join(ROOT_DIR, "output_plots", "regression_scatter.jpg")
plt.savefig(plot_path, dpi=200, bbox_inches='tight')
print(f"Saved evaluation scatter plot to: {plot_path}")
true_beta = sample_beta_physical(1000)

f, ux, uy, plaq = generate_langevin_u1_batch(
    true_beta.view(-1),
    1000,
    L,
    steps=200,
    dt=0.02
)

print("Beta range:", true_beta.min().item(), true_beta.max().item())
print("Plaquette range:", plaq.min().item(), plaq.max().item())

plt.scatter(
    true_beta.view(-1).numpy(),
    plaq.numpy()
)

plt.xlabel("Beta")
plt.ylabel("Average Plaquette")
plt.grid(True)
plt.show()