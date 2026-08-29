import os
import sys
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
def generate_langevin_u1_batch(beta_tensor, batch_size, L, steps=200, dt=0.01):
    """
    Simulates true U(1) thermal fluctuations using Langevin Dynamics.
    beta_tensor: Expected shape [B]
    """
    lx = (torch.rand(batch_size, L, L) * 2 - 1) * torch.pi
    ly = (torch.rand(batch_size, L, L) * 2 - 1) * torch.pi
    
    lx.requires_grad_(True)
    ly.requires_grad_(True)
    
    beta = beta_tensor.view(batch_size, 1, 1)
    
    for step in range(steps):
        if lx.grad is not None: lx.grad.zero_()
        if ly.grad is not None: ly.grad.zero_()
        
        p = lx + torch.roll(ly, shifts=-1, dims=1) - torch.roll(lx, shifts=-1, dims=2) - ly
        action = (beta * (1.0 - torch.cos(p))).sum()
        action.backward()
        
        with torch.no_grad():
            noise_x = torch.randn_like(lx) * (2 * dt)**0.5
            noise_y = torch.randn_like(ly) * (2 * dt)**0.5
            
            lx.data.add_(-dt * lx.grad + noise_x)
            ly.data.add_(-dt * ly.grad + noise_y)

    with torch.no_grad():
        lx = torch.remainder(lx + torch.pi, 2 * torch.pi) - torch.pi
        ly = torch.remainder(ly + torch.pi, 2 * torch.pi) - torch.pi
        
        u_x = torch.polar(torch.ones_like(lx), lx).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        u_y = torch.polar(torch.ones_like(ly), ly).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        f = torch.ones(batch_size, 1, L, L, 1, 1, dtype=torch.cfloat)
        
    return f, u_x, u_y

# ==========================================
# 2. MODEL, OPTIMIZER & LOSS INITIALIZATION
# ==========================================
model = LgeConvNet(in_channels=1, hidden_channels=16, n_layers=3, out_features=1, gauge_invariant=True)
# Lower learning rate to prevent bouncing out of the physical minimum
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

print("Training LGE-CNN for Action Parameter Regression (Predicting Beta)...")

# ==========================================
# 3. TRAINING LOOP
# ==========================================
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    true_beta = torch.empty(B, 1, 1, 1).uniform_(1.0, 10.0)
    true_flat = true_beta.view(-1)
    
    # Generate data (requires autograd under the hood)
    f, u_x, u_y = generate_langevin_u1_batch(true_flat, B, L, steps=200, dt=0.01)
    
    predicted_beta = model(f, u_x, u_y)
    pred_flat = predicted_beta.mean(dim=(2, 3)).view(-1)
    
    loss = criterion(pred_flat, true_flat)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0:
        with torch.no_grad():
            sample_pred = pred_flat[0].item()
            sample_true = true_flat[0].item()
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                  f"Sample - True: {sample_true:.2f}, Predicted: {sample_pred:.2f}")

print("Training Complete.")

# ==========================================
# 4. EVALUATION & PLOTTING
# ==========================================
print("Evaluating model on a fresh test set...")
model.eval()
test_batch_size = 100

# 1. Generate data OUTSIDE the no_grad block because Langevin needs gradients
true_test_beta = torch.empty(test_batch_size, 1, 1, 1).uniform_(1.0, 10.0)
true_test_flat = true_test_beta.view(-1)
f_test, u_x_test, u_y_test = generate_langevin_u1_batch(true_test_flat, test_batch_size, L, steps=200, dt=0.01)

# 2. Predict INSIDE the no_grad block
with torch.no_grad():
    predicted_test_beta = model(f_test, u_x_test, u_y_test).mean(dim=(2, 3)).view(-1)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(true_test_flat.numpy(), predicted_test_beta.numpy(), alpha=0.7, color='blue', edgecolors='k')

# Draw the y = x reference line
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