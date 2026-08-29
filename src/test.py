import torch

L = 3
torch.manual_seed(42)

# Generate small random grids for links and the gauge phase
lx = (torch.rand(L, L) * 2 - 1) * torch.pi
alpha = (torch.rand(L, L) * 2 - 1) * torch.pi

# The exact core math from your apply_local_gauge_open function
lx_new = lx[:-1, :] + alpha[:-1, :] - alpha[1:, :]

print("Original x-links (radians):\n", lx[:-1, :])
print("\nNew x-links (after gauge transform):\n", lx_new)
print("\nAbsolute Difference (must be non-zero):\n", torch.abs(lx_new - lx[:-1, :]))