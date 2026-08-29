import torch
import torch.nn as nn

def _init_weights(out_c, in_c):
    return torch.randn(out_c, in_c, dtype=torch.cfloat) * (1.0 / in_c) ** 0.5

class ParallelTransport(nn.Module):
    """
    Computes exact gauge-equivariant parallel transport using matrix multiplication.
    Supports U(1) as 1x1 matrices, SU(2) as 2x2, and SU(3) as 3x3.
    Expects tensors of shape: [Batch, Channels, L, L, Nc, Nc]
    """
    @staticmethod
    def forward_transport(w, u, shift_dim):
        # Transport W from x+mu to x: U_mu(x) W(x+mu) U_mu^dagger(x)
        w_shifted = torch.roll(w, shifts=-1, dims=shift_dim)
        u_dagger = u.conj().mT
        return u @ w_shifted @ u_dagger

    @staticmethod
    def backward_transport(w, u, shift_dim):
        # Transport W from x-mu to x: U_mu^dagger(x-mu) W(x-mu) U_mu(x-mu)
        u_shifted = torch.roll(u, shifts=1, dims=shift_dim)
        w_shifted = torch.roll(w, shifts=1, dims=shift_dim)
        u_dagger = u_shifted.conj().mT
        return u_dagger @ w_shifted @ u_shifted


class LConvLinear(nn.Module):
    """
    Generalized L-Conv layer (Eq 16 from Holland et al. 2024).
    Linearly combines local features with their parallel-transported neighbors.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.w_center = nn.Parameter(_init_weights(out_channels, in_channels))
        self.w_fwd_x = nn.Parameter(_init_weights(out_channels, in_channels))
        self.w_bwd_x = nn.Parameter(_init_weights(out_channels, in_channels))
        self.w_fwd_y = nn.Parameter(_init_weights(out_channels, in_channels))
        self.w_bwd_y = nn.Parameter(_init_weights(out_channels, in_channels))

    def forward(self, w, u_x, u_y):
        # w: [B, C_in, L, L, Nc, Nc]
        # u_x, u_y: [B, 1, L, L, Nc, Nc]
        
        w_fwd_x = ParallelTransport.forward_transport(w, u_x, shift_dim=2)
        w_bwd_x = ParallelTransport.backward_transport(w, u_x, shift_dim=2)
        w_fwd_y = ParallelTransport.forward_transport(w, u_y, shift_dim=3)
        w_bwd_y = ParallelTransport.backward_transport(w, u_y, shift_dim=3)

        # Einsum handles the channel mixing while preserving the spatial and NcxNc dimensions
        out = torch.einsum('oi,bixy...->boxy...', self.w_center, w)
        out = out + torch.einsum('oi,bixy...->boxy...', self.w_fwd_x, w_fwd_x)
        out = out + torch.einsum('oi,bixy...->boxy...', self.w_bwd_x, w_bwd_x)
        out = out + torch.einsum('oi,bixy...->boxy...', self.w_fwd_y, w_fwd_y)
        out = out + torch.einsum('oi,bixy...->boxy...', self.w_bwd_y, w_bwd_y)
        return out


class MatrixTrNorm(nn.Module):
    """
    Trace Normalization adapted for matrix traces.
    Normalizes the feature maps by the spatial mean of the matrix trace magnitude.
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, w):
        # Compute the trace over the Nc x Nc dimensions
        tr_w = torch.diagonal(w, dim1=-2, dim2=-1).sum(-1)
        mag_sq = tr_w.real**2 + tr_w.imag**2
        mean_mag = torch.mean(mag_sq, dim=(2, 3), keepdim=True)
        scale = torch.rsqrt(mean_mag + self.eps)
        # Reshape scale to broadcast across the NcxNc dimensions
        return w * scale.unsqueeze(-1).unsqueeze(-1)


class PlaquetteMatrixLayer(nn.Module):
    """
    Computes 1x1 Wilson loops using exact non-commutative matrix multiplication.
    """
    def __init__(self):
        super().__init__()

    def forward(self, u_x, u_y):
        # P_xy = U_x(x) @ U_y(x+x) @ U_x^dagger(x+y) @ U_y^dagger(x)
        u_y_fwd_x = torch.roll(u_y, shifts=-1, dims=2)
        u_x_fwd_y = torch.roll(u_x, shifts=-1, dims=3)
        
        p_xy = u_x @ u_y_fwd_x @ u_x_fwd_y.conj().mT @ u_y.conj().mT
        return p_xy


class LgeConvNet(nn.Module):
    """
    SU(N)-ready Lattice-Gauge-Equivariant CNN architecture.
    """
    def __init__(self, in_channels, hidden_channels, n_layers, out_features, gauge_invariant=True):
        super().__init__()
        self.gauge_invariant = gauge_invariant
        self.plaquette_layer = PlaquetteMatrixLayer()
        
        chans = [in_channels + 1] + [hidden_channels] * n_layers
        self.blocks = nn.ModuleList()
        
        for i in range(n_layers):
            self.blocks.append(nn.ModuleDict({
                'conv': LConvLinear(chans[i], chans[i+1]),
                'norm': MatrixTrNorm()
            }))
            
        if self.gauge_invariant:
            self.readout = nn.Linear(chans[-1], out_features)

    def forward(self, f, u_x, u_y):
        # Format adapter: if inputs are [B, C, L, L] scalars, unsqueeze to [B, C, L, L, 1, 1] matrices
        if f.dim() == 4:
            f = f.unsqueeze(-1).unsqueeze(-1)
            u_x = u_x.unsqueeze(-1).unsqueeze(-1)
            u_y = u_y.unsqueeze(-1).unsqueeze(-1)

        # 1. Compute Matrix Plaquette
        p_xy = self.plaquette_layer(u_x, u_y)
        
        # 2. Append Plaquette to input features
        z = torch.cat([f, p_xy], dim=1)
        
        # 3. Hidden Blocks (no normalization - causes mode collapse)
        # Normalization removed: it destroys gauge-invariant structure
        # Skip connections would be better, but left for future work
        for block in self.blocks:
            z = block['conv'](z, u_x, u_y)
            # z = block['norm'](z)  # DISABLED: causes information loss
            
        # 4. Invariant Readout: Use REAL part of trace (gauge-invariant, Holland et al. 2024)
        # FIXED: |tr(z)|² loses information; use tr(z).real instead
        if self.gauge_invariant:
            tr_z = torch.diagonal(z, dim1=-2, dim2=-1).sum(-1)
            # Use real part only: gauge-invariant and preserves information
            # For U(1): tr(z) is complex; real part is invariant under global phase
            inv = tr_z.real.permute(0, 2, 3, 1)  # [B, L, L, C]
            out = self.readout(inv).permute(0, 3, 1, 2)  # [B, out_features, L, L]
            return out
            
        # If not invariant, return the raw [B, C, L, L, Nc, Nc] tensor field
        return z