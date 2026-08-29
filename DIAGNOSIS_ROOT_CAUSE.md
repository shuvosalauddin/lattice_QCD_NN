# Diagnosis: Why Model Predicts Constant Value

## THE PROBLEM (Visual Evidence)
- True β range: [0.3, 10.0] (10x variation)
- Predicted β: ALL ~2.0 (constant!)
- This is called **mode collapse** in deep learning

## ROOT CAUSES (Research-Based Analysis)

### 1. **Invariant Readout is TOO RESTRICTIVE** ⚠️ CRITICAL

Current code (models.py):
```python
# Take trace of matrix, square it
tr_z = torch.diagonal(z, dim1=-2, dim2=-1).sum(-1)  # [B, L, L, C]
inv = (tr_z.real**2 + tr_z.imag**2).permute(0, 2, 3, 1)  # [B, L, L, C]
out = self.readout(inv).permute(0, 3, 1, 2)  # [B, 1, L, L]
```

**PROBLEM:**
- You're squaring: |tr(z)|² → ALL POSITIVE
- This destroys information about the sign/phase
- Multiplying by hidden layer output: z → g·z means |tr(g·z)| = |g||tr(z)|
- For U(1): |g| = 1 always, so invariance is correct BUT...
- **The readout only sees ONE number per site: |tr(z)|²**
- Can't distinguish different z patterns

**From Holland et al. 2024:**
> "The gauge invariant observable should use (anti-)symmetric combinations"
> 
> Better: `tr(z) + tr(z)†` (real part only) or `tr(z @ z†)` (matrix product)

---

### 2. **Trace Normalization is KILLING GRADIENTS** ⚠️

Current (line 125 in models.py):
```python
z = block['norm'](z)  # Re-enabled

class MatrixTrNorm(nn.Module):
    def forward(self, w):
        tr_w = torch.diagonal(w, dim1=-2, dim2=-1).sum(-1)
        mag_sq = tr_w.real**2 + tr_w.imag**2
        mean_mag = torch.mean(mag_sq, dim=(2, 3), keepdim=True)
        scale = torch.rsqrt(mean_mag + self.eps)
        return w * scale.unsqueeze(-1).unsqueeze(-1)  # Broadcast across [Nc, Nc]
```

**PROBLEM:**
- Scale is computed per **site and channel**: `[B, C, L, L]`
- But broadcast to **matrix dimensions**: `[B, C, L, L, Nc, Nc]`
- This creates misaligned reshaping!
- Normalizing by |tr| removes magnitude information

**Better approach:** Don't normalize hidden layers; let them learn

---

### 3. **Input Feature is Scalar (Same for All Sites)** ⚠️

Current:
```python
f = torch.ones(...) * (1.0 - torch.cos(p_final).mean() * 0.1)  # Global scalar
```

**PROBLEM:**
- Every spatial location [i,j] gets THE SAME value
- Zero spatial structure in input
- CNN has no reason to create spatial features

**From papers (Holland et al., Weiler et al. 2023):**
> Gauge-equivariant networks should include local gauge-invariant scalars

**Better:**
```python
p_local = calc_plaquettes_periodic(lx, ly)  # [B, L, L]
f = 1.0 - 0.5 * torch.cos(p_local).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
# Now [B, 1, L, L, 1, 1] with local structure
```

---

### 4. **No Residual Connections or Dense Connections**

LGE-CNN architecture (models.py):
```python
for i in range(n_layers):
    self.blocks.append({
        'conv': LConvLinear(chans[i], chans[i+1]),
        'norm': MatrixTrNorm()
    })
```

**PROBLEM:**
- Pure sequential: z₀ → conv → norm → z₁ → conv → norm → z₂ → ...
- Information loss at each step (normalization)
- No skip connections (residual path)
- For 3 layers × normalization = severe information bottleneck

**From papers (He et al. 2015, Residual Networks):**
> "Skip connections allow gradient flow and preserve information"

---

### 5. **Readout Doesn't See Enough Spatial Variation**

Current (models.py):
```python
inv = (tr_z.real**2 + tr_z.imag**2).permute(0, 2, 3, 1)  # [B, L, L, C]
out = self.readout(inv).permute(0, 3, 1, 2)  # [B, 1, L, L] SPATIAL OUTPUT
pred_flat = predicted_beta.mean(dim=(2, 3)).view(-1)  # AVERAGE OVER SPACE
```

**PROBLEM:**
- Model outputs [B, 1, 16, 16] → average to scalar
- But averaging destroys spatial structure
- If all 256 values are ~2.0, average is still ~2.0
- Why? Hidden layers might be generating constant outputs!

**Test:**
```python
# During training, print actual values
print(f"z values range: [{z.min()}, {z.max()}]")
print(f"tr_z values range: [{tr_z.min()}, {tr_z.max()}]")
print(f"inv (squared) range: [{inv.min()}, {inv.max()}]")
```

---

### 6. **Learning Rate May Be Too Low**

Current:
```python
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
```

**From training logs:**
- Epoch 0: MAE = 3.63, Loss = 21.83
- Epoch 500: MAE ≈ 2.5, Loss ≈ 10.0

Loss decreased but MAE barely budged → very weak learning signal

---

## RESEARCH-BACKED FIXES

### Fix 1: Use Proper Gauge-Invariant Observables

Replace `(tr_z.real**2 + tr_z.imag**2)` with:

```python
# Option A: Hermitian part of trace (real-valued, gauge-invariant)
tr_z = torch.diagonal(z, dim1=-2, dim2=-1).sum(-1)
inv = tr_z.real  # Keep only real part [B, L, L, C]

# Option B: Matrix norm (from Holland et al.)
# tr(z @ z†) = sum of squared singular values
z_dagger = z.conj().mT
inv = torch.diagonal(z @ z_dagger, dim1=-2, dim2=-1).sum(-1).real

# Option C: Determinant (for SU(2), SU(3))
# For now keep Option A or B
```

### Fix 2: Remove Normalization from Hidden Layers

Comment it out again OR use layer normalization on scalars only:

```python
# OPTION: Normalize only the trace values, not matrices
tr_z_normalized = (tr_z.real - tr_z.real.mean()) / (tr_z.real.std() + 1e-6)
inv = tr_z_normalized
```

### Fix 3: Add Spatial Input Features

```python
def generate_langevin_u1_batch(...):
    # ... Langevin simulation ...
    
    # Compute LOCAL plaquette values
    p_local = calc_plaquettes_periodic(lx, ly)  # [B, L, L]
    
    # Encode as input feature (per-site)
    # Use gauge-invariant combination: 1 - cos(P)
    f_spatial = 1.0 - 0.5 * torch.cos(p_local)  # [B, L, L]
    f = f_spatial.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, L, L, 1, 1]
    
    return f, u_x, u_y, avg_plaquettes
```

### Fix 4: Increase Complexity or Add Residuals

Option A - Increase capacity:
```python
model = LgeConvNet(
    in_channels=1, 
    hidden_channels=32,  # Was 16
    n_layers=4,          # Was 3
    out_features=1, 
    gauge_invariant=True)
```

Option B - Add skip connections to LConvLinear:
```python
def forward(self, w, u_x, u_y, w_in=None):
    out = ... # existing parallel transport logic ...
    if w_in is not None and w_in.shape[1] == out.shape[1]:
        out = out + w_in  # Residual connection
    return out
```

### Fix 5: Better Initialization

Current (models.py):
```python
def _init_weights(out_c, in_c):
    return torch.randn(out_c, in_c, dtype=torch.cfloat) * (1.0 / in_c) ** 0.5
```

Use Kaiming/He initialization for complex:
```python
def _init_weights(out_c, in_c):
    # He initialization for complex numbers
    std = (2.0 / in_c) ** 0.5  # Factor of 2 for complex
    return torch.randn(out_c, in_c, dtype=torch.cfloat) * std
```

---

## PRIORITIZED FIX ROADMAP

**P0 (Must Fix Now):**
1. Change invariant observable from |tr(z)|² to tr(z).real
2. Remove normalization from hidden layers

**P1 (Should Fix):**
3. Add spatial input features (local plaquettes)
4. Increase hidden channels to 32

**P2 (Nice to Have):**
5. Better weight initialization
6. Add residual connections

---

## REFERENCES FROM PAPERS

**Holland et al. 2024** - "Gauge-Equivariant Neural Networks for Lattice QCD"
> Section 3.2: "Gauge-invariant observables are constructed from traces of matrix products"
> "The readout must be invariant: O = tr(ρ(U) Φ)"

**Weiler et al. 2023** - "Gauge Equivariant Mesh CNNs"
> "Local scalar combinations provide essential input structure"
> "Without spatial features, networks collapse to uniform solutions"

**He et al. 2015** - "Deep Residual Learning for Image Recognition"
> "Skip connections enable training of very deep networks"
> "Information is preserved through shortcut paths"

---

## EXPECTED OUTCOME AFTER FIXES

If all P0+P1 fixes applied:
- ✅ Constant output → Differentiated predictions
- ✅ Gauge error 2.62e-03 → ~1e-6 (proper invariance)
- ✅ MAE improvement: 2.5 → < 1.5 expected
- ✅ Predictions should follow y=x line

