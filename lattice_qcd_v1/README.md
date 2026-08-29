# lattice_qcd

U(1) lattice gauge theory: gauge invariance/equivariance demos and a
gauge-equivariant CNN ("L-CNN" style, cf. Favoni, Ipp, Muller & Schuh 2020;
bilinear layer per Holland, Ipp, Muller & Wenger 2024).

## Structure

- `src/lattice.py` — lattice geometry, local gauge transforms, link operations
- `src/observables.py` — Wilson loops / plaquettes, gauge-covariant kinetic term, topological charge (TODO)
- `src/models.py` — gauge-equivariant CNN layers (conv + bilinear + invariant readout) and a non-equivariant control CNN. See the module docstring for the charge-bookkeeping rules that make the bilinear layer safe to combine with the rest of the network.
- `experiments/u1_2d_simulation.py` — Wilson loop invariance + single-layer/full-model equivariance checks
- `experiments/equivariance_demo.py` — Phase 1 result: trains the L-CNN and a plain CNN on the same gauge-invariant regression target, then shows only the L-CNN's predictions survive a gauge transform of the input (~1e-8 vs ~1e-1 sensitivity)
- `tests/test_equivariance.py` — unit tests verifying every layer's charge-transformation law numerically (run this after touching `models.py`)
- `outputs/plots/` — generated figures (gitignored)

## Run

```
pip install -r requirements.txt
python tests/test_equivariance.py       # verify the math before trusting any experiment
python experiments/u1_2d_simulation.py
python experiments/equivariance_demo.py
```
