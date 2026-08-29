# lattice_qcd

U(1) lattice gauge theory: gauge invariance/equivariance demos and a
gauge-equivariant CNN ("L-CNN" style, cf. Favoni, Ipp, Muller & Schuh 2020).

## Structure

- `src/lattice.py` — lattice geometry, local gauge transforms, link operations
- `src/observables.py` — Wilson loops / plaquettes, topological charge
- `src/models.py` — gauge-equivariant CNN layers and model
- `experiments/u1_2d_simulation.py` — runnable demo (Wilson loop invariance + CNN equivariance checks)
- `outputs/plots/` — generated figures (gitignored)

## Run

```
pip install -r requirements.txt
python experiments/u1_2d_simulation.py
```
