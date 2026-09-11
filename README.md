# StellarEv_emulator

Fast emulator for a 7-dimensional stellar-evolution output vector as a function of **5 initial-condition (IC) parameters** and **time**.

The emulator is built as a **Neural Operator / DeepONet-style model** evaluated on an internal fixed grid, plus a learned **time-mapping model**. In practice, you provide:

- **IC**: a vector of length **5**
- **t_phys**: one or more **physical times**

and you get back:

- **y(t)**: a vector of length **7** (or a batch of them), in physical units (after de-scaling).

---

## Which inference script should I use?

This repository provides **two user-facing inference scripts**, optimized for different time regimes:

- **`make_inferences_log.py`** → **best at early times** (small `t_phys`)
  - Uses a log-time treatment to better resolve early-time dynamics.

- **`make_inferences_diff.py`** → **best at late times** (large `t_phys`)
  - Uses a time model that stays accurate when the system evolves slowly and late-time precision matters.

A simple rule of thumb:
- If your science depends on **early evolution** (small times / rapid transients), start with `make_inferences_log.py`.
- If your science depends on **late evolution** (large times / slow drift), start with `make_inferences_diff.py`.

You can also run both and stitch results if you want a single curve spanning the full time range.

---

## Model I/O

### Inputs

- `IC`: shape `(5,)` or `(B, 5)` for a batch, `float32`
- `t_phys`: scalar or array of shape `(T,)` (physical query times, in **Gyr**)

The 5 initial-condition parameters, **in this exact order**, are:

| idx | name    | meaning                                                                 | units         |
|-----|---------|-------------------------------------------------------------------------|---------------|
| 0   | `Mstar` | stellar mass                                                            | M<sub>☉</sub> |
| 1   | `FeH`   | metallicity `[Fe/H]`                                                    | dex           |
| 2   | `PMMA`  | rotational-evolution parameter — mass loss vs. angular velocity: `dM/dt ∝ ω^PMMA` | dimensionless |
| 3   | `PMMB`  | rotational-evolution parameter — magnetic field vs. angular velocity: `B ∝ ω^PMMB` | dimensionless |
| 4   | `PMMM`  | rotational-evolution parameter — angular-momentum coupling `m`: `dJ/dt ∝ B^(4m) · (dM/dt)^(1−2m)` | dimensionless |

`Mstar`/`FeH` set the stellar model; `PMMA`/`PMMB`/`PMMM` are the Rotevol
angular-momentum-evolution scaling exponents. Example (from the scripts):
`IC = [1.0, 0.1, 1.5, 2.1, 0.3]`.

> **Tip:** To reproduce a Skumanich spin-down index, set `PMMA = 2`,
> `PMMB = 1`, and `PMMM = 0.22`.

#### ⚠️ Training-set boundaries (valid input domain)

The emulator was trained on **4320 simulations** sampled on a **discrete grid**.
Predictions are only reliable **inside** the ranges below — the model
**interpolates** within the sampled grid, and **extrapolating outside these
ranges (or far from the sampled grid points) is not safe** and can give
non-physical results.

| idx | param   | min | max | sampled grid values                    |
|-----|---------|-----|-----|----------------------------------------|
| 0   | `Mstar` | 0.2 | 1.4 | 0.2 → 1.3 in steps of 0.05, then 1.4 (M<sub>☉</sub>) |
| 1   | `FeH`   | 0.0 | 0.2 | 0.0, 0.1, 0.2 (dex)                    |
| 2   | `PMMA`  | 1.0 | 2.0 | 1.0, 2.0                               |
| 3   | `PMMB`  | 0.5 | 5.0 | 0.5, 1.0, 2.0, 3.0, 4.0, 5.0          |
| 4   | `PMMM`  | 0.1 | 0.5 | 0.1, 0.2, 0.3, 0.4, 0.5               |

- **Physical time `t_phys`**: the training tracks span roughly
  **1.1×10⁻⁶ Gyr to 13.8 Gyr**. The exact valid time window is
  IC-dependent; the inference scripts warn when a requested time falls outside
  the model's predicted range for a given IC.

### Outputs (7 channels)

The emulator returns **7 predicted quantities** per query time, **in this exact order**:

| idx | name       | meaning                        | units                    |
|-----|------------|--------------------------------|--------------------------|
| 0   | `logTeff`  | effective surface temperature  | log₁₀(K)                 |
| 1   | `Prot`     | rotation period                | days                     |
| 2   | `Bcoronal` | coronal magnetic field strength| B / B<sub>☉</sub> (dimensionless) |
| 3   | `Patm`     | photospheric pressure          | cgs                      |
| 4   | `tau_cz`   | convective turnover time       | s                        |
| 5   | `dMdt`     | mass-loss rate                 | M<sub>☉</sub>/yr         |
| 6   | `luminosity` | stellar luminosity           | L<sub>☉</sub>            |

(These match the `output_cols` list in `data_maker.py` and the `OUTPUT_NAMES`
labels in the inference scripts.)

---

## Quickstart

### 1) Create an environment
You need Python + JAX + Flax (GPU optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install jax jaxlib flax optax numpy scipy matplotlib autocvd
```

> If you use CUDA, install the correct JAX build for your CUDA version (see JAX docs).

### 2) Run one of the inference scripts

```bash
python make_inferences_log.py
# or
python make_inferences_diff.py
```

Each script contains an example section where you set:
- `ic_single = np.array([...], dtype=np.float32)`  (length 5)
- `user_time = np.array([...], dtype=np.float64)`  (physical times)

---

## Using the emulator from Python

Both scripts expose a `CombinedPredictor` class. Example:

```python
import numpy as np
from make_inferences_log import CombinedPredictor  # early-time
# from make_inferences_diff import CombinedPredictor  # late-time

pred = CombinedPredictor(output_mode="physical")

ic = np.array([1.0, 0.1, 1.5, 2.1, 0.3], dtype=np.float32)  # (5,)
tq = np.array([1e-4, 1e-2, 1e0], dtype=np.float64)

y = pred.predict(ic, target_time=tq)  # returns 7 outputs per query time
```

---

## Model checkpoints (too large for GitHub)

The trained checkpoint folders are **not stored in this repository** because they exceed GitHub’s 100 MB file limit.

Instead:
- They are stored on **Zenodo**.
- The inference scripts include a loader that:
  1) tries to load from a **local path** (fast, offline),
  2) otherwise **downloads** the corresponding `.zip` from Zenodo,
  3) extracts it into the expected checkpoint folder structure,
  4) restores the Flax/JAX checkpoint.

Zenodo record:
```
https://zenodo.org/records/19736519
```

### Expected local layout
```
checkpoints_new/
  deeponet_params_new_log15_output/
    checkpoint_0/...
  deeponet_params_new_log15_time/
    checkpoint_0/...
  deeponet_params_new_log15_time_diff/
    checkpoint_0/...
```

- `make_inferences_log.py` uses **output** + **time** (`..._time`)
- `make_inferences_diff.py` uses **output** + **time_diff** (`..._time_diff`)

If you already have the checkpoints locally, place them under `checkpoints_new/` and the scripts will not download anything.

---

## Performance

Measured 2026-09-11 on one NVIDIA A100 (40 GB) and an AMD EPYC 7452 CPU with JAX 0.11,
for a solar-mass star, using `bench_emulator.py`. Each `CombinedPredictor.predict` call returns
the full 7-channel track on the 3999-point internal grid plus the age axis (out to ~13 Gyr).
Both predictors `jax.jit` their network forward pass at load time. One-off costs are excluded:
model loading (~6 s) and first-call compilation (~1 s per new batch shape).

| mode (log variant)               | per star, GPU | per star, CPU |
|----------------------------------|--------------:|--------------:|
| single star                      |        3.1 ms |        7.5 ms |
| single star + age query          |        3.5 ms |        8.3 ms |
| batch of 1000 stars              |       0.16 ms |       0.33 ms |
| network forward only (2 models)  |       0.71 ms |        6.6 ms |

The diff variant (`make_inferences_diff.py`, public Zenodo checkpoint) times within noise of the
log variant: 4.4 ms per single star and 0.19 ms per star in a batch of 1000 on the GPU.

**Speed-up over the physical code.** Reference timings for a 1 M☉ star evolved to 9.5 Gyr
(stopping at central X_H < 10⁻³): non-rotating YREC track 100.2 s + rotevol 0.15 s = **100.4 s**;
full YREC model with rotation **142.0 s** (single run, one CPU).

| emulator mode (GPU)     | vs YREC + rotevol (100.4 s) | vs rotating YREC (142.0 s) |
|-------------------------|----------------------------:|---------------------------:|
| single star             |                    ~3 × 10⁴ |                 ~4.6 × 10⁴ |
| batch of 1000, per star |                    ~6 × 10⁵ |                   ~9 × 10⁵ |

Round numbers: four orders of magnitude for single-star latency, five to six for batched
throughput. Full measurements, ratios and caveats are in `speedup_report.txt`; reproduce with

```bash
python bench_emulator.py gpu log    # or: cpu | diff
```

---

## Notes & troubleshooting

- **GPU selection**: the scripts call `autocvd(num_gpus=1)` at import to pick a free GPU,
  but only if `CUDA_VISIBLE_DEVICES` is not already set. To force a device, or to run on CPU:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python make_inferences_log.py   # specific GPU
  CUDA_VISIBLE_DEVICES="" python make_inferences_log.py  # CPU only
  ```
- **First call per batch shape** takes ~1 s extra: the network forward pass is `jax.jit`-compiled
  once per distinct input batch size. Repeated calls with the same shape are fast (see *Performance*).
- **First run can be slow** if it has to download and extract checkpoints.
- If Zenodo filenames change, update the configuration used by `params_loader.py`.

---
