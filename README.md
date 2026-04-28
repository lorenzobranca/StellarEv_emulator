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
- `IC`: shape `(5,)` or `(B, 5)` for a batch
- `t_phys`: scalar or array of shape `(T,)` (physical query times)

### Outputs (7 channels)
The emulator returns **7 predicted quantities** per time. (Channel names depend on your configuration; check the plotting labels / `titles` list in the scripts for the exact mapping.)

---

## Quickstart

### 1) Create an environment
You need Python + JAX + Flax (GPU optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install jax jaxlib flax optax numpy scipy matplotlib
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

## Notes & troubleshooting

- **GPU selection**:
  ```bash
  export CUDA_VISIBLE_DEVICES=0
  ```
- **First run can be slow** if it has to download and extract checkpoints.
- If Zenodo filenames change, update the configuration used by `params_loader.py`.

---

## License
Add your chosen license here (e.g., MIT / BSD-3 / GPL-3).
