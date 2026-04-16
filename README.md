# StellarEv_emulator

Emulator for stellar-evolution tracks built with DeepONet-style models.

This repository contains two complementary neural surrogates:

- a **time model** that predicts the time coordinate on a native latent domain `u ∈ [0,1]`
- an **output model** that predicts the stellar quantities on the same native domain

The main user-facing script is:

- **`make_inference_log.py`** — inference script for the current log-time model family

A second user-facing inference script is planned:

- **`make_inference_diff.py`** — future alternative expected to be less accurate at short times but more accurate at long times

---

## Repository structure

### User inference

- **`make_inference_log.py`**  
  Main entry point for users. Loads the trained time and output checkpoints, reconstructs the native curves, and provides:
  - predictions on the native domain `u`
  - predictions in physical time
  - optional queried outputs at user-requested times
  - plotting utilities for `time(u)`, `output(u)`, and `output(time)`

### Training scripts

- **`main_log15_time.py`**  
  Trains and evaluates the **time model**.

- **`main_grid_split_don.py`**  
  Trains and evaluates the **output model**.

- **`main_log15_time_diff.py`**, **`main_time_diff.py`**, **`main_log15_timeconcatenated.py`**  
  Experimental / alternative model variants.

### Plotting and analysis

- **`plot_combined.py`**, **`plot_combined_diff.py`**  
  Combine independently predicted time and output curves for visualization.

- **`plot_interpolation.py`**  
  Checks the interpolation used during preprocessing.

### Data preparation

- **`data_maker.py`** / **`data_maker.ipynb`**  
  Dataset generation and preprocessing.

### Core modules

- **`arch_grid_split_don.py`**  
  Neural architecture definition.

- **`train_grid_split_don.py`**  
  Training routine for the output model.

- **`train_grid_split_don_pos.py`**  
  Training routine for the time model.

- **`train_grid_split_don_pos_diff.py`**  
  Training routine for the diff-time variant.

- **`utils.py`**  
  Dataset utilities and helper functions.

---

## Conceptual workflow

The emulator works in two stages.

1. The **time model** predicts a curve on the native domain `u ∈ [0,1]`.
2. The **output model** predicts the stellar quantities on that same native domain.
3. The inference script combines both predictions to obtain curves in physical time.

So the native prediction domain is **not physical time directly**. Instead, the models first predict on the internal domain `u`, and physical-time curves are obtained afterward.

---

## Quick start

### 1. Install dependencies

The project is configured with a `pyproject.toml` file and requires Python `>=3.11,<3.12`.

Typical dependencies include:

- `jax[cuda]`
- `flax`
- `keras`
- `numpy`
- `scipy`
- `matplotlib`
- `optuna`

Install with your preferred environment manager.

### 2. Prepare checkpoints

Make sure the trained checkpoints exist in the expected folders:

- `checkpoints_new/deeponet_params_new_log15_time/`
- `checkpoints_new/deeponet_params_new_log15_output/`

### 3. Run inference

The main user script is:

```bash
python make_inference_log.py
```

This script generates:

- native prediction arrays (`.npz`)
- plots in the inference output folder

---

## Using `make_inference_log.py`

This script is intended for end users who want to:

- provide one set of initial conditions
- reconstruct the predicted time curve
- reconstruct the predicted stellar-output curves
- optionally evaluate outputs at specific requested times
- save and plot the results

At a high level, the script:

1. loads the trained checkpoints
2. rebuilds the two models with the saved architecture
3. predicts the time and output curves on the native domain
4. converts the time prediction back to physical time
5. optionally interpolates to user-requested times
6. saves results and figures

---

## Outputs

Depending on the selected options, the inference scripts produce plots such as:

- `time_vs_u.png`
- `outputs_vs_u.png`
- `outputs_vs_time.png`
- query/interpolation result plots

and NumPy archives such as:

- `native_curves.npz`
- `queried_outputs.npz`

---

## Notes

- The current user-focused inference workflow is based on **`make_inference_log.py`**.
- A future **`make_inference_diff.py`** will provide an alternative long-time inference mode.
- The time and output models should be used together only when they come from compatible training/preprocessing pipelines.

---

## License

This repository is released under the **GPL-3.0** license.
