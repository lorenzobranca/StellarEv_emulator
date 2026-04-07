import numpy as np
import matplotlib.pyplot as plt
import os


# =========================
PLOTS_DIR_TIME = "./plots_new_log15_time/"
PLOTS_DIR_OUTPUT =  "./plots_new_log15_output/"
PLOTS_DIR = "./plots_new_log15_combined/"
os.makedirs(PLOTS_DIR, exist_ok=True)

m = np.load(os.path.join(PLOTS_DIR_TIME, "predictions.npz"))['m']
s = np.load(os.path.join(PLOTS_DIR_TIME, "predictions.npz"))['s']
time_true = np.load(os.path.join(PLOTS_DIR_TIME, "predictions.npz"))['y_true']
output_true = np.load(os.path.join(PLOTS_DIR_OUTPUT, "predictions.npz"))['y_true']
time_pred = np.load(os.path.join(PLOTS_DIR_TIME, "predictions.npz"))['y_pred']
output_pred = np.load(os.path.join(PLOTS_DIR_OUTPUT, "predictions.npz"))['y_pred']
OUTPUT_DIM = output_true.shape[-1]
num_examples = 10
rng = np.random.default_rng(10)
idxs = rng.integers(low=0, high=output_pred.shape[0], size=num_examples)
print(f"Loaded predictions: time_true {time_true.shape}, output_true {output_true.shape}, time_pred {time_pred.shape}, output_pred {output_pred.shape}")
print(f"Example idxs: {idxs}")
colors = plt.cm.tab10.colors[:len(idxs)]
titles = [
        r"log($T_{\mathrm{eff}}$) - Effective Surface Temperature",
        r"$P_{\mathrm{rot}}$ (days) - Rotation Period",
        r"$B_{\mathrm{coronal}}/B_\odot$ - Coronal Magnetic Field Strength",
        r"$P_{\mathrm{atm}}$ - Photospheric Pressure (cgs)",
        r"$\tau_{\mathrm{cz}}$ - Convective Turnover Time (s)",
        r"$\dot{M}$ - Mass Loss Rate",
        r"Luminosity - Predicted Stellar Luminosity",
    ]

fig, axes = plt.subplots(1, OUTPUT_DIM, figsize=(25, 4), sharex=True)

for i in range(OUTPUT_DIM):
    ax = axes[i] if OUTPUT_DIM > 1 else axes
    for j, idx in enumerate(idxs):
        color = colors[j]
        y_true = np.array(output_true[idx, :, i])
        y_model = np.array(output_pred[idx, :, i])

        # same color for truth and prediction, different linestyle
        time_true_plot = 1.5**(time_true[idx, :, 0]*s + m)
        time_true_plot[time_true_plot < 1 ] = np.log10(time_true_plot[time_true_plot < 1 ])
        time_pred_plot = 1.5**(time_pred[idx, :, 0]*s + m)
        time_pred_plot[time_pred_plot < 1 ] = np.log10(time_pred_plot[time_pred_plot < 1 ])
        ax.plot(time_true_plot, y_true, color=color, linestyle="-")
        ax.plot(time_pred_plot, y_model, color=color, linestyle="--")

    ax.set_title(titles[i] if i < len(titles) else f"Output {i}", fontsize=10)
    ax.grid(True)
    if i == 0:
        ax.set_ylabel("Scaled value")
    if i == max(0, OUTPUT_DIM // 2):
        ax.set_xlabel("Fixed grid (0..1)")

fig.suptitle("Grid Split-Branch DeepONet vs Reference", fontsize=14)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

outpath = os.path.join(PLOTS_DIR, "test_predictions_comparison.png")
plt.savefig(outpath, dpi=300)
plt.close()
print(f"Figure saved: {outpath}")

