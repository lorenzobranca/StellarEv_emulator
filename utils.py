import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Data scaler function
def data_scaler(data):

    max_data, min_data = jnp.amax(data), jnp.amin(data)

    data = (data - min_data) / (max_data - min_data)
    return data, max_data, min_data

def data_scaler_log(data):
    

    data = jnp.log10(data)
    max_data, min_data = jnp.amax(data), jnp.amin(data)

    print("!!!!!!!", max_data, min_data)

    data = (data - min_data) / (max_data - min_data) 
    return data, max_data, min_data

def data_scaler_log_NaN(data):

    

    data = jnp.log10(data)
    max_data, min_data = jnp.nanmax(data), jnp.nanmin(data)



    data = (data - min_data) / (max_data - min_data)
    return data, max_data, min_data



# Data descaler function 
def data_descaler(data, max_data, min_data):
    data = (max_data - min_data)*data + min_data
    return data

def train_test_split(X, Y, test_ratio=0.1, seed=0):
    """Splits X and Y into train and test sets using JAX."""
    key = jax.random.PRNGKey(seed)
    n = X.shape[0]
    indices = jax.random.permutation(key, n)
    split_idx = int(n * (1 - test_ratio))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]


def train_test_split_unaligned(X, Y, t, test_ratio=0.1, seed=0):
    """Splits X and Y into train and test sets using JAX."""
    key = jax.random.PRNGKey(seed)
    n = X.shape[0]
    indices = jax.random.permutation(key, n)
    split_idx = int(n * (1 - test_ratio))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx], t[train_idx], t[test_idx]


def plot_predictions(y_true, y_pred, sample_idx, save_path="plots/pred_vs_true.png"):
    """
    Plot predicted vs true outputs for a single sample over time.
    Args:
        y_true: Array of shape (T, 7) – ground truth
        y_pred: Array of shape (T, 7) – predicted output
        sample_idx: Index of the sample from the test set
        save_path: Where to save the resulting plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(18, 10))
    for i in range(7):
        plt.subplot(3, 3, i+1)
        plt.plot(y_true[:, i], label="True", linewidth=2)
        plt.plot(y_pred[:, i], label="Predicted", linestyle='--')
        plt.title(f"Output {i}")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

    plt.suptitle(f"Predicted vs True Outputs (Sample #{sample_idx})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

