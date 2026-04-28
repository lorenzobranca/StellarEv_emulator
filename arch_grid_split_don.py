import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence, Callable


class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable = nn.tanh

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(int(feat))(x))
        x = nn.Dense(int(self.features[-1]))(x)
        return x


class GridSplitBranchDeepONet(nn.Module):
    """
    Split-branch DeepONet on a fixed internal grid.

    - No external sensor/time tensor.
    - The trunk is a learned basis Phi over a fixed grid of length n_eval.
    - Each output channel has its own branch MLP.

    Inputs
    ------
    ic : (B, D_ic) or (B, T_ic, D_ic)

    Output
    ------
    y  : (B, n_eval, output_dim)
    """
    branch_layers: Sequence[int] = (128, 128)
    latent_dim: int = 128
    output_dim: int = 7
    n_eval: int = 100
    activation: Callable = nn.tanh
    use_curve_bias: bool = True
    basis_init_scale: float = 0.02

    def setup(self):
        # One branch per output channel
        self.sub_branches = [
            MLP((*self.branch_layers, self.latent_dim), activation=self.activation)
            for _ in range(self.output_dim)
        ]

    def _prepare_ic(self, ic):
        # Flatten sequence ICs for MLP branches
        if ic.ndim == 3:
            ic = ic.reshape(ic.shape[0], -1)
        elif ic.ndim != 2:
            raise ValueError(f"Expected ic (B, D_ic) or (B, T_ic, D_ic), got {ic.shape}")
        return ic

    @nn.compact
    def __call__(self, ic):
        ic = self._prepare_ic(ic)

        # Shared learned trunk basis over the fixed grid: (n_eval, latent_dim)
        phi = self.param(
            "grid_basis",
            nn.initializers.normal(stddev=self.basis_init_scale),
            (int(self.n_eval), int(self.latent_dim)),
        )

        # Optional per-output learned mean curve / bias: (output_dim, n_eval)
        if self.use_curve_bias:
            curve_bias = self.param(
                "curve_bias",
                nn.initializers.zeros,
                (int(self.output_dim), int(self.n_eval)),
            )

        outputs = []
        for k, sub_branch in enumerate(self.sub_branches):
            b = sub_branch(ic)                              # (B, latent_dim)
            out = b @ phi.T                                # (B, n_eval)

            if self.use_curve_bias:
                out = out + curve_bias[k][None, :]         # (B, n_eval)

            outputs.append(out[:, :, None])                # (B, n_eval, 1)

        return jnp.concatenate(outputs, axis=-1)           # (B, n_eval, output_dim)

