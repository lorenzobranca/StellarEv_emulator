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


class LSTMEncoder(nn.Module):
    """Optional LSTM branch for sequence ICs."""
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        # x: (B, T_ic, D_ic) or (B, D_ic)
        if x.ndim == 2:
            x = x[:, None, :]
        B = x.shape[0]

        try:
            Cell = nn.OptimizedLSTMCell
        except AttributeError:
            Cell = nn.LSTMCell

        ScannedLSTM = nn.scan(
            Cell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        lstm = ScannedLSTM(features=int(self.hidden_size))

        c0 = jnp.zeros((B, int(self.hidden_size)))
        h0 = jnp.zeros((B, int(self.hidden_size)))
        carry0 = (c0, h0)

        carry_T, _ = lstm(carry0, x)
        return carry_T[1]  # final hidden


class LSTMRefiner(nn.Module):
    """Post-DeepONet LSTM that refines the predicted sequence."""
    hidden_size: int
    output_dim: int

    @nn.compact
    def __call__(self, y_seq):
        # y_seq: (B, N_eval, output_dim)
        assert y_seq.ndim == 3, f"Expected (B, N_eval, C), got {y_seq.shape}"
        B = y_seq.shape[0]

        try:
            Cell = nn.OptimizedLSTMCell
        except AttributeError:
            Cell = nn.LSTMCell

        ScannedLSTM = nn.scan(
            Cell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        lstm = ScannedLSTM(features=int(self.hidden_size))

        c0 = jnp.zeros((B, int(self.hidden_size)))
        h0 = jnp.zeros((B, int(self.hidden_size)))
        carry0 = (c0, h0)

        _carry_T, h_seq = lstm(carry0, y_seq)  # (B, N_eval, hidden_size)
        delta = nn.Dense(int(self.output_dim))(h_seq)
        return delta


class DON_LSTM(nn.Module):
    """
    DeepONet core + LSTM sequence refiner.

    DeepONet:
      - branch encodes IC -> latent vector
      - trunk encodes time grid -> latent features per time
      - dot product -> scalar trajectory
      - Dense -> output_dim channels

    LSTM head:
      - takes the predicted sequence and outputs a residual correction
    """
    latent_dim: int = 128
    output_dim: int = 7

    # Branch
    use_lstm_branch: bool = False
    branch_hidden: int = 128
    branch_mlp: Sequence[int] = (128, 128)

    # Trunk
    trunk_layers: Sequence[int] = (128, 128, 128)
    activation: Callable = nn.tanh

    # Refiner
    refiner_hidden: int = 128
    use_residual: bool = True

    def setup(self):
        if self.use_lstm_branch:
            self.branch_encoder = LSTMEncoder(hidden_size=int(self.branch_hidden))
            self.branch_proj = nn.Dense(int(self.latent_dim))
        else:
            self.branch_net = MLP((*self.branch_mlp, self.latent_dim), activation=self.activation)

        self.trunk_net = MLP((*self.trunk_layers, self.latent_dim), activation=self.activation)

        # scalar DeepONet output -> multivariate output
        self.output_head = nn.Dense(int(self.output_dim))

        # sequence refiner
        self.refiner = LSTMRefiner(hidden_size=int(self.refiner_hidden), output_dim=int(self.output_dim))

    def _branch_forward(self, ic):
        if self.use_lstm_branch:
            z = self.branch_encoder(ic)
            z = self.branch_proj(z)
            return z

        if ic.ndim == 3:
            ic = ic.reshape(ic.shape[0], -1)  # flatten sequence for MLP branch
        elif ic.ndim != 2:
            raise ValueError(f"Expected ic (B, D) or (B, T, D), got {ic.shape}")

        return self.branch_net(ic)

    def __call__(self, ic, t):
        """
        ic: (B, D_ic) or (B, T_ic, D_ic)
        t : (B, N_eval) or (B, N_eval, D_t)
        returns: (B, N_eval, output_dim)
        """
        # normalize t to 3D
        if t.ndim == 2:
            t_seq = t[..., None]  # (B, N_eval, 1)
        elif t.ndim == 3:
            t_seq = t
        else:
            raise ValueError(f"Expected t (B, N_eval) or (B, N_eval, D_t), got {t.shape}")

        B, N_eval, D_t = t_seq.shape

        # branch
        b = self._branch_forward(ic)  # (B, latent_dim)

        # trunk
        t_flat = t_seq.reshape(-1, D_t)                    # (B*N_eval, D_t)
        trunk_out = self.trunk_net(t_flat)                 # (B*N_eval, latent_dim)
        trunk_out = trunk_out.reshape(B, N_eval, self.latent_dim)

        # DeepONet scalar core
        y0_scalar = jnp.sum(trunk_out * b[:, None, :], axis=-1, keepdims=True)  # (B, N_eval, 1)

        # map to output channels
        y0 = self.output_head(y0_scalar)  # (B, N_eval, output_dim)

        # sequence refinement
        delta = self.refiner(y0)

        return y0 + delta if self.use_residual else delta

