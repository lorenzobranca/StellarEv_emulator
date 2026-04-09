import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from functools import partial


class TrainState(train_state.TrainState):
    pass


def l2_norm(params):
    return sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))


def monotonic_penalty(
    y_pred: jnp.ndarray,
    mono_idx: int | None,
    mono_weight,
    mode: str = "hinge",
    beta: float = 10.0,
) -> jnp.ndarray:
    """
    Penalize violations of monotonic non-decreasing behavior for channel mono_idx.
    y_pred: (B, T, D)
    """
    if mono_idx is None:
        return jnp.array(0.0, dtype=y_pred.dtype)

    dy = y_pred[:, 1:, mono_idx] - y_pred[:, :-1, mono_idx]  # (B, T-1)

    if mode == "hinge":
        viol = jnp.minimum(dy, 0.0)
        base = jnp.mean(viol**2)
    elif mode == "softplus":
        base = jnp.mean(jax.nn.softplus(-beta * dy) / beta)
    else:
        raise ValueError(f"Unknown monotonic penalty mode: {mode}")

    # IMPORTANT: no Python comparison on mono_weight (it can be traced)
    w = jnp.asarray(mono_weight, dtype=y_pred.dtype)
    w = jnp.maximum(w, 0.0)  # if you pass negative, treat as 0
    return w * base

def log_cumsum_loss(y_pred,
               y_true,
               lambda_logsum
               ):
    y_pred = jnp.cumsum(y_pred, axis=1)
    y_true = jnp.cumsum(y_true, axis=1)
    y_pred = jnp.log10(jnp.clip(y_pred, a_min=1e-12))
    y_true = jnp.log10(jnp.clip(y_true, a_min=1e-12))

    return lambda_logsum * jnp.mean((y_pred - y_true) ** 2)

def cumsum_loss(y_pred, y_true, lambda_logsum):
    y_pred = jnp.cumsum(y_pred, axis=1)
    y_true = jnp.cumsum(y_true, axis=1)
    return lambda_logsum * jnp.mean((y_pred - y_true) ** 2)

def batched_loss(
    params,
    apply_fn,
    ic_data,
    y_data,
    batch_size=64,
    l2_reg=0.0,
    mono_idx: int | None = None,
    mono_weight: float = 0.0,
    mono_mode: str = "hinge",
    mono_beta: float = 10.0,
    lambda_logsum: float = 1.0
    ):
    n = ic_data.shape[0]
    num_batches = (n + batch_size - 1) // batch_size

    total_loss = 0.0
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)

        ic_batch = ic_data[start:end]
        y_batch = y_data[start:end]

        pred = apply_fn(params, ic_batch)
        # pred = jnp.abs(pred)  # ensure non-negativity for monotonic penalty
        pred = jnp.where(pred < 0, -pred, pred)  # more stable than abs for gradients
        mse = jnp.mean((pred - y_batch) ** 2)
        mono = monotonic_penalty(pred, mono_idx, mono_weight, mode=mono_mode, beta=mono_beta)
        # log_cumsum = log_cumsum_loss(pred, y_batch, lambda_logsum)
        cm_loss = cumsum_loss(pred, y_batch, lambda_logsum)

        total_loss += (mse + mono + cm_loss) 

    total_loss = total_loss / num_batches
    return total_loss + l2_reg * l2_norm(params)


@partial(jax.jit, static_argnames=("mono_idx", "mono_mode"))
def train_step(
    state,
    ic_batch,
    y_batch,
    l2_reg=0.0,
    mono_idx: int | None = None,
    mono_weight: float = 0.0,
    mono_mode: str = "hinge",
    mono_beta: float = 10.0,
    lambda_logsum: float = 1.0
):
    def loss_fn(params):
        pred = state.apply_fn(params, ic_batch)
        pred = jnp.where(pred < 0, -pred, pred)  # more stable than abs for gradients
        mse = jnp.mean((pred - y_batch) ** 2)
        mono = monotonic_penalty(pred, mono_idx, mono_weight, mode=mono_mode, beta=mono_beta)
        # log_cumsum = log_cumsum_loss(pred, y_batch, lambda_logsum)
        cm_loss = cumsum_loss(pred, y_batch, lambda_logsum)
        return mse + l2_reg * l2_norm(params) + mono + cm_loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


def train(
    model,
    ic_train,
    y_train,
    ic_test=None,
    y_test=None,
    num_epochs=100,
    batch_size=32,
    lr=1e-3,
    l2_reg=0.0,
    seed=0,
    eval_every=1,   # evaluate test loss every N epochs (1 = true best-epoch tracking)
    log_every=10,   # print losses every N epochs

    # --- monotonic constraint controls ---
    mono_idx: int | None = -1,     # default: last output channel
    mono_weight: float = 0.0,      # set >0 to enable penalty
    mono_mode: str = "hinge",      # "hinge" or "softplus"
    mono_beta: float = 10.0,       # only used for softplus

    # --logcumsum controls---
    lambda_logsum: float = 1.0,           # weight for log-cumsum loss
):
    """
    If ic_test/y_test are provided, returns the best TrainState (lowest test loss).
    Monotonic penalty is applied to predictions only (not labels).
    """
    assert ic_train.shape[0] == y_train.shape[0], \
        "ic_train and y_train must have the same batch dimension"

    key = jax.random.PRNGKey(seed)

    # state.params stores the full variables dict returned by model.init(...)
    params = model.init(key, ic_train[:1])

    optimizer = optax.adamw(lr)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    n = ic_train.shape[0]
    num_batches = (n + batch_size - 1) // batch_size

    # --- best-epoch tracking (by test loss) ---
    has_test = (ic_test is not None) and (y_test is not None)
    best_state = state
    best_epoch = -1
    best_test_loss = float("inf")

    for epoch in range(num_epochs):
        key, perm_key = jax.random.split(key)
        perm = jax.random.permutation(perm_key, n)

        ic_shuf = ic_train[perm]
        y_shuf = y_train[perm]

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n)

            ic_batch = ic_shuf[start:end]
            y_batch = y_shuf[start:end]

            state = train_step(
                state,
                ic_batch,
                y_batch,
                l2_reg=l2_reg,
                mono_idx=mono_idx,
                mono_weight=mono_weight,
                mono_mode=mono_mode,
                mono_beta=mono_beta,
                lambda_logsum =lambda_logsum
            )

        # --- evaluation (for best checkpoint selection) ---
        test_loss_f = None
        if has_test and (epoch % eval_every == 0):
            test_loss = batched_loss(
                state.params, state.apply_fn, ic_test, y_test,
                batch_size=batch_size, l2_reg=l2_reg,
                mono_idx=mono_idx, mono_weight=mono_weight,
                mono_mode=mono_mode, mono_beta=mono_beta,
                lambda_logsum=lambda_logsum
            )
            test_loss_f = float(test_loss)

            if test_loss_f < best_test_loss:
                best_test_loss = test_loss_f
                best_epoch = epoch
                best_state = state

        # --- logging ---
        if epoch % log_every == 0:
            train_loss = batched_loss(
                state.params, state.apply_fn, ic_train, y_train,
                batch_size=batch_size, l2_reg=l2_reg,
                mono_idx=mono_idx, mono_weight=mono_weight,
                mono_mode=mono_mode, mono_beta=mono_beta,
                lambda_logsum=lambda_logsum
            )
            log = f"Epoch {epoch} | Train Loss: {float(train_loss):.4e}"

            if has_test:
                if test_loss_f is None:
                    test_loss = batched_loss(
                        state.params, state.apply_fn, ic_test, y_test,
                        batch_size=batch_size, l2_reg=l2_reg,
                        mono_idx=mono_idx, mono_weight=mono_weight,
                        mono_mode=mono_mode, mono_beta=mono_beta,
                    )
                    test_loss_f = float(test_loss)
                log += f" | Test Loss: {test_loss_f:.4e} | Best: {best_test_loss:.4e} (ep {best_epoch})"

            if mono_weight > 0.0 and mono_idx is not None:
                log += f" | Mono(idx={mono_idx}, w={mono_weight:g}, mode={mono_mode})"

            print(log)

    if has_test:
        print(f"[best checkpoint] epoch={best_epoch}, best_test_loss={best_test_loss:.6e}")
        return best_state

    return state
