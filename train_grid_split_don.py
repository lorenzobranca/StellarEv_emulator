import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class TrainState(train_state.TrainState):
    pass


def l2_norm(params):
    return sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))


def batched_loss(params, apply_fn, ic_data, y_data, batch_size=64, l2_reg=0.0):
    n = ic_data.shape[0]
    num_batches = (n + batch_size - 1) // batch_size

    total_loss = 0.0
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)

        ic_batch = ic_data[start:end]
        y_batch = y_data[start:end]

        pred = apply_fn(params, ic_batch)
        mse = jnp.mean((pred - y_batch) ** 2)
        total_loss += mse

    total_loss = total_loss / num_batches
    return total_loss + l2_reg * l2_norm(params)


@jax.jit
def train_step(state, ic_batch, y_batch, l2_reg=0.0):
    def loss_fn(params):
        pred = state.apply_fn(params, ic_batch)
        mse = jnp.mean((pred - y_batch) ** 2)
        return mse + l2_reg * l2_norm(params)

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
):
    assert ic_train.shape[0] == y_train.shape[0], \
        "ic_train and y_train must have the same batch dimension"

    key = jax.random.PRNGKey(seed)

    # Keep the same convention as your previous scripts:
    # state.params stores the full variables dict returned by model.init(...)
    params = model.init(key, ic_train[:1])

    optimizer = optax.adamw(lr)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    n = ic_train.shape[0]
    num_batches = (n + batch_size - 1) // batch_size

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

            state = train_step(state, ic_batch, y_batch, l2_reg=l2_reg)

        if epoch % 10 == 0:
            train_loss = batched_loss(
                state.params, state.apply_fn, ic_train, y_train,
                batch_size=batch_size, l2_reg=l2_reg
            )
            log = f"Epoch {epoch} | Train Loss: {float(train_loss):.4e}"

            if ic_test is not None and y_test is not None:
                test_loss = batched_loss(
                    state.params, state.apply_fn, ic_test, y_test,
                    batch_size=batch_size, l2_reg=l2_reg
                )
                log += f" | Test Loss: {float(test_loss):.4e}"

            print(log)

    return state

