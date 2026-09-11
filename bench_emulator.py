"""Time the StellarEv emulator (log and diff variants) for a solar-mass star.

Usage: python bench_emulator.py [gpu|cpu] [log|diff]

Reference runtimes (collaborator, 2026-09-11, 1 Msun, stop at X_H,c < 1e-3 at 9.5 Gyr):
  non-rotating YREC track 100.233 s + rotevol 0.152 s = 100.385 s
  full YREC model with rotation                      = 142.004 s
"""
import os, sys, time
import numpy as np

device = sys.argv[1] if len(sys.argv) > 1 else "gpu"
variant = sys.argv[2] if len(sys.argv) > 2 else "log"

if device == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # stop the imported module from calling autocvd
else:
    from autocvd import autocvd
    autocvd(num_gpus=1)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
chosen = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT); sys.path.insert(0, ROOT)

import importlib
mod = importlib.import_module(f"make_inferences_{variant}")  # scripts call autocvd only if CUDA_VISIBLE_DEVICES is unset

import jax, jax.numpy as jnp
print(f"variant={variant} device={device} CUDA_VISIBLE_DEVICES={chosen} jax_devices={jax.devices()}", flush=True)

t0 = time.perf_counter()
pred = mod.CombinedPredictor(output_mode="scaled")
t_load = time.perf_counter() - t0

ic = np.array([1.0, 0.0, 1.5, 2.1, 0.3], dtype=np.float32)   # solar mass, solar metallicity
ages_gyr = np.array([9.5])                                    # collaborator's stopping age

def block(res):
    # results are numpy already (np.asarray inside predictors) -> host sync done
    return res

# 1) first call incl. compile/dispatch warm-up
t0 = time.perf_counter(); block(pred.predict(ic)); t_first = time.perf_counter() - t0

def bench(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); block(fn()); ts.append(time.perf_counter() - t0)
    ts = np.array(ts); return np.median(ts), ts.min(), ts.mean()

n = 50 if device == "gpu" else 20
# 2) single star, full native curves (3999 pts x 7 outputs + time axis)
m_native, mn_native, _ = bench(lambda: pred.predict(ic), n)
# 3) single star + query at 9.5 Gyr (adds interpolation work)
m_query, mn_query, _ = bench(lambda: pred.predict(ic, target_time=ages_gyr), n)
# 4) batched stars, native curves
res = {}
for B in (100, 1000):
    icb = np.tile(ic, (B, 1)).astype(np.float32)
    icb[:, 0] = np.linspace(0.6, 1.4, B)
    pred.predict(icb)  # warm-up for this shape
    m, mn, _ = bench(lambda: pred.predict(icb), 5)
    res[B] = m

# 5) jitted raw network forward (both models) for reference, single star
tm, om = pred.time_model, pred.output_model
fwd = jax.jit(lambda p1, p2, x: (tm.state.apply_fn(p1, x), om.state.apply_fn(p2, x)))
x1 = jnp.asarray(ic[None])
jax.block_until_ready(fwd(tm.state.params, om.state.params, x1))
m_jit, mn_jit, _ = bench(lambda: jax.block_until_ready(fwd(tm.state.params, om.state.params, x1)), 200)

print(f"RESULT variant={variant} device={device}")
print(f"  model load (one-off)            : {t_load:8.3f} s")
print(f"  first predict (incl. compile)   : {t_first:8.3f} s")
print(f"  single star, native curves      : median {m_native*1e3:8.2f} ms  (min {mn_native*1e3:.2f} ms)")
print(f"  single star + query @9.5 Gyr    : median {m_query*1e3:8.2f} ms  (min {mn_query*1e3:.2f} ms)")
for B, m in res.items():
    print(f"  batch {B:5d} stars, native curves : median {m*1e3:8.2f} ms total -> {m/B*1e3:.4f} ms/star")
print(f"  jitted NN forward only, 1 star  : median {m_jit*1e3:8.3f} ms  (min {mn_jit*1e3:.3f} ms)")
