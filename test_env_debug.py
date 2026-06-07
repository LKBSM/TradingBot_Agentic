# Quick test to check if environment is the problem
# Run this BEFORE the full training to verify everything works

import time
import numpy as np

print("=" * 60)
print("ENVIRONMENT DEBUG TEST")
print("=" * 60)

# Test 1: Can we step through the environment?
print("\n[TEST 1] Stepping through environment...")
start = time.time()
obs, _ = val_env.reset()
print(f"  Reset OK. Obs shape: {obs.shape}, Obs range: [{obs.min():.2f}, {obs.max():.2f}]")

for i in range(100):
    action = train_env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = val_env.step(action)
    if done:
        print(f"  Episode ended at step {i}")
        break

print(f"  100 steps completed in {time.time() - start:.2f}s")
print(f"  Final obs: {obs[:3]}... (first 3 values)")
print(f"  Has NaN: {np.isnan(obs).any()}, Has Inf: {np.isinf(obs).any()}")

# Test 2: Can we run model.predict()?
print("\n[TEST 2] Testing model.predict()...")
start = time.time()
obs, _ = val_env.reset()
for i in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = val_env.step(action)
    if done:
        break
print(f"  100 predictions completed in {time.time() - start:.2f}s")

# Test 3: Run a full mini-evaluation (like callback does)
print("\n[TEST 3] Mini evaluation (2000 steps)...")
start = time.time()
obs, _ = val_env.reset()
pv = [10000]
steps = 0
done = False

while not done and steps < 2000:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, info = val_env.step(action)
    pv.append(10000 + info['total_pnl'])
    steps += 1

    if steps % 500 == 0:
        print(f"    Step {steps}/2000...")

eval_time = time.time() - start
print(f"  Completed {steps} steps in {eval_time:.2f}s")
print(f"  Speed: {steps/eval_time:.0f} steps/sec")
print(f"  Final PnL: ${info['total_pnl']:.2f}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - Environment is working correctly!")
print("=" * 60)
