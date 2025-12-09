# scripts/generate_synthetic_metrla.py
import numpy as np
import h5py
import os

os.makedirs("data/metr_la", exist_ok=True)

N = 50        # number of sensors (use 207 later)
T = 24*12*7   # one week of 5-min intervals (7 days) ~ 2016
print("Creating synthetic speeds with shape", (T, N))

# synthetic traffic speeds: base daily pattern + random noise + occasional spikes
time = np.arange(T)
# daily pattern: sin wave (24h => 288 time-steps per day)
daily = 10 * (np.sin(2 * np.pi * (time % 288) / 288 - 0.5) + 1)  # range ~0-20
base = daily.reshape(-1, 1) + 40  # base speed ~40-60
noise = np.random.randn(T, N) * 3.0
speeds = (base + noise).astype(np.float32)
speeds = np.clip(speeds, 0.5, 80.0)

# adjacency: simple distance-based grid neighbors for N sensors
coords = np.random.rand(N, 2) * 100
A = np.zeros((N, N), dtype=np.float32)
for i in range(N):
    for j in range(i+1, N):
        d = np.linalg.norm(coords[i] - coords[j])
        # connect if distance < threshold
        if d < 15:
            w = np.exp(-d/10.0)
            A[i, j] = w
            A[j, i] = w

# save in HDF5 with expected keys
h5_path = "data/metr_la/metr-la.h5"
with h5py.File(h5_path, "w") as f:
    f.create_dataset("speed", data=speeds, compression="gzip")
    # optionally include timestamps
    # create readable timestamps: start now, step 5 min
    import datetime
    start = datetime.datetime(2023,1,1,0,0)
    times = [(start + datetime.timedelta(minutes=5*i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(T)]
    # store as variable-length ASCII
    dt = h5py.string_dtype(encoding='utf-8')
    f.create_dataset("date", data=np.array(times, dtype=dt))
print("Saved synthetic file:", h5_path)

# save adjacency as numpy file
np.save("data/metr_la/adj_mat.npy", A)
print("Saved adjacency at data/metr_la/adj_mat.npy (shape {})".format(A.shape))
