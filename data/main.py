import h5py
import numpy as np

h5_path = "metr-la.h5"

with h5py.File(h5_path, "r") as f:
    print("Keys in H5:", list(f.keys()))
    speeds = np.array(f["speed"])
    print("Speeds shape:", speeds.shape)
