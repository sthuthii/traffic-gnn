# src/data_utils.py
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

def load_metr_la(data_folder="data/metr_la"):
    """
    Load METR-LA dataset.
    Returns:
        speeds_scaled: normalized speed data (timesteps x num_sensors)
        adj: adjacency matrix (num_sensors x num_sensors)
        scaler: fitted StandardScaler
    """
    h5_path = os.path.join(data_folder, "metr-la.h5")
    adj_path = os.path.join(data_folder, "adj_mx.pkl")

    # Load speeds
    with h5py.File(h5_path, "r") as f:
        speeds = np.array(f["speed"])
    print("Speeds shape:", speeds.shape)

    # Load adjacency
    if os.path.exists(adj_path):
        with open(adj_path, "rb") as f:
            adj = pickle.load(f, encoding="latin1")
        print("Adjacency shape:", adj.shape)
    else:
        # fallback: create synthetic adjacency
        N = speeds.shape[1]
        coords = np.random.rand(N, 2)
        adj = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(coords[i] - coords[j])
                w = np.exp(-d/0.1)
                adj[i,j] = w
                adj[j,i] = w
        print("Synthetic adjacency created:", adj.shape)

    # Normalize speeds
    scaler = StandardScaler()
    speeds_scaled = scaler.fit_transform(speeds)

    return speeds_scaled, adj, scaler

def create_windows(data, input_window=12, predict_window=3):
    """
    Create sequences for training: input_window -> predict_window
    """
    X, Y = [], []
    for i in range(len(data) - input_window - predict_window):
        X.append(data[i:i+input_window])
        Y.append(data[i+input_window:i+input_window+predict_window])
    return np.array(X), np.array(Y)
