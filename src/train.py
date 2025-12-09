# src/train.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_utils import load_metr_la, create_windows
from src.model import GNNTrafficPredictor

# ---------- Simple Dataset ----------
class TrafficTorchDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ---------- Training one epoch ----------
def train_one_epoch(model, loader, optimizer, loss_fn, device, A_tensor):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:

        # xb or yb may already be tensors (because DataLoader auto-converts numpy to tensor)
        if isinstance(xb, np.ndarray):
            xb = torch.from_numpy(xb).float()
        if isinstance(yb, np.ndarray):
            yb = torch.from_numpy(yb).float()

        xb = xb.to(device).unsqueeze(-1)   # [B, in_len, N, 1]
        yb = yb.to(device)                 # [B, out_len, N]

        optimizer.zero_grad()
        preds = model(xb, A_tensor)        # -> [B, N, out_window]
        preds = preds.permute(0, 2, 1)     # -> [B, out_len, N]

        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, A_tensor):
    model.eval()
    preds_all, trues_all = [], []

    with torch.no_grad():
        for xb, yb in loader:

            if isinstance(xb, np.ndarray):
                xb = torch.from_numpy(xb).float()
            if isinstance(yb, np.ndarray):
                yb = torch.from_numpy(yb).float()

            xb = xb.to(device).unsqueeze(-1)
            yb = yb.to(device)

            preds = model(xb, A_tensor)               # [B, N, out_window]
            preds = preds.permute(0, 2, 1).cpu().numpy()
            trues = yb.cpu().numpy()

            preds_all.append(preds)
            trues_all.append(trues)

    preds_all = np.concatenate(preds_all, axis=0)
    trues_all = np.concatenate(trues_all, axis=0)

    mae = mean_absolute_error(trues_all.flatten(), preds_all.flatten())
    mse = mean_squared_error(trues_all.flatten(), preds_all.flatten())
    rmse = mse ** 0.5

    return mae, rmse, preds_all, trues_all




# ---------- Main training ----------
def main(
    data_folder="data/metr_la",
    input_window=12,
    predict_window=3,
    batch_size=32,
    epochs=12,
    lr=1e-3,
    model_save_path="saved_models/stgcn_from_scratch.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    print("Device:", device)

    # Load data
    speeds_scaled, adj, _ = load_metr_la(data_folder)
    T, N = speeds_scaled.shape
    print("Loaded speeds:", speeds_scaled.shape, "Adj:", adj.shape)

    # windows
    X, Y = create_windows(speeds_scaled, input_window, predict_window)
    print("Created windows X.shape, Y.shape:", X.shape, Y.shape)

    # time split
    S = X.shape[0]
    s1, s2 = int(0.7 * S), int(0.9 * S)

    X_train, Y_train = X[:s1], Y[:s1]
    X_val, Y_val = X[s1:s2], Y[s1:s2]
    X_test, Y_test = X[s2:], Y[s2:]

    print("Train/Val/Test shapes:", X_train.shape, X_val.shape, X_test.shape)

    # datasets
    train_loader = DataLoader(TrafficTorchDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TrafficTorchDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TrafficTorchDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

    A_tensor = torch.tensor(adj, dtype=torch.float32, device=device)

    # model
    model = GNNTrafficPredictor(
        num_nodes=N,
        hidden_dim=32,
        gcn_layers=2,
        gru_hidden=64,
        out_window=predict_window
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    best_val = float("inf")

    # ---------- Training Loop ----------
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, A_tensor)
        val_mae, val_rmse, _, _ = evaluate(model, val_loader, device, A_tensor)

        print(f"Epoch {epoch}/{epochs}  Train MAE: {train_loss:.4f}  Val MAE: {val_mae:.4f}  Val RMSE: {val_rmse:.4f}")

        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), model_save_path)
            print("Saved best model ->", model_save_path)

    # ---------- Load best model ----------
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    model.eval()

    # ---------- Test ----------
    test_mae, test_rmse, preds, trues = evaluate(model, test_loader, device, A_tensor)
    print("Test MAE:", test_mae, "Test RMSE:", test_rmse)

    # save example predictions
    np.save("saved_models/preds.npy", preds[:5])
    np.save("saved_models/trues.npy", trues[:5])
    print("Saved example predictions to saved_models/")


if __name__ == "__main__":
    main()
