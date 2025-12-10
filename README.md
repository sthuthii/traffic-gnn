
Traffic Prediction using STGCN (Spatio-Temporal Graph Convolutional Networks)

This project implements a from-scratch STGCN model for traffic speed forecasting.
It supports both synthetic datasets and the real METR-LA dataset.

What STGCN Does:
- Learns temporal traffic patterns (how speed changes through time)
- Learns spatial patterns (how connected road sensors influence each other)
- Combines Graph Convolution + Temporal Convolution

Project Structure:
traffic-gnn/
│
├── src/
│   ├── gnn_layers.py        # Graph Convolution Layer
│   ├── model.py             # STGCN Model
│   ├── train.py             # Training loop
│   └──data_utils.py             # Windowing + metrics
│
└── README.md

Features:
- Full STGCN (from scratch, easy to understand)
- METR-LA dataset download support (yet to be implemented)
- CPU‑friendly training
- Prediction samples saved as .npy files

Installation:
    git clone https://github.com/your-username/traffic-gnn.git
    cd traffic-gnn
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

Run training:
    python -m src.train
    python -m src.train --use_metr_la

Example Output:
Epoch 1/12  Train MAE: 0.5081  Val MAE: 0.3341
Test MAE: 0.3151  Test RMSE: 0.3946

Model Architecture:
STGCN(
  TemporalConv1
  GraphConv1
  TemporalConv2
  OutputConv
)

Future Improvements:
- ChebNet GCN
- DCRNN
- Visualizations
- FastAPI deployment


