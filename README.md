# QuantLSTM (Stationary-V3): Multivariate Log Return Inference Engine

QuantLSTM is a high-performance, locally-optimized deep learning pipeline designed to analyze historical asset pricing patterns and infer future price trajectories. Built using **PyTorch** and **FastAPI**, the application processes live financial market data, transforms raw pricing vectors into stationary statistical features, and executes real-time LSTM neural network inference through an interactive glassmorphic web dashboard.

---

## 🚀 Architectural Overview

The system is split into an asynchronous decoupled architecture to maximize computational efficiency and bypass cloud-hosted hardware bottlenecks:

1. **Frontend Client:** A vibrant, dark-themed dashboard built with HTML5, Tailwind CSS, and Chart.js that handles user input and visualizes multi-step predictive outputs.
2. **Backend Engine:** A lightweight FastAPI server acting as the orchestration layer for data ingestion, mathematical scaling, and model execution.
3. **Deep Learning Pipeline:** An unthrottled PyTorch Long Short-Term Memory (LSTM) network optimizing feature processing locally using native CPU multi-threading.

---

## 🛠️ Deep Tech Stack

- **Core Intelligence:** PyTorch (`torch`, `torch.nn`)
- **Data Infrastructure:** Pandas, NumPy, Scikit-Learn
- **API Orchestration:** FastAPI, Uvicorn (ASGI web server)
- **Data Ingestion:** `yfinance` API + Stooq Failover Pipeline
- **Visualization Engine:** HTML5, Tailwind CSS, Chart.js

---

## ⚡ Key Engineering Features

### 1. Dual-Source Redundant Data Failover
To survive aggressive cloud-based Web Application Firewalls (WAF) and datacenter IP rate-limiting (`429 Too Many Requests`), the ingestion engine features an automated try/except fallback architecture. If the primary Yahoo Finance API blocks data traffic, the system instantly catches the exception and hot-swaps the ingestion routing to the European Stooq Financial Database in real-time without interrupting the user experience.

### 2. Stationary Feature Engineering
Feeding raw stock prices directly into neural networks often fails due to market non-stationarity. This framework mathematically transforms raw pricing curves into four highly stationary features before execution:
- **Log Returns:** $np.log(P_t / P_{t-1})$ to stabilize variance over time.
- **Volatility Scaling:** High-low range spreads standardized by closing price.
- **Moving Average Distance (Dist_SMA20):** Quantifying asset deviations from structural baselines.
- **Volume Ratio:** Historical liquidity spikes relative to rolling averages.

### 3. Local-First Hardware Optimization
Deep learning models carry massive memory payloads (PyTorch packages exceed 2GB). By optimizing this system as a high-performance local engine rather than running inside throttled 512MB cloud free-tiers, the model utilizes native CPU multi-core processing to run deep tensor math and complete training phases in under 5 seconds.

---

## 📁 Repository Structure

```text
quant_lstm/
│
├── src/
│   ├── api.py           # FastAPI server orchestration, failover ingestion, and inference loop
│   └── train.py         # PyTorch LSTM neural network architecture definition
│
├── index.html           # Glassmorphic user interface & Chart.js engine
├── requirements.txt     # Python production dependencies 
└── README.md            # Comprehensive project documentation