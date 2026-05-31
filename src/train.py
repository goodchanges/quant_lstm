import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# 1. Ensure Strict Mathematical Reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("="*60)
print("STARTING QUANTITATIVE LSTM ENGINE PURE PIPELINE")
print("="*60)

# 2. Ingest Real Financial Market Volatility Data
TICKER = "AAPL"  # Can be changed to NVDA, TSLA, or BTC-USD
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"

print(f"[1/6] Downloading historical data for ticker: {TICKER}...")
df = yf.download(TICKER, start=START_DATE, end=END_DATE)

# Extract daily closing price
raw_prices = df['Close'].values.reshape(-1, 1)
print(f"[2/6] Successfully ingested {len(raw_prices)} trading days.")

# 3. Pre-Processing and Normalization
# Scaling is mathematically required to prevent exploding/vanishing gradients in Recurrent Layers
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(raw_prices)

# Generate sliding temporal windows
def rolling_window_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)

LOOKBACK_WINDOW = 50  # Model looks at the past 50 trading days to predict the next day
X, y = rolling_window_sequences(scaled_prices, LOOKBACK_WINDOW)

# 80/20 Chronological Train-Test Split (No shuffling allowed in time-series data)
split_idx = int(len(X) * 0.8)
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# 4. Deep Deep Recurrent Neural Network Architecture
class ProductionQuantLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1):
        super(ProductionQuantLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=0.2  # Regularization to combat overfitting
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Forward pass through recurrent layers
        lstm_out, _ = self.lstm(x)
        # Decode the hidden state of the absolute last time-step in the sequence
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

model = ProductionQuantLSTM()
criterion = nn.MSELoss()  # Mean Squared Error Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Iterative Optimization Loop (Model Training)
EPOCHS = 80
print(f"[3/6] Initializing neural network optimization over {EPOCHS} epochs...")
model.train()

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)
    loss.backward()  # Backpropagation / Gradient Computation
    optimizer.step()  # Weights Optimization Update Step
    
    if (epoch + 1) % 10 == 0:
        print(f"    Epoch [{epoch+1:02d}/{EPOCHS:02d}] | Mean Squared Error Loss: {loss.item():.6f}")

# 6. Evaluation and Static Artifact Extraction
print("[4/6] Training completed. Entering evaluation mode...")
model.eval()
with torch.no_grad():
    test_predictions = model(X_test).numpy()

# Reverse scaling back to actual financial dollar denominations
unscaled_test_preds = scaler.inverse_transform(test_predictions).flatten().tolist()
unscaled_actual_prices = scaler.inverse_transform(scaled_prices).flatten().tolist()

# Calculate descriptive final testing metrics for validation display
final_mse = np.mean((scaler.inverse_transform(y_test.numpy()) - scaler.inverse_transform(test_predictions))**2)
rmse_val = float(np.sqrt(final_mse))

# Structure clean JSON payload for static web visualization
output_data = {
    "ticker": TICKER,
    "rmse": round(rmse_val, 4),
    "historical": unscaled_actual_prices[:split_idx + LOOKBACK_WINDOW],
    "predictions": [None] * (split_idx + LOOKBACK_WINDOW) + unscaled_test_preds
}

print("[5/6] Compiling predictions and formatting JSON serialization structures...")
os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "predictions.json")

with open(output_path, "w") as file:
    json.dump(output_data, file, indent=2)

print(f"[6/6] SUCCESS! Quantitative data artifacts written to: {output_path}")
print("="*60)