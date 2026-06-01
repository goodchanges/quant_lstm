from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import torch
import requests
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import traceback

# Initialize Web Server
app = FastAPI(title="Quant LSTM Inference Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Neural Architecture (4 institutional features)
class QuantLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=1):
        super(QuantLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

@app.get("/predict/{ticker}")
def run_quant_engine(ticker: str):
    try:
        print(f"\n[{ticker.upper()}] Initializing Quantitative Pipeline...")
        
        # 1. BULLETPROOF DATA INGESTION (Bypasses yfinance download bug)
        # Check the live system clock
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        four_years_ago = today - timedelta(days=4*365)
        
        # Fetch data dynamically using today's exact date
        # === STEALTH MODE: Bypassing Yahoo's Cloud Firewall ===
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        })
        
        stock = yf.Ticker(ticker.upper(), session=session)
        df = stock.history(start=four_years_ago.strftime('%Y-%m-%d'), end=tomorrow.strftime('%Y-%m-%d'))
            
        if df.empty:
            raise HTTPException(status_code=404, detail="Ticker not found or delisted.")
            
        # 2. MIT-GRADE MATH: Engineer Stationary Features
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = (df['High'] - df['Low']) / df['Close']
        df['Dist_SMA20'] = (df['Close'].rolling(20).mean() / df['Close']) - 1
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        df = df.dropna()
        raw_close = df['Close'].values  
        
        # 3. Scaling
        features = df[['Log_Return', 'Volatility', 'Dist_SMA20', 'Volume_Ratio']].values
        target = df[['Log_Return']].values
        
        feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_features = feature_scaler.fit_transform(features)
        
        target_scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_targets = target_scaler.fit_transform(target)

        # 4. Sequence Generation
        LOOKBACK = 40
        X, y = [], []
        for i in range(len(scaled_features) - LOOKBACK):
            X.append(scaled_features[i:(i + LOOKBACK)])
            y.append(scaled_targets[i + LOOKBACK][0])
            
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

        # 5. Neural Network Training
        print(f"[{ticker.upper()}] Optimizing Neural Weights...")
        model = QuantLSTM(input_size=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(30): 
            optimizer.zero_grad()
            loss = criterion(model(X_tensor), y_tensor)
            loss.backward()
            optimizer.step()

        # 6. Inference & Mathematical Price Reconstruction
        print(f"[{ticker.upper()}] Reconstructing Log Returns to USD...")
        model.eval()
        with torch.no_grad():
            predicted_scaled_returns = model(X_tensor).numpy()
            
            last_sequence = scaled_features[-LOOKBACK:]
            last_tensor = torch.tensor(np.array([last_sequence]), dtype=torch.float32)
            tomorrow_scaled_return = model(last_tensor).numpy()

        predicted_returns = target_scaler.inverse_transform(predicted_scaled_returns).flatten()
        tomorrow_return = target_scaler.inverse_transform(tomorrow_scaled_return).flatten()[0]
        actual_returns = target_scaler.inverse_transform(scaled_targets).flatten()
        
        reconstructed_prices = []
        start_index = LOOKBACK
        
        for i in range(len(predicted_returns)):
            prev_actual_price = raw_close[start_index + i - 1]
            pred_price = prev_actual_price * np.exp(predicted_returns[i])
            reconstructed_prices.append(float(pred_price))
            
        last_actual_price = raw_close[-1]
        tomorrow_price = last_actual_price * np.exp(tomorrow_return)
        reconstructed_prices.append(float(tomorrow_price))

        # Slice off the 40-day warm-up period so the arrays are the exact same size
        aligned_actual_returns = actual_returns[LOOKBACK:]
        rmse_returns = float(np.sqrt(np.mean((aligned_actual_returns - predicted_returns)**2)))
        print(f"[{ticker.upper()}] Pipeline Complete. Transmitting Payload.")

        # === NEW: Extract calendar dates from the dataframe ===
        # Formats the index into 'YYYY-MM-DD' strings
        calendar_dates = df.index.strftime('%Y-%m-%d').tolist()
        # ======================================================

        return {
            "ticker": ticker.upper(),
            "rmse": round(rmse_returns, 5),
            "dates": calendar_dates,         # Send dates to the website
            "historical": raw_close.tolist(),
            "predictions": [None] * LOOKBACK + reconstructed_prices
        }

    except Exception as e:
        print("\n!!! SERVER CRASH DETECTED !!!")
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=str(e))