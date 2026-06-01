# QuantLSTM: Production-Grade Market Engine

A full-stack, real-time quantitative inference engine designed to analyze financial market volatility.

## Architecture
* **Backend:** FastAPI (Python) for asynchronous market data ingestion.
* **Inference:** Multivariate LSTM Neural Network using PyTorch, trained on stationary log-returns rather than volatile raw prices.
* **Frontend:** Interactive dashboard built with Chart.js and TailwindCSS for real-time visualization.

## Technical Highlights
* **Stationary Processing:** Uses logarithmic returns and multivariate features (Volatility, SMA, Volume Ratio) to avoid the "Lag Illusion."
* **Dynamic Time-Travel:** Automatically computes 4-year historical training windows relative to the current system date.
* **Statistical Validation:** RMSE metrics are calculated on out-of-sample data to ensure the model survives market chaos.