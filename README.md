# Power Demand Forecasting

A baseline machine-learning project connecting electrical-engineering knowledge with time-series forecasting.

## Objective

Forecast the next period of electricity demand from historical demand and calendar/time features.

This repository starts with a reproducible synthetic dataset so the pipeline can be tested without depending on a particular external data provider. A later version should evaluate the same pipeline on a public utility/grid dataset.

## Pipeline

1. Generate hourly demand with trend, daily seasonality, weekly seasonality and noise.
2. Build lag and calendar features.
3. Split chronologically (no random shuffle).
4. Train a baseline regression model.
5. Compare predictions with held-out observations.

## Run

```bash
pip install -r requirements.txt
python src/train.py
```

## Research caution

Synthetic-data performance should not be presented as real grid forecasting performance. Real evaluation requires a public dataset, a documented train/test period, appropriate baselines, and metrics such as MAE/RMSE.
