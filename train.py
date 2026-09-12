import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def make_data(n=24*90, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    daily = 120 * np.sin(2*np.pi*t/24 - 0.8)
    weekly = 45 * np.sin(2*np.pi*t/(24*7))
    trend = 0.02 * t
    noise = rng.normal(0, 12, n)
    demand = 500 + daily + weekly + trend + noise
    return pd.DataFrame({"demand": demand})

df = make_data()

for lag in [1, 24, 168]:
    df[f"lag_{lag}"] = df["demand"].shift(lag)

df["hour"] = np.arange(len(df)) % 24
df["day_of_week"] = (np.arange(len(df)) // 24) % 7
df = df.dropna().reset_index(drop=True)

features = ["lag_1", "lag_24", "lag_168", "hour", "day_of_week"]
split = int(len(df) * 0.8)

train, test = df.iloc[:split], df.iloc[split:]
model = RandomForestRegressor(n_estimators=200, random_state=7, n_jobs=-1)
model.fit(train[features], train["demand"])

pred = model.predict(test[features])
mae = mean_absolute_error(test["demand"], pred)
rmse = np.sqrt(mean_squared_error(test["demand"], pred))

print(f"Chronological test rows: {len(test)}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
