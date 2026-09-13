import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics
import sklearn.preprocessing
import matplotlib.pyplot as plt

np.random.seed(42)

n = 1000
dates = pd.date_range(start="2023-01-01", periods=n, freq="B")

price_changes = np.random.normal(loc=0.0002, scale=0.015, size=n)
closing_prices = 100 * np.cumprod(1 + price_changes)

df = pd.DataFrame(index=dates)
df['Close'] = closing_prices

df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, 0.005, n))
df.loc[df.index[0], 'Open'] = 100

df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.003, n)))
df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.003, n)))
df['Volume'] = np.random.randint(1000000, 20000000, size=n)

df['Return'] = df['Close'].pct_change()
df['SMA_5'] = df['Close'].rolling(5).mean()
df['SMA_20'] = df['Close'].rolling(20).mean()
df['Ratio'] = df['SMA_5'] / df['SMA_20']
df['Volatility'] = df['Return'].rolling(10).std()

df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

df = df.dropna()

features = ['Return', 'Ratio', 'Volatility']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, shuffle=False)

scaler = sklearn.preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

majority_class = y_train.mode()[0]
baseline_preds = [majority_class] * len(y_test)
print("Baseline Accuracy:", sklearn.metrics.accuracy_score(y_test, baseline_preds))

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

preds = rf.predict(X_test_scaled)
print("Model Accuracy:", sklearn.metrics.accuracy_score(y_test, preds))
print("\nClassification Report:")
print(sklearn.metrics.classification_report(y_test, preds))

plt.figure(figsize=(10, 4))
plt.plot(y_test.index[:30], y_test.values[:30], label='Actual', marker='o')
plt.plot(y_test.index[:30], preds[:30], label='Predicted', marker='x', linestyle='--')
plt.legend()
plt.title('Actual vs Predicted Direction')
plt.tight_layout()
plt.show()