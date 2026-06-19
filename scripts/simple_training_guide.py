#!/usr/bin/env python3
"""
QuantumFlow AI - ODDIY BOSQICHMA-BOSQICH QO'LLANMA
Model o'qitish uchun eng oddiy yo'l
"""

# ============================================================
# 1-QADAM: KERAKLI KUTUBXONALARNI O'RNATISH
# ============================================================
"""
Terminalda quyidagi buyruqlarni bajaring:

pip install yfinance pandas numpy torch gymnasium scikit-learn
pip install ccxt  # (Crypto uchun)
pip install MetaTrader5  # (MT5 bilan ishlash uchun)
"""

# ============================================================
# 2-QADAM: MA'LUMOTLARNI YUKLASH (Yahoo Finance - BEPUL)
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np

# Forex ma'lumotlarini yuklash (EUR/USD)
print("📊 EUR/USD ma'lumotlarini yuklash...")
df = yf.download('EURUSD=X', start='2020-01-01', end='2024-12-31', interval='1h')
print(f"✅ Yuklandi: {len(df)} qator")
print(df.head())

# Saqlash
df.to_csv('eurusd_1h.csv')
print("💾 Saqlandi: eurusd_1h.csv")

# ============================================================
# 3-QADAM: GOLD (XAU/USD) MA'LUMOTLARINI YUKLASH
# ============================================================

print("
📊 Gold ma'lumotlarini yuklash...")
df_gold = yf.download('GC=F', start='2020-01-01', end='2024-12-31', interval='1h')
print(f"✅ Yuklandi: {len(df_gold)} qator")
df_gold.to_csv('gold_1h.csv')
print("💾 Saqlandi: gold_1h.csv")

# ============================================================
# 4-QADAM: CRYPTO (Bitcoin) MA'LUMOTLARINI YUKLASH
# ============================================================

print("
📊 Bitcoin ma'lumotlarini yuklash...")
df_btc = yf.download('BTC-USD', start='2020-01-01', end='2024-12-31', interval='1d')
print(f"✅ Yuklandi: {len(df_btc)} qator")
df_btc.to_csv('btc_1d.csv')
print("💾 Saqlandi: btc_1d.csv")

# ============================================================
# 5-QADAM: MA'LUMOTLARNI TAYYORLASH (Feature Engineering)
# ============================================================

print("
🔧 Feature engineering...")

# OHLCV formatiga o'tkazish
df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
df = df[['open', 'high', 'low', 'close', 'volume']]

# Technical indicators
# SMA
df['sma_20'] = df['close'].rolling(20).mean()
df['sma_50'] = df['close'].rolling(50).mean()

# RSI
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# ATR
high_low = df['high'] - df['low']
high_close = abs(df['high'] - df['close'].shift())
low_close = abs(df['low'] - df['close'].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()

# Returns
df['returns'] = df['close'].pct_change()
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

# NaN larni tozalash
df = df.dropna()

print(f"✅ Features tayyor: {df.shape}")

# ============================================================
# 6-QADAM: MODEL O'QITISH (Oddiy PPO misoli)
# ============================================================

print("
🧠 Model o'qitish...")

import torch
import torch.nn as nn
import numpy as np

# Data preparation
features = df[['sma_20', 'sma_50', 'rsi', 'atr', 'returns']].values.astype(np.float32)
returns = df['returns'].fillna(0).values.astype(np.float32)

# Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Simple neural network
class SimpleTradingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # [flat, long, short]
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

# Initialize
model = SimpleTradingNet(features.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop (simplified)
window = 20
epochs = 100

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for i in range(window, len(features) - 1):
        # Input: window of features
        x = torch.FloatTensor(features[i-window:i]).unsqueeze(0)

        # Target: next return direction
        next_return = returns[i+1]
        if next_return > 0.001:
            target = torch.LongTensor([1])  # Long
        elif next_return < -0.001:
            target = torch.LongTensor([2])  # Short
        else:
            target = torch.LongTensor([0])  # Flat

        # Forward
        output = model(x)
        loss = criterion(output, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy
        pred = output.argmax(dim=-1)
        correct += (pred == target).sum().item()
        total += 1

    if epoch % 10 == 0:
        acc = correct / total * 100
        print(f"Epoch {epoch}: Loss={total_loss:.4f}, Accuracy={acc:.1f}%")

# Save model
torch.save(model.state_dict(), 'simple_model.pt')
print("💾 Model saqlandi: simple_model.pt")

# ============================================================
# 7-QADAM: BOSQICHMA-BOSQICH QUANTUMFLOW MODELINI O'QITISH
# ============================================================

print("
" + "="*60)
print("🚀 QUANTUMFLOW MODELINI O'QITISH")
print("="*60)

"""
1. Ma'lumotlarni yuklang:
   python scripts/train_pipeline.py --download --symbols EURUSD=X XAUUSD=X --source yahoo

2. Modelni o'qiting:
   python scripts/train_pipeline.py --train --symbols EURUSD=X --steps 500000

3. Yoki to'liq pipeline:
   python scripts/train_pipeline.py --download --train --symbols EURUSD=X --ensemble --steps 1000000
"""

print("
✅ Qo'llanma yakunlandi!")
print("📚 Keyingi qadam: scripts/train_pipeline.py ni ishga tushiring")
