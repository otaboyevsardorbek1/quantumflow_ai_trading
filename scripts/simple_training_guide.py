#!/usr/bin/env python3
"""
QuantumFlow AI - ODDIY BOSQICHMA-BOSQICH QO'LLANMA
Model o'qitish uchun eng oddiy yo'l
"""

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ============================================================
# 2-QADAM: MA'LUMOTLARNI YUKLASH (Yahoo Finance - BEPUL)
# ============================================================

print("📊 EUR/USD ma'lumotlarini yuklash (oxirgi 730 kun)...")
df_forex = yf.download('EURUSD=X', period='730d', interval='1h', progress=False)
if df_forex.empty:
    print("⚠️ EUR/USD ma'lumotlari yuklanmadi.")
else:
    print(f"✅ Yuklandi: {len(df_forex)} qator")
    if isinstance(df_forex.columns, pd.MultiIndex):
        df_forex.columns = df_forex.columns.get_level_values(0)
    df_forex.to_csv('eurusd_1h.csv')
    print("💾 Saqlandi: eurusd_1h.csv")

print("📊 Gold ma'lumotlarini yuklash (oxirgi 730 kun)...")
df_gold = yf.download('GC=F', period='730d', interval='1h', progress=False)
if df_gold.empty:
    print("⚠️ Gold ma'lumotlari yuklanmadi.")
else:
    print(f"✅ Yuklandi: {len(df_gold)} qator")
    if isinstance(df_gold.columns, pd.MultiIndex):
        df_gold.columns = df_gold.columns.get_level_values(0)
    df_gold.to_csv('gold_1h.csv')
    print("💾 Saqlandi: gold_1h.csv")

print("📊 Bitcoin ma'lumotlarini yuklash...")
df_btc = yf.download('BTC-USD', start='2020-01-01', end='2024-12-31', interval='1d', progress=False)
if df_btc.empty:
    print("⚠️ Bitcoin ma'lumotlari yuklanmadi.")
else:
    print(f"✅ Yuklandi: {len(df_btc)} qator")
    if isinstance(df_btc.columns, pd.MultiIndex):
        df_btc.columns = df_btc.columns.get_level_values(0)
    df_btc.to_csv('btc_1d.csv')
    print("💾 Saqlandi: btc_1d.csv")

# ============================================================
# 5-QADAM: MA'LUMOTLARNI TAYYORLASH (BTC asosida)
# ============================================================

print("🔧 Feature engineering...")
df = df_btc.copy() if not df_btc.empty else None
if df is None:
    print("❌ Hech qanday ma'lumot yuklanmadi.")
    exit(1)

rename_map = {
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Adj Close': 'adj_close',
    'Volume': 'volume'
}
df = df.rename(columns=rename_map)
required = ['open', 'high', 'low', 'close', 'volume']
df = df[[c for c in required if c in df.columns]]

# Indicators
df['sma_20'] = df['close'].rolling(20).mean()
df['sma_50'] = df['close'].rolling(50).mean()

delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

high_low = df['high'] - df['low']
high_close = abs(df['high'] - df['close'].shift())
low_close = abs(df['low'] - df['close'].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()

df['returns'] = df['close'].pct_change()
df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

df = df.dropna()
if df.empty:
    print("❌ Ma'lumotlar yetarli emas.")
    exit(1)

print(f"✅ Features tayyor: {df.shape}")

# ============================================================
# 6-QADAM: MODEL O'QITISH (Oddiy feedforward)
# ============================================================

print("🧠 Model o'qitish...")

features = df[['sma_20', 'sma_50', 'rsi', 'atr', 'returns']].values.astype(np.float32)
returns = df['returns'].fillna(0).values.astype(np.float32)

scaler = StandardScaler()
features = scaler.fit_transform(features)

class SimpleTradingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

model = SimpleTradingNet(features.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

epochs = 100
total_samples = len(features) - 1
if total_samples <= 0:
    print("❌ Kamida 2 qator kerak.")
    exit(1)

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for i in range(len(features) - 1):
        x = torch.FloatTensor(features[i]).unsqueeze(0)
        next_return = returns[i+1]
        if next_return > 0.001:
            target = torch.LongTensor([1])
        elif next_return < -0.001:
            target = torch.LongTensor([2])
        else:
            target = torch.LongTensor([0])

        output = model(x)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=-1)
        correct += (pred == target).sum().item()
        total += 1

    if epoch % 10 == 0:
        acc = correct / total * 100
        print(f"Epoch {epoch}: Loss={total_loss:.4f}, Accuracy={acc:.1f}%")

torch.save(model.state_dict(), 'simple_model.pt')
print("💾 Model saqlandi: simple_model.pt")

print(" " + "="*60)
print("🚀 QUANTUMFLOW MODELINI O'QITISH")
print("="*60)
print("✅ Qo'llanma yakunlandi!")