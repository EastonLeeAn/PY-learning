# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     apple
   Description :
   Author :       lizhenhui
   date：          2024/3/15
-------------------------------------------------
   Change Activity:
                   2024/3/15:
-------------------------------------------------
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 导入苹果股票数据
df = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=True)

# 绘制苹果股票价格走势图
plt.plot(df['Close'], label='Close Price')
plt.legend()
plt.show()

# 使用移动平均线预测苹果股票走势
mavg_10 = df['Close'].rolling(10).mean()
mavg_50 = df['Close'].rolling(50).mean()

# 绘制移动平均线
plt.plot(df['Close'], label='Close Price')
plt.plot(mavg_10, label='10-day Moving Average')
plt.plot(mavg_50, label='50-day Moving Average')
plt.legend()
plt.show()

# 使用相对强弱指数 (RSI) 预测苹果股票走势
rsi = pd.Series(np.nan, index=df.index, name='RSI')

for i in range(len(df)):
    close_prices = df['Close'].iloc[i-14:i]
    up_moves = close_prices.diff().clip(lower=0)
    down_moves = -close_prices.diff().clip(upper=0)
    rsi[i] = 100 * up_moves.mean() / (up_moves.mean() + down_moves.mean())

# 绘制 RSI 指标
plt.plot(rsi, label='RSI')
plt.legend()
plt.show()

# 使用机器学习模型预测苹果股票走势
from sklearn.linear_model import LinearRegression

# 准备训练数据
X = df[['Close', 'Volume']]
y = df['Close'].shift(-1)

# 训练模型
model = LinearRegression()
model.fit(X, y)

# 预测未来苹果股票价格
future_prices = model.predict(X)

# 绘制预测结果
plt.plot(df['Close'], label='Actual Price')
plt.plot(future_prices, label='Predicted Price')
plt.legend()
plt.show()
