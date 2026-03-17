import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ================== 参数设置 ==================
ETF_CODE = '510300'          # 可换成你感兴趣的ETF
START_DATE = '2020-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')

# 聚类特征列（可以根据需要增删）
feature_cols = ['adx', 'bb_width', 'MA5', 'MA20', 'MA60', 'MA120', 'volume_ratio', 'close']
N_CLUSTERS = 5                # 聚类数量，可调整

# ================== 数据获取 ==================
def fetch_data(code, start, end):
    full_code = 'sh' + code if code.startswith('5') else 'sz' + code
    df = ak.fund_etf_hist_sina(symbol=full_code)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df = df[(df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))]
    df.set_index('date', inplace=True)
    df = df.astype(float)
    return df[['close', 'high', 'low', 'volume']]

df = fetch_data(ETF_CODE, START_DATE, END_DATE)
print(f"数据范围: {df.index.min()} 至 {df.index.max()}\n")

# ================== 计算技术指标 ==================
# 均线
ma_periods = [5, 20, 60, 120]
for p in ma_periods:
    df[f'MA{p}'] = df['close'].rolling(p).mean()

# 成交量均线
df['vol_ma20'] = df['volume'].rolling(20).mean()
df['vol_ma3'] = df['volume'].rolling(3).mean()
df['volume_ratio'] = df['vol_ma3'] / df['vol_ma20']

# 计算ADX（简化版）
def compute_adx(df, period=14):
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.abs(df['high'] - df['close'].shift(1)),
                          np.abs(df['low'] - df['close'].shift(1)))
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['+dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['atr'] = df['tr'].rolling(period).mean()
    df['+di'] = 100 * df['+dm'].rolling(period).mean() / df['atr']
    df['-di'] = 100 * df['-dm'].rolling(period).mean() / df['atr']
    df['dx'] = 100 * np.abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])
    df['adx'] = df['dx'].rolling(period).mean()
    return df

df = compute_adx(df, 14)

# 布林带
bb_period = 20
bb_std = 2
df['bb_mid'] = df['close'].rolling(bb_period).mean()
df['bb_std'] = df['close'].rolling(bb_period).std()
df['bb_upper'] = df['bb_mid'] + bb_std * df['bb_std']
df['bb_lower'] = df['bb_mid'] - bb_std * df['bb_std']
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

# 丢弃前期的NaN值，确保特征列都有值
df = df.dropna(subset=feature_cols)

# ================== 聚类分析 ==================
# 提取特征数据
X = df[feature_cols].copy()

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means聚类
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# 将聚类标签添加回DataFrame
df['cluster'] = cluster_labels

# ================== 结果分析 ==================
# 1. 聚类分布
print("聚类分布（每个类别的天数）：")
print(df['cluster'].value_counts().sort_index())

# 2. 聚类中心（原始尺度）
centers = scaler.inverse_transform(kmeans.cluster_centers_)
center_df = pd.DataFrame(centers, columns=feature_cols)
print("\n聚类中心（各特征平均值）：")
print(center_df.round(2))

# 3. 每个聚类的特征均值（用原始数据计算，更准确）
print("\n各聚类特征均值（用原始数据）：")
for c in sorted(df['cluster'].unique()):
    print(f"\n聚类 {c}:")
    print(df[df['cluster'] == c][feature_cols].mean().round(2))

# ================== 可视化 ==================
# 绘制聚类结果随时间的变化（用不同颜色标记）
plt.figure(figsize=(14, 6))
# 用收盘价作为背景，颜色代表聚类
sc = plt.scatter(df.index, df['close'], c=df['cluster'], cmap='tab10', s=10, alpha=0.7)
plt.colorbar(sc, label='Cluster')
plt.title(f'{ETF_CODE} 市场状态聚类结果 (K={N_CLUSTERS})')
plt.xlabel('日期')
plt.ylabel('收盘价')
plt.grid(True)
plt.tight_layout()
plt.show()

# 可选：绘制特征分布雷达图（使用聚类中心）
from math import pi
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
# 对特征中心进行归一化（0-1）以便在同一图上比较
normalized_centers = (center_df - center_df.min()) / (center_df.max() - center_df.min())
categories = feature_cols
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # 闭合
for i in range(N_CLUSTERS):
    values = normalized_centers.iloc[i].values.tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i}')
    ax.fill(angles, values, alpha=0.1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.title('聚类特征对比')
plt.show()