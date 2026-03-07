import akshare as ak
from akquant import Strategy, run_backtest
import get_data

# 1. 准备数据 (这里利用 AKShare 来获取数据，需要通过 pip install akshare 来进行安装)
df = get_data.get_etf_data(symbol="sh510050")
print(df)








