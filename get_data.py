import akshare as ak
from datetime import datetime, timedelta
import pandas as pd

import numpy as np

def get_stock_data(symbol, start_date, end_date = datetime.now().strftime("%Y%m%d"), adjust="qfq"):
    """
    获取股票数据
    :param symbol: 股票代码，例如 "sh600000"
    :param start_date: 起始日期，格式为 "YYYYMMDD"
    :param end_date: 结束日期，格式为 "YYYYMMDD"
    :return: 股票数据的 DataFrame

    获得的数据是正序的，最新日期在最后一行，最旧日期在第一行。
    """
    df = ak.stock_zh_a_daily(symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust)
    df['date'] = pd.to_datetime(df['date'])

     # 按日期升序排序（从早到晚）
    df = df.sort_values(by='date', ascending=True).reset_index(drop=True)

    # 计算涨跌幅：用收盘价计算（前一天的收盘价和当天的收盘价）
    df['涨跌幅百分比'] = df['close'].pct_change() * 100  # pct_change() 计算的是相邻两天的百分比变化
    
    # 删除第一行（最早的一天），因为涨跌幅为 NaN
    df = df.dropna(subset=['涨跌幅百分比']).reset_index(drop=True)

    return df

def get_etf_data(symbol):
    """
    获取基金ETF历史数据
    :param symbol: 基金ETF代码，例如 "sh510050"
    :return: 基金ETF历史数据的 DataFrame
    获得的数据是正序的，最新日期在最后一行，最旧日期在第一行。
    """

    df = ak.fund_etf_hist_sina(symbol=symbol)
    df['date'] = pd.to_datetime(df['date'])

    # 按日期排序
    df = df.sort_values(by='date', ascending=True).reset_index(drop=True)

    # 计算涨跌幅：用收盘价计算（前一天的收盘价和当天的收盘价）
    df['涨跌幅百分比'] = df['close'].pct_change() * 100  # pct_change() 计算的是相邻两天的百分比变化
    
    # 删除第一行（最早的一天），因为涨跌幅为 NaN
    df = df.dropna(subset=['涨跌幅百分比']).reset_index(drop=True)

    return df

def get_all_etf_code():
    """
    获取所有ETF基金的代码
    :return: ETF基金代码的 DataFrame
    """
    df = ak.fund_etf_category_sina(symbol="ETF基金")
    return df


#股票换手成本线
def calc_turnover_cost_line(df, date_col='date', price_col='close', turnover_col='turnover', threshold=1.0):
    """
    计算每日的换手成本线：从当天向前累加换手率，直到总和≥threshold（默认1.0，代表100%），
    用这些天的收盘价计算均值作为当天的成本线。

    参数：
        df : DataFrame，必须包含日期列、收盘价列和换手率列。
        date_col : 日期列名，默认为'date'。
        price_col : 收盘价列名，默认为'close'。
        turnover_col : 换手率列名，默认为'turnover'。
        threshold : 累加阈值，小数形式（例如1.0表示100%），如果换手率是百分比数值，请相应调整。

    返回：
        DataFrame，新增一列 'turnover_cost_line'，包含每日的成本线（若数据不足则为NaN）。
    """
    df = df.copy()
    
    # 确保日期列为datetime并升序排列
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    else:
        # 如果日期是索引
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().reset_index()
        df.rename(columns={'index': date_col}, inplace=True)

    n = len(df)
    cost_line = np.full(n, np.nan)

    for i in range(n):
        turnover_sum = 0
        j = i
        # 从当天向前累加换手率，直到总和≥threshold 或到达数据起点
        while j >= 0 and turnover_sum < threshold:
            turnover_sum += df.loc[j, turnover_col]
            j -= 1
        if j < i:  # 至少累加了一天
            start = j + 1
            end = i
            cost_line[i] = df.loc[start:end, price_col].mean()

    df['turnover_cost_line'] = cost_line
    return df




