import pandas as pd
import numpy as np
import akshare as ak
import time
from datetime import datetime, timedelta
from scipy import stats
from typing import Optional

# -------------------- 数据获取 --------------------
def fetch_etf_daily(code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    获取单个 ETF 的日线数据（收盘价和成交量）
    返回 DataFrame，索引为日期，列有 'close', 'volume'
    """
    code_str = str(code).strip()
    if code_str.startswith(('5', '51', '56')):
        full_code = 'sh' + code_str
    elif code_str.startswith(('1', '15', '16', '18')):
        full_code = 'sz' + code_str
    else:
        full_code = 'sh' + code_str

    try:
        df = ak.fund_etf_hist_sina(symbol=full_code)
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
        df.set_index('date', inplace=True)
        result = pd.DataFrame({
            'close': df['close'].astype(float),
            'volume': df['volume'].astype(float)
        })
        return result
    except Exception as e:
        print(f"获取 {code} 失败: {e}")
        return None

# -------------------- 成交量分析核心函数 --------------------
def analyze_volume(
    price_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    vol_ratio_threshold: float = 0.2,
    vol_spike_threshold: float = 1.5,
    corr_threshold: float = 0.3
) -> pd.DataFrame:
    """
    对给定的ETF池进行成交量分析，返回最新一期的成交量状态和指标。
    """
    # 确保数据对齐
    common_dates = price_df.index.intersection(volume_df.index)
    price_df = price_df.loc[common_dates]
    volume_df = volume_df.loc[common_dates]

    # 计算均量
    vol_ma_short = volume_df.rolling(short_window).mean()
    vol_ma_long = volume_df.rolling(long_window).mean()

    # 量比趋势
    vol_ratio = vol_ma_short / vol_ma_long - 1.0

    # 量能斜率（线性回归）
    def calc_slope(series):
        if len(series) < long_window:
            return np.nan
        x = np.arange(len(series))
        y = series.values
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope

    vol_slope = volume_df.rolling(long_window).apply(lambda s: calc_slope(s), raw=False)

    # 价格趋势斜率
    price_slope = price_df.rolling(long_window).apply(lambda s: calc_slope(s), raw=False)

    # 量价相关系数（逐列计算）
    corr_dict = {}
    for code in price_df.columns:
        price_vals = price_df[code].iloc[-long_window:]
        vol_vals = volume_df[code].iloc[-long_window:]
        if len(price_vals) == long_window:
            corr, _ = stats.spearmanr(price_vals, vol_vals)
            corr_dict[code] = corr
        else:
            corr_dict[code] = np.nan
    corr_series = pd.Series(corr_dict, name='量价相关系数')

    # 放量天数占比
    vol_spike_days = (volume_df > vol_ma_long).rolling(long_window).sum() / long_window

    # 连续放量天数（逐列计算，避免 rolling apply 的歧义）
    consecutive_days = pd.DataFrame(index=volume_df.index, columns=volume_df.columns)
    for code in volume_df.columns:
        vol_series = volume_df[code]
        ma_series = vol_ma_long[code]
        days = []
        for i in range(len(vol_series)):
            if i < long_window - 1:
                days.append(0)
                continue
            window = vol_series.iloc[i-long_window+1:i+1]
            threshold = ma_series.iloc[i]
            count = 0
            for val in window[::-1]:
                if val > threshold:
                    count += 1
                else:
                    break
            days.append(count)
        consecutive_days[code] = days
    consecutive_days.index = volume_df.index

    # 价格位置（相对于 long_window 高低点）
    high_20 = price_df.rolling(long_window).max()
    low_20 = price_df.rolling(long_window).min()
    price_position = (price_df - low_20) / (high_20 - low_20)

    # 最新一期数据
    latest_idx = price_df.index[-1]

    # 构建结果
    results = pd.DataFrame(index=price_df.columns)
    results['最新日期'] = latest_idx
    results['最新收盘价'] = price_df.loc[latest_idx].values
    results['短期均量'] = vol_ma_short.loc[latest_idx].values
    results['长期均量'] = vol_ma_long.loc[latest_idx].values
    results['量比趋势'] = vol_ratio.loc[latest_idx].values
    results['量能斜率'] = vol_slope.loc[latest_idx].values
    results['价格趋势斜率'] = price_slope.loc[latest_idx].values
    results['量价相关系数'] = corr_series.values
    results['放量天数占比'] = vol_spike_days.loc[latest_idx].values
    results['连续放量天数'] = consecutive_days.loc[latest_idx].values
    results['价格位置'] = price_position.loc[latest_idx].values

    # 生成成交量状态标签
    def get_volume_label(row):
        labels = []
        # 量价齐升
        if row['量比趋势'] > vol_ratio_threshold and row['价格趋势斜率'] > 0 and row['量价相关系数'] > corr_threshold:
            labels.append("量价齐升")
        # 缩量上涨
        if row['价格趋势斜率'] > 0 and row['量比趋势'] < -vol_ratio_threshold and row['量价相关系数'] < -corr_threshold:
            labels.append("缩量上涨(背离)")
        # 放量滞涨
        if row['量比趋势'] > vol_ratio_threshold and row['价格趋势斜率'] < 0.1 and row['量价相关系数'] < -corr_threshold:
            labels.append("放量滞涨")
        # 底部堆量
        if row['价格位置'] < 0.2 and row['量比趋势'] > vol_ratio_threshold and row['短期均量'] > row['长期均量'] * vol_spike_threshold:
            labels.append("底部堆量")
        # 高位爆量
        if row['价格位置'] > 0.8 and row['量比趋势'] > vol_ratio_threshold and row['短期均量'] > row['长期均量'] * vol_spike_threshold:
            labels.append("高位爆量")
        # 连续放量
        if row['连续放量天数'] >= 3:
            labels.append(f"连续放量{int(row['连续放量天数'])}天")
        # 量价背离
        if row['价格趋势斜率'] > 0 and row['量价相关系数'] < -corr_threshold:
            labels.append("量价背离")
        if row['价格趋势斜率'] < 0 and row['量价相关系数'] > corr_threshold:
            labels.append("量价底背离")

        if not labels:
            return "正常"
        return " | ".join(labels)

    results['成交量状态'] = results.apply(get_volume_label, axis=1)

    return results

# -------------------- 从文件运行的主函数 --------------------
def analyze_volume_from_file(
    input_file: str = 'data/etf_top1_by_category.csv',
    start_date: str = None,
    end_date: str = None,
    short_window: int = 5,
    long_window: int = 20,
    output_file: str = 'data/etf_volume_analysis.csv'
) -> pd.DataFrame:
    """
    从 CSV 文件读取 ETF 列表，获取日线数据，进行成交量分析，保存结果并返回。
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')

    # 读取 ETF 列表
    df_etf = pd.read_csv(input_file)
    df_etf['基金代码'] = df_etf['基金代码'].astype(str).str.strip()
    etf_codes = df_etf['基金代码'].tolist()
    print(f"共读取 {len(etf_codes)} 只 ETF")

    # 获取数据
    price_dict = {}
    volume_dict = {}
    for code in etf_codes:
        df = fetch_etf_daily(code, start_date, end_date)
        if df is not None and len(df) > long_window:
            price_dict[code] = df['close']
            volume_dict[code] = df['volume']
        time.sleep(0.5)

    if not price_dict:
        raise ValueError("没有获取到任何有效 ETF 数据")

    price_df = pd.DataFrame(price_dict).sort_index()
    price_df = price_df.asfreq('D').ffill().dropna()
    price_df = price_df.dropna(axis=1, how='all')

    volume_df = pd.DataFrame(volume_dict).sort_index()
    volume_df = volume_df.asfreq('D').ffill().fillna(0)
    volume_df = volume_df.reindex(columns=price_df.columns, fill_value=0)

    print(f"数据范围: {price_df.index.min()} 至 {price_df.index.max()}, 共 {len(price_df)} 天")

    # 执行成交量分析
    vol_result = analyze_volume(
        price_df,
        volume_df,
        short_window=short_window,
        long_window=long_window,
        vol_ratio_threshold=0.2,
        vol_spike_threshold=1.5,
        corr_threshold=0.3
    )

    # 合并原始信息（基金简称、小类）
    vol_result = vol_result.merge(df_etf[['基金代码', '基金简称', '小类']], left_index=True, right_on='基金代码', how='left')
    vol_result.set_index('基金代码', inplace=True)

    # 重新排序列
    cols = ['基金简称', '小类', '最新日期', '最新收盘价', '成交量状态', '短期均量', '长期均量',
            '量比趋势', '量能斜率', '价格趋势斜率', '量价相关系数', '放量天数占比', '连续放量天数', '价格位置']
    vol_result = vol_result[cols]

    # 保存
    vol_result.to_csv(output_file, encoding='utf-8-sig')
    print(f"成交量分析结果已保存至 {output_file}")

    return vol_result

# -------------------- 使用示例 --------------------
if __name__ == "__main__":
    result = analyze_volume_from_file(
        input_file='data/etf_top1_by_category.csv',
        start_date='2025-01-01',
        end_date='2026-03-12',
        short_window=5,
        long_window=20,
        output_file='data/etf_volume_analysis.csv'
    )
    print("\n成交量异常ETF示例：")
    # 筛选出非“正常”状态的ETF
    abnormal = result[result['成交量状态'] != '正常']
    print(abnormal[['基金简称', '小类', '成交量状态']].head(10))