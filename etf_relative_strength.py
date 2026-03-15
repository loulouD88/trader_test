import pandas as pd
import numpy as np
import akshare as ak
import time
from datetime import datetime, timedelta
from typing import List, Optional

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

def fetch_benchmark_daily(code: str = '000300', start_date: str = None, end_date: str = None) -> Optional[pd.Series]:
    """
    获取基准指数日线收盘价（默认沪深300），用于计算相对强弱（如果需要，但本版本未使用）
    """
    try:
        df = ak.stock_zh_index_daily(symbol=f"sh{code}")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        df.set_index('date', inplace=True)
        return df['close'].astype(float)
    except Exception as e:
        print(f"获取基准指数失败: {e}")
        return None

# -------------------- 核心计算（因子） --------------------
def compute_factors(price_df: pd.DataFrame, volume_df: pd.DataFrame,
                    lookbacks: List[int] = [20, 60],
                    vol_window: int = 20,
                    trend_window: int = 20,
                    volume_window: int = 20) -> dict:
    """
    计算六个优化因子，返回每日的因子DataFrame（与price_df同索引，列为ETF代码）
    """
    factors = {}

    for n in lookbacks:
        factors[f'ret{n}'] = price_df / price_df.shift(n) - 1.0

    daily_ret = price_df.pct_change()
    vol = daily_ret.rolling(vol_window, min_periods=int(vol_window/2)).std() * np.sqrt(252)
    factors['volatility'] = vol

    def rolling_r2(series, window):
        def r2(arr):
            y = arr.values
            x = np.arange(len(y))
            if len(y) < 2 or np.std(y) == 0:
                return np.nan
            coeffs = np.polyfit(x, y, 1)
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        return series.rolling(window).apply(r2, raw=False)

    r2_df = price_df.apply(lambda col: rolling_r2(col, trend_window), axis=0)
    factors['trend_stability'] = r2_df

    ma20 = price_df.rolling(20).mean()
    bias = (price_df / ma20) - 1.0
    factors['bias'] = bias

    vol_ma = volume_df.rolling(volume_window, min_periods=1).mean()
    volume_ratio = volume_df / vol_ma
    volume_ratio = volume_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    factors['volume_ratio'] = volume_ratio

    return factors

def normalize_factors(factors_dict: dict) -> dict:
    """对每个因子进行横截面 Z-score 标准化"""
    norm_factors = {}
    for name, df in factors_dict.items():
        df_clean = df.ffill().fillna(0)
        mean = df_clean.mean(axis=1)
        std = df_clean.std(axis=1, ddof=0).replace(0, np.nan)
        norm = (df_clean.sub(mean, axis=0)).div(std, axis=0).fillna(0)
        norm_factors[name] = norm
    return norm_factors

def compute_composite(norm_factors: dict, weights: dict) -> pd.DataFrame:
    """根据权重计算综合得分"""
    first_key = list(norm_factors.keys())[0]
    composite = pd.DataFrame(0.0, index=norm_factors[first_key].index, columns=norm_factors[first_key].columns)
    for key, w in weights.items():
        if key in norm_factors:
            composite += w * norm_factors[key]
    return composite

# -------------------- 新增：市场宽度计算及仓位建议 --------------------


def advanced_market_breadth(price_df: pd.DataFrame, volume_df: pd.DataFrame) -> dict:
    """
    计算市场宽度指标，并返回最新一天的宽度值和仓位建议（在字典中）。
    """
    n_etfs = price_df.shape[1]
    turnover = price_df * volume_df
    total_turnover = turnover.sum(axis=1)
    total_turnover_safe = total_turnover.replace(0, np.nan)

    ma20 = price_df.rolling(20).mean()
    above_ma20 = (price_df > ma20).sum(axis=1) / n_etfs

    above_turnover = (price_df > ma20) * turnover
    weighted_above = above_turnover.sum(axis=1) / total_turnover_safe
    weighted_above = weighted_above.fillna(0)

    up = (price_df.diff() > 0).sum(axis=1) / n_etfs

    up_turnover = (price_df.diff() > 0) * turnover
    weighted_up = up_turnover.sum(axis=1) / total_turnover_safe
    weighted_up = weighted_up.fillna(0)

    high20 = price_df.rolling(20).max()
    low20 = price_df.rolling(20).min()
    new_high = (price_df == high20).sum(axis=1) / n_etfs
    new_low = (price_df == low20).sum(axis=1) / n_etfs
    net_high_low = new_high - new_low

    adv_dec = (up * n_etfs) - ((1 - up) * n_etfs)
    ad_line = adv_dec.cumsum()

    ema_short = adv_dec.ewm(span=19, adjust=False).mean()
    ema_long = adv_dec.ewm(span=39, adjust=False).mean()
    mcclellan = ema_short - ema_long

    latest = {}
    for name, series in [
        ('above_ma20', above_ma20),
        ('weighted_above_ma20', weighted_above),
        ('up_ratio', up),
        ('weighted_up', weighted_up),
        ('net_high_low', net_high_low),
        ('ad_line', ad_line),
        ('mcclellan', mcclellan)
    ]:
        latest[name] = series.iloc[-1] if not series.empty else np.nan

    # 生成仓位建议
    pct = latest['above_ma20']
    weighted = latest['weighted_above_ma20']
    mcc = latest['mcclellan']

    if pd.isna(pct):
        advice = "宽度数据不足，无法给出建议。"
    else:
        if pct < 0.3:
            advice = "市场弱势（站上MA20比例<30%），建议轻仓(≤20%)或空仓观望。"
        elif pct < 0.6:
            advice = "市场中性（站上MA20比例30%-60%），建议中等仓位(20%-50%)，精选个股。"
        else:
            advice = "市场强势（站上MA20比例>60%），建议较高仓位(50%-100%)，但需注意风险。"
            if mcc > 100:
                advice += " 麦克莱伦指标过高，可能短期超买，追高需谨慎。"

        if weighted < pct - 0.1:
            advice += " 加权宽度低于简单宽度，小盘股相对活跃，大资金参与度不高。"
        elif weighted > pct + 0.1:
            advice += " 加权宽度高于简单宽度，大市值ETF更受资金青睐。"

    latest['仓位建议'] = advice

    # 控制台输出（可选，但已包含在返回值中）
    print("\n========== 市场宽度监控 ==========")
    print(f"最新日期: {price_df.index[-1].strftime('%Y-%m-%d') if not price_df.empty else '无数据'}")
    print(f"站上20日线比例: {pct:.1%} (加权: {weighted:.1%})")
    print(f"上涨比例: {latest['up_ratio']:.1%} (加权: {latest['weighted_up']:.1%})")
    print(f"净新高比例(20日): {latest['net_high_low']:.1%}")
    print(f"麦克莱伦振荡器: {mcc:.2f}")
    print(f"【仓位建议】{advice}")

    return latest


# -------------------- 主监控函数 --------------------
def monitor_etfs_from_file(
    input_file: str = 'data/etf_top1_by_category.csv',
    output_file: str = 'data/etf_ranked.csv',
    start_date: str = None,
    end_date: str = None,
    weights: dict = None
) -> pd.DataFrame:
    """
    从 CSV 文件读取 ETF 列表，计算优化后的六个因子，输出综合得分排名，并附加市场宽度建议。
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')

    df_etf = pd.read_csv(input_file)
    df_etf['基金代码'] = df_etf['基金代码'].astype(str).str.strip()
    etf_codes = df_etf['基金代码'].tolist()
    print(f"共读取 {len(etf_codes)} 只 ETF")

    price_dict = {}
    volume_dict = {}
    for code in etf_codes:
        df = fetch_etf_daily(code, start_date, end_date)
        if df is not None and len(df) > 30:
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

    # 计算宽度并输出建议
    breadth_latest = advanced_market_breadth(price_df, volume_df)

    # 计算因子排名
    factors = compute_factors(price_df, volume_df)
    norm_factors = normalize_factors(factors)

    if weights is None:
        weights = {
            'ret20': 0.25,
            'ret60': 0.10,
            'volatility': 0.15,
            'trend_stability': 0.20,
            'bias': 0.15,
            'volume_ratio': 0.10
        }
        print("使用默认权重:", weights)

    composite = compute_composite(norm_factors, weights)
    latest_date = composite.index[-1]
    latest_scores = composite.loc[latest_date].sort_values(ascending=False)

    result_df = pd.DataFrame(index=latest_scores.index)
    result_df['基金代码'] = result_df.index
    result_df['最新收盘价'] = price_df.loc[latest_date].values
    for name in factors.keys():
        if name in factors and latest_date in factors[name].index:
            result_df[f'{name}'] = factors[name].loc[latest_date].round(4)
    result_df['bias'] = result_df.get('bias', 0)
    result_df['过热标记'] = result_df['bias'].apply(lambda x: '⚠️高乖离' if x > 0.15 else '')
    result_df['综合得分'] = latest_scores.values
    result_df['排名'] = latest_scores.rank(ascending=False).astype(int)

    final_df = result_df.merge(df_etf, on='基金代码', how='left')

    # 处理列名冲突
    if '综合得分_x' in final_df.columns:
        final_df.rename(columns={'综合得分_x': '综合得分'}, inplace=True)
        if '综合得分_y' in final_df.columns:
            final_df.drop(columns=['综合得分_y'], inplace=True)
    elif '综合得分_y' in final_df.columns:
        final_df.rename(columns={'综合得分_y': '综合得分'}, inplace=True)

    final_df = final_df.sort_values('综合得分', ascending=False).reset_index(drop=True)

    cols = ['基金代码', '基金简称', '小类', '综合得分', '排名', '最新收盘价'] + \
           [c for c in final_df.columns if c not in ['基金代码', '基金简称', '小类', '综合得分', '排名', '最新收盘价']]
    final_df = final_df[cols]

    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n因子排名结果已保存至 {output_file}")

    return final_df

# -------------------- 执行 --------------------
if __name__ == "__main__":
    ranked = monitor_etfs_from_file(
        input_file='data/etf_top1_by_category.csv',
        output_file='data/etf_ranked.csv'
    )
    print("\n综合得分排名前10：")
    print(ranked[['基金代码', '基金简称', '小类', '综合得分', '排名']].head(10))