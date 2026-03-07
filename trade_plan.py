import numpy as np
import pandas as pd
import get_data

# ==================== 波动率计算函数（复用之前的内容） ====================
def estimate_daily_95_range(df, long_window=500, short_window=20, quantile=0.95, scale_limits=(0.5, 2.0)):
    """
    计算每日振幅的95%范围估计（基于历史数据）
    参数：
        df : DataFrame，必须包含 'high' 和 'low' 列
    返回：
        Series，估计值序列，索引与df相同
    """
    amplitude = df['high'] - df['low']
    long_quantile = amplitude.rolling(window=long_window, min_periods=long_window).quantile(quantile)
    long_mean = amplitude.rolling(window=long_window, min_periods=long_window).mean()
    short_mean = amplitude.rolling(window=short_window, min_periods=short_window).mean()
    scale = short_mean / long_mean
    scale = scale.clip(*scale_limits)
    estimate = long_quantile * scale
    return estimate

def estimate_sigma_from_data(df, method='auto', atr_window=20, quantile_factor=1.8):
    """
    从历史数据中估计当前的当日95%波动幅度 sigma
    参数：
        df : DataFrame，已按日期正序排列，包含 'high', 'low', 'date'
        method : 'auto' 根据数据量自动选择；'quantile' 强制用分位数法；'atr' 强制用ATR近似
        atr_window : ATR窗口
        quantile_factor : ATR近似为95%分位数的系数（默认1.8，需根据品种校准）
    返回：
        sigma : float，当前估计值（最新一天）
        如果数据不足，返回None
    """
    if len(df) < 20:
        return None
    if method == 'auto':
        if len(df) >= 100:
            method = 'quantile'
        else:
            method = 'atr'
    if method == 'quantile':
        est_series = estimate_daily_95_range(df, long_window=min(500, len(df)//2), short_window=20)
        sigma = est_series.iloc[-1]
        if pd.isna(sigma):
            # 如果分位数仍为NaN（窗口不足），回退到ATR
            method = 'atr'
        else:
            return sigma
    if method == 'atr':
        # 计算ATR(20)的简单近似：最近20天振幅均值
        df['amplitude'] = df['high'] - df['low']
        atr = df['amplitude'].rolling(window=atr_window, min_periods=atr_window).mean().iloc[-1]
        sigma = atr * quantile_factor   # 经验系数
        return sigma
    return None

# ==================== 交易计划生成函数 ====================
def generate_trade_plan(etf_code, current_price, capital,
                        risk_per_trade=0.02,
                        T=5,
                        k=2.0,
                        alpha=0.6,
                        beta=0.4,
                        drawdown=0.08,
                        max_adds=2,
                        sigma_method='auto',
                        atr_factor=1.8):
    """
    生成ETF交易计划（建仓、止损、加仓、止盈）
    参数：
        etf_code : str, ETF代码，用于获取历史数据
        current_price : float, 当前价格（计划买入价）
        capital : float, 初始资金
        risk_per_trade : float, 单笔风险比例（默认2%）
        T : int, 持仓验证周期（天）
        k : float, 加仓倍数（相对于初始止损距离）
        alpha, beta : 第一次和第二次加仓比例（相对于初始底仓）
        drawdown : float, 总资产回撤清仓比例
        max_adds : int, 最大加仓次数（默认2）
        sigma_method : 'auto', 'quantile' 或 'atr'
        atr_factor : float, ATR换算为95%分位数的系数（默认1.8）
    返回：
        打印交易计划，并返回包含详细信息的字典
    """
    # 1. 获取历史数据
    try:
        df = get_data.get_etf_data(etf_code)   # 用户提供的函数，返回DataFrame，至少包含 'date', 'high', 'low', 'close'
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None

    # 2. 确保数据正序
    if 'date' not in df.columns:
        raise ValueError("数据缺少日期列")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 3. 估计当前sigma
    sigma = estimate_sigma_from_data(df, method=sigma_method, quantile_factor=atr_factor)
    if sigma is None or np.isnan(sigma):
        print("历史数据不足，无法估计波动率，请补充数据后重试。")
        return None

    # 4. 计算初始止损距离
    stop_distance = sigma * np.sqrt(T)
    # 5. 计算建仓数量
    risk_amount = capital * risk_per_trade
    max_shares_by_risk = int(risk_amount / stop_distance)
    max_shares_by_cash = int(capital / current_price)
    shares_to_buy = min(max_shares_by_risk, max_shares_by_cash)
    if shares_to_buy <= 0:
        print("资金不足或风险预算太小，无法建仓")
        return None

    # 建仓占用资金
    cost = shares_to_buy * current_price
    remaining_cash = capital - cost

    # 6. 止损价
    stop_price = current_price - stop_distance

    # 7. 加仓点
    add_price_1 = current_price + k * stop_distance
    add_price_2 = add_price_1 + k * stop_distance   # 如果允许第二次加仓

    # 8. 加仓数量
    add_shares_1 = int(alpha * shares_to_buy)
    add_shares_2 = int(beta * shares_to_buy)

    # 9. 计算加仓后的累计持仓和平均成本
    # 第一次加仓后
    total_shares_1 = shares_to_buy + add_shares_1
    total_cost_1 = cost + add_shares_1 * add_price_1
    avg_price_1 = total_cost_1 / total_shares_1 if total_shares_1 > 0 else 0
    remaining_cash_1 = remaining_cash - add_shares_1 * add_price_1

    # 第二次加仓后
    total_shares_2 = total_shares_1 + add_shares_2
    total_cost_2 = total_cost_1 + add_shares_2 * add_price_2
    avg_price_2 = total_cost_2 / total_shares_2 if total_shares_2 > 0 else 0
    remaining_cash_2 = remaining_cash_1 - add_shares_2 * add_price_2

    # 10. 输出计划
    print("=" * 70)
    print(f"ETF代码: {etf_code}")
    print(f"当前价格: {current_price:.3f}")
    print(f"初始资金: {capital:.2f}")
    print("-" * 70)
    print("【建仓建议】")
    print(f"  买入数量: {shares_to_buy} 股")
    print(f"  买入金额: {cost:.2f} 元")
    print(f"  剩余现金: {remaining_cash:.2f} 元")
    print(f"  初始止损价: {stop_price:.3f} (距离 {stop_distance:.3f} 元)")
    print(f"  单笔风险: {risk_amount:.2f} 元 ({risk_per_trade:.0%})")
    print("-" * 70)
    print("【加仓计划】（基于固定初始止损距离）")
    if max_adds >= 1:
        print(f"  第一次加仓价: {add_price_1:.3f} (需上涨 {k*stop_distance:.3f} 元)")
        print(f"  第一次加仓数量: {add_shares_1} 股 (需资金 {add_shares_1*add_price_1:.2f} 元)")
        print(f"    → 第一次加仓后总持股: {total_shares_1} 股")
        print(f"    → 第一次加仓后平均成本: {avg_price_1:.3f} 元")
        print(f"    → 第一次加仓后剩余现金: {remaining_cash_1:.2f} 元")
    if max_adds >= 2:
        print(f"  第二次加仓价: {add_price_2:.3f} (需再涨 {k*stop_distance:.3f} 元)")
        print(f"  第二次加仓数量: {add_shares_2} 股 (需资金 {add_shares_2*add_price_2:.2f} 元)")
        print(f"    → 第二次加仓后总持股: {total_shares_2} 股")
        print(f"    → 第二次加仓后平均成本: {avg_price_2:.3f} 元")
        print(f"    → 第二次加仓后剩余现金: {remaining_cash_2:.2f} 元")
    print("-" * 70)
    print("【止盈规则】")
    print(f"  当总资产从最高点回撤 {drawdown:.0%} 时，全部清仓。")
    print("  请每日记录持仓市值，更新最高点，触发时无条件卖出。")
    print("=" * 70)

    # 返回详细数据字典（也包含新计算的字段）
    plan = {
        'etf_code': etf_code,
        'current_price': current_price,
        'capital': capital,
        'shares_to_buy': shares_to_buy,
        'cost': cost,
        'remaining_cash': remaining_cash,
        'stop_price': stop_price,
        'stop_distance': stop_distance,
        'add_prices': [add_price_1, add_price_2][:max_adds],
        'add_shares': [add_shares_1, add_shares_2][:max_adds],
        'after_add1': {
            'total_shares': total_shares_1,
            'avg_price': avg_price_1,
            'remaining_cash': remaining_cash_1
        } if max_adds >= 1 else None,
        'after_add2': {
            'total_shares': total_shares_2,
            'avg_price': avg_price_2,
            'remaining_cash': remaining_cash_2
        } if max_adds >= 2 else None,
        'drawdown': drawdown,
        'sigma': sigma
    }
    return plan


#计算历史回撤分布用于止盈判断
def calibrate_take_profit(df, T=5, quantile=0.95, mode='peak', high_lookback=20,
                          method='percent', date_col='date', high_col='high', price_col='close'):
    """
    基于历史数据计算止盈阈值（回撤分位数）。

    方法A适用于全局移动止盈，
    如果你随机在某天买入并持有T天，那么有95%的概率，你在这段时间内遇到的最大回撤不会超过这个阈值。
    方法A要用的话T要改成持仓天数

    方法A阈值: 18.31%，代表着如果你在任意一天买入并持有T天，那么有95%的概率，在这T天内从最高点往下的最大回撤不会超过18.31%。
    
    方法B适用于创新高后的固定止盈。
    当价格创出新高时，未来T天内可能出现的最大回撤有95%的概率不会超过这个阈值
    
    方法B阈值: 10.72%，代表着当价格创出新高时，未来T天内可能出现的最大回撤有95%的概率不会超过10.72%。

    参数：
        df : DataFrame，必须包含日期列、最高价列和价格列（通常用收盘价）
        T : 持仓天数（用于方法A）或新高后观察天数（用于方法B）
        quantile : 目标分位数（例如0.95）
        mode : 'peak' 使用方法B（创新高后），'entry' 使用方法A（任意建仓日）
        high_lookback : 仅 mode='peak' 时有效，定义“新高”的回顾窗口（例如20天）
        method : 'percent' 返回百分比回撤，'absolute' 返回金额回撤
        date_col, high_col, price_col : 列名

    返回：
        threshold : 回撤阈值，若 method='percent' 则返回百分比（如0.05），否则返回金额
        drawdown_series : 所有样本的最大回撤序列（可用于稳定性检验）
    """
    df = df.copy()
    # 确保日期正序
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)

    n = len(df)
    max_drawdowns = []

    if mode == 'entry':
        # 方法A：对每个可能的建仓日，计算持有T天内的最大回撤
        for i in range(n - T):
            window = df.iloc[i: i+T+1]  # 包含建仓日和T天后的日期
            # 找到窗口内最高价的位置
            peak_idx = window[high_col].idxmax()
            # 如果最高价在最后一天，则回撤为0
            if peak_idx == window.index[-1]:
                max_dd = 0
            else:
                # 从最高价之后到窗口结束的最低收盘价
                after_peak = window.loc[peak_idx+1:]
                lowest = after_peak[price_col].min()
                peak_price = window.loc[peak_idx, price_col]  # 用收盘价作为高点价格
                if method == 'percent':
                    dd = (peak_price - lowest) / peak_price
                else:
                    dd = peak_price - lowest
                max_dd = dd
            max_drawdowns.append(max_dd)

    elif mode == 'peak':
        # 方法B：先识别新高点
        # 计算过去 high_lookback 天的最高价（用最高价列）
        df['past_high'] = df[high_col].rolling(window=high_lookback, min_periods=high_lookback).max().shift(1)
        df['new_high'] = df[high_col] > df['past_high']

        for i in range(n - T):
            if not df.loc[i, 'new_high']:
                continue
            # 从新高日之后一天开始，到T天后结束
            start = i + 1
            end = i + T + 1
            if end >= n:
                continue
            window = df.iloc[start:end]
            # 新高日的收盘价作为基准
            peak_price = df.loc[i, price_col]
            lowest = window[price_col].min()
            if method == 'percent':
                dd = (peak_price - lowest) / peak_price
            else:
                dd = peak_price - lowest
            max_drawdowns.append(dd)

    else:
        raise ValueError("mode 必须是 'peak' 或 'entry'")

    if len(max_drawdowns) == 0:
        print("没有足够的样本，请调整参数或检查数据")
        return None, None

    drawdown_series = pd.Series(max_drawdowns)
    threshold = drawdown_series.quantile(quantile)

    return threshold, drawdown_series


