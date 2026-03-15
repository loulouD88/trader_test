import pandas as pd
import numpy as np
import akshare as ak
import time
from datetime import datetime, timedelta

# -------------------- 获取ETF数据（使用新浪接口） --------------------
def fetch_etf_data(symbol_with_prefix):
    try:
        df = ak.fund_etf_hist_sina(symbol=symbol_with_prefix)
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
        prices = df['close'].astype(float)
        prices = prices[~prices.index.duplicated(keep='first')]
        return prices
    except Exception as e:
        print(f"  获取 {symbol_with_prefix} 数据失败: {e}")
        return None

# -------------------- 获取沪深300指数数据 --------------------
def fetch_benchmark_data(end_date):
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    end_ts = pd.to_datetime(end_date)
    df = df[df.index <= end_ts]
    if df.empty:
        raise ValueError("获取沪深300指数数据为空")
    return df['close'].astype(float)

# -------------------- 为基金代码添加前缀 --------------------
def add_prefix(code):
    code_str = str(code).strip()
    if code_str.startswith(('5', '588')):
        return 'sh' + code_str
    elif code_str.startswith('159'):
        return 'sz' + code_str
    else:
        return 'sh' + code_str

# -------------------- 主计算函数 --------------------
def calculate_metrics_from_csv(csv_path,
                               end_date=None,
                               risk_free_rate=0.025,
                               min_common_days=30,
                               output_path='etf_metrics_result.csv'):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # 读取ETF列表，并将基金代码列转为字符串
    df_etf = pd.read_csv(csv_path)
    df_etf['基金代码'] = df_etf['基金代码'].astype(str)

    required_cols = ['基金代码', '基金简称', '类型', '机构持有比例', '个人持有比例',
                     '资产规模', '原始摘要', '小类']
    if not all(col in df_etf.columns for col in required_cols):
        raise ValueError(f"CSV文件必须包含列: {required_cols}")

    # 获取基准指数数据
    print("正在获取沪深300指数数据...")
    bench_prices = fetch_benchmark_data(end_date)
    bench_prices = bench_prices[~bench_prices.index.duplicated(keep='first')].sort_index()
    bench_returns = (bench_prices / bench_prices.shift(1) - 1).dropna()
    print(f"基准指数数据范围: {bench_prices.index.min()} 至 {bench_prices.index.max()}, 共 {len(bench_prices)} 天")

    records = []

    for idx, row in df_etf.iterrows():
        code_num = row['基金代码']  # 已经是字符串
        full_code = add_prefix(code_num)
        print(f"正在处理 {idx+1}/{len(df_etf)}: {full_code} {row['基金简称']}")

        etf_prices = fetch_etf_data(full_code)
        if etf_prices is None or len(etf_prices) == 0:
            print(f"  {full_code} 无数据，跳过")
            records.append({'基金代码': code_num, '计算状态': '无数据'})
            time.sleep(0.5)
            continue

        end_ts = pd.to_datetime(end_date)
        etf_prices = etf_prices[etf_prices.index <= end_ts]
        if len(etf_prices) < 30:
            print(f"  {full_code} 有效数据不足30天，跳过")
            records.append({'基金代码': code_num, '计算状态': '数据不足'})
            time.sleep(0.5)
            continue

        print(f"  数据范围: {etf_prices.index.min()} 至 {etf_prices.index.max()}, 共{len(etf_prices)}天")

        etf_returns = (etf_prices / etf_prices.shift(1) - 1).dropna()
        common_dates = etf_returns.index.intersection(bench_returns.index)
        print(f"  共同交易日: {len(common_dates)} 天")

        if len(common_dates) < min_common_days:
            print(f"  {full_code} 共同交易日{len(common_dates)} < {min_common_days}，跳过")
            records.append({'基金代码': code_num, '计算状态': f'共同交易日<{min_common_days}'})
            time.sleep(0.5)
            continue

        r_etf = etf_returns.loc[common_dates]
        r_bench = bench_returns.loc[common_dates]

        total_return = (etf_prices.loc[common_dates[-1]] / etf_prices.loc[common_dates[0]]) - 1
        years = len(common_dates) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1
        annual_vol = r_etf.std() * np.sqrt(252)
        sharpe = (annual_return - risk_free_rate) / annual_vol
        cov = np.cov(r_etf, r_bench)[0, 1]
        var = np.var(r_bench)
        beta = cov / var
        alpha_daily = r_etf.mean() - beta * r_bench.mean()
        alpha_annual = (1 + alpha_daily) ** 252 - 1

        records.append({
            '基金代码': code_num,
            '年化收益率': annual_return,
            '年化波动率': annual_vol,
            '夏普比率': sharpe,
            'Beta': beta,
            'Alpha': alpha_annual,
            '数据起止': f"{common_dates[0].strftime('%Y-%m-%d')} 至 {common_dates[-1].strftime('%Y-%m-%d')}",
            '交易日数': len(common_dates),
            '计算状态': '成功'
        })

        time.sleep(0.5)

    df_metrics = pd.DataFrame(records)
    df_result = df_etf.merge(df_metrics, on='基金代码', how='left')
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至 {output_path}")
    success = df_result[df_result['计算状态'] == '成功'].shape[0]
    print(f"成功计算 {success} 只ETF，总计 {len(df_etf)} 只")
    return df_result

if __name__ == "__main__":
    result = calculate_metrics_from_csv(
        csv_path='data/etf_list.csv',
        end_date='2025-03-01',
        risk_free_rate=0.025,
        min_common_days=30,
        output_path='data/etf_with_metrics.csv'
    )