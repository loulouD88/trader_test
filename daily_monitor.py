"""
daily_monitor.py - 每日ETF监控主程序（完整版）
整合相对强度、市场宽度、资金流向（多周期）、成交量异动、情绪指标五大模块
输出Excel报告，资金流向按最新排名排序
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# -------------------- 导入各模块 --------------------
from etf_relative_strength import (
    fetch_etf_daily,
    compute_factors,
    normalize_factors,
    compute_composite,
    advanced_market_breadth
)
from new_etf_fund_flow_analyze import FundFlowTimeSeriesAnalyzer
from etf_volume_analyze import analyze_volume
from margin_data import MarginDataUpdater, sentiment_from_margin_file
# 新增导入风格轮动模块和分类字典
from style_rotation import style_rotation_monitor, CLASSIFICATION

# -------------------- 配置 --------------------
ETF_LIST_FILE = 'data/etf_top1_by_category.csv'
PRICE_DATA_START = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
PRICE_DATA_END = datetime.now().strftime('%Y-%m-%d')
OUTPUT_REPORT = f"reports/daily_report_{datetime.now().strftime('%Y%m%d')}.xlsx"
os.makedirs('reports', exist_ok=True)

RS_WEIGHTS = {
    'ret20': 0.25,
    'ret60': 0.10,
    'volatility': 0.15,
    'trend_stability': 0.20,
    'bias': 0.15,
    'volume_ratio': 0.10
}
FUND_FLOW_PERIODS = ['3日', '5日', '10日', '20日']

# -------------------- 数据获取 --------------------
print("="*50)
print("开始获取基础数据...")

etf_info = pd.read_csv(ETF_LIST_FILE)
etf_info['基金代码'] = etf_info['基金代码'].astype(str).str.strip()
etf_codes = etf_info['基金代码'].tolist()
print(f"共 {len(etf_codes)} 只ETF")

price_dict = {}
volume_dict = {}
for code in etf_codes:
    df = fetch_etf_daily(code, PRICE_DATA_START, PRICE_DATA_END)
    if df is not None and len(df) > 30:
        price_dict[code] = df['close']
        volume_dict[code] = df['volume']
    time.sleep(0.3)

if not price_dict:
    raise RuntimeError("没有获取到任何有效ETF数据")

price_df = pd.DataFrame(price_dict).sort_index().asfreq('D').ffill().dropna()
volume_df = pd.DataFrame(volume_dict).sort_index().asfreq('D').ffill().fillna(0)
volume_df = volume_df.reindex(columns=price_df.columns, fill_value=0)
print(f"价格数据范围: {price_df.index.min()} 至 {price_df.index.max()}, 共 {len(price_df)} 天")

print("\n更新融资余额数据...")
margin_updater = MarginDataUpdater(csv_path='data/margin_data.csv')
margin_updater.update(start_date='20180101', delay_sz=0.5)
margin_df = margin_updater.load_data()

print("\n更新资金流向数据...")
fund_analyzer = FundFlowTimeSeriesAnalyzer(data_dir='fund_flow_data')
fund_analyzer.update_daily(delay=0.5)

# -------------------- 运行各模块 --------------------
print("\n" + "="*50)
print("开始分析...")

# ---- 1. 相对强度排名 ----
print("\n【相对强度】")
factors = compute_factors(price_df, volume_df)
norm_factors = normalize_factors(factors)
composite = compute_composite(norm_factors, RS_WEIGHTS)
latest_date = composite.index[-1]
latest_scores = composite.loc[latest_date].sort_values(ascending=False)
rs_result = pd.DataFrame({
    '基金代码': latest_scores.index,
    '综合得分': latest_scores.values,
    '排名': latest_scores.rank(ascending=False).astype(int)
})
rs_result = rs_result.merge(etf_info[['基金代码', '基金简称', '小类']], on='基金代码', how='left')
rs_result = rs_result.sort_values('排名').reset_index(drop=True)

# ---- 2. 市场宽度（含仓位建议） ----
print("\n【市场宽度】")
breadth = advanced_market_breadth(price_df, volume_df)   # 返回字典，包含'仓位建议'
breadth_df = pd.DataFrame([breadth])

# ---- 3. 成交量异动 ----
print("\n【成交量异动】")
vol_result = analyze_volume(
    price_df, volume_df,
    short_window=5, long_window=20,
    vol_ratio_threshold=0.2, vol_spike_threshold=1.5, corr_threshold=0.3
)
vol_result = vol_result.merge(etf_info[['基金代码', '基金简称', '小类']], left_index=True, right_on='基金代码', how='left')
vol_result.set_index('基金代码', inplace=True)

# ---- 4. 资金流向（多周期分析） ----
print("\n【资金流向】")
fund_results = {}
fund_reports = {}
for period in FUND_FLOW_PERIODS:
    print(f"  分析 {period} 周期...")
    res_df, report_text = fund_analyzer.analyze(period=period, window=20, short_window=5, long_window=20, rank_window=5)
    if res_df is not None and not res_df.empty:
        # 按最新排名升序（排名1在上）
        if '最新排名' in res_df.columns:
            res_df = res_df.sort_values('最新排名', ascending=True)
        else:
            # 降级按动量排序
            if '动量(短-长)' in res_df.columns:
                res_df = res_df.sort_values('动量(短-长)', ascending=False)
        fund_results[period] = res_df
        fund_reports[period] = report_text

# ---- 5. 情绪指标 ----
print("\n【情绪指标】")
sentiment = sentiment_from_margin_file('data/margin_data.csv', years=1)
sentiment_df = pd.DataFrame([sentiment])

# ---- 6. 风格轮动（新增） ----
print("\n【风格轮动】")
style_result, _ = style_rotation_monitor(price_df, etf_info, window=5, long_window=20)

# 解析风格结论，得到走强的风格列表
strong_styles = []
if style_result is not None and '风格结论' in style_result.columns:
    conclusion_str = style_result.iloc[0]['风格结论']
    if '价值相对走强' in conclusion_str:
        strong_styles.append('价值')
    if '成长相对走强' in conclusion_str:
        strong_styles.append('成长')
    if '周期相对走强' in conclusion_str:
        strong_styles.append('周期')
    if '防御相对走强' in conclusion_str:
        strong_styles.append('防御')
    if '大盘相对走强' in conclusion_str:
        strong_styles.append('大盘')
    if '小盘相对走强' in conclusion_str:
        strong_styles.append('小盘')
    if '上游相对走强' in conclusion_str:
        strong_styles.append('上游')
    if '下游相对走强' in conclusion_str:
        strong_styles.append('下游')

# 为 rs_result 添加风格分类列
# 建立小类到分类的映射字典
subclass_to_styles = {}
for sub, info in CLASSIFICATION.items():
    subclass_to_styles[sub] = info

def get_style_category(row, key):
    sub = row['小类']
    if sub in subclass_to_styles:
        return subclass_to_styles[sub].get(key, '-')
    return '-'

rs_result['价值/成长'] = rs_result.apply(lambda row: get_style_category(row, 'value_growth'), axis=1)
rs_result['周期/防御'] = rs_result.apply(lambda row: get_style_category(row, 'cycle_defensive'), axis=1)
rs_result['大盘/小盘'] = rs_result.apply(lambda row: get_style_category(row, 'cap'), axis=1)
rs_result['上游/下游'] = rs_result.apply(lambda row: get_style_category(row, 'chain'), axis=1)

# 生成风格匹配列：如果该ETF的某一风格在 strong_styles 中，则标记
def style_match(row):
    match = []
    if row['价值/成长'] in strong_styles:
        match.append(row['价值/成长'])
    if row['周期/防御'] in strong_styles:
        match.append(row['周期/防御'])
    if row['大盘/小盘'] in strong_styles:
        match.append(row['大盘/小盘'])
    if row['上游/下游'] in strong_styles:
        match.append(row['上游/下游'])
    return '、'.join(match) if match else '-'

rs_result['风格匹配'] = rs_result.apply(style_match, axis=1)

# -------------------- 生成报告 --------------------
print("\n" + "="*50)
print("生成报告...")

with pd.ExcelWriter(OUTPUT_REPORT, engine='openpyxl') as writer:
    # 相对强度排名（前50）
    rs_result.head(50).to_excel(writer, sheet_name='相对强度Top50', index=False)

    # 市场宽度（含仓位建议）
    breadth_df.to_excel(writer, sheet_name='市场宽度', index=False)

    # 成交量异动（仅显示非正常状态的ETF）
    vol_abnormal = vol_result[vol_result['成交量状态'] != '正常']
    if not vol_abnormal.empty:
        vol_abnormal.to_excel(writer, sheet_name='成交量异动', index=True)
    else:
        vol_result.head(20).to_excel(writer, sheet_name='成交量异动', index=True)

    # 资金流向：每个周期一个工作表，报告在上，表格在下（动态调整行距）
    for period, df in fund_results.items():
        sheet_name = f'资金流向_{period}'
        report_lines = fund_reports[period].strip().split('\n')
        start_row = len(report_lines) + 2
        df.to_excel(writer, sheet_name=sheet_name, index=True, startrow=start_row)
        worksheet = writer.sheets[sheet_name]
        for i, line in enumerate(report_lines):
            worksheet.cell(row=i+1, column=1, value=line)

    # 情绪指标
    sentiment_df.to_excel(writer, sheet_name='情绪指标', index=False)

    # ETF综合信息（合并相对强度和成交量状态，并包含风格列）
    combined = rs_result.merge(
        vol_result[['成交量状态', '量比趋势', '连续放量天数']],
        left_on='基金代码', right_index=True, how='left'
    )
    # 调整列顺序，让风格列靠前
    col_order = ['基金代码', '基金简称', '小类', '价值/成长', '周期/防御', '大盘/小盘', '上游/下游', '风格匹配',
                 '综合得分', '排名', '最新收盘价', '成交量状态', '量比趋势', '连续放量天数']
    combined = combined[col_order]
    combined.to_excel(writer, sheet_name='ETF综合信息', index=False)

    # 风格轮动（新增工作表）
    if style_result is not None:
        style_result.to_excel(writer, sheet_name='风格轮动', index=False)

print(f"报告已保存至: {OUTPUT_REPORT}")
print("所有分析完成。")