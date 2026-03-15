import pandas as pd
import numpy as np
import akshare as ak

from scipy import stats

#####初级版资金分析
def analyze_fund_flow_multi_period():
    """
    获取多周期行业资金流向数据（3日、5日、10日、20日排行），生成分析结论。
    返回 flow_df（各周期净额，单位亿元）和 rank_df（排名）。
    """
    periods = ['3日排行', '5日排行', '10日排行', '20日排行']
    flow_data = {}

    print("正在获取行业资金流向数据...")
    for p in periods:
        try:
            df = ak.stock_fund_flow_industry(symbol=p)
            # 根据用户提供的列名，行业列名为 '行业'，净额列名为 '净额'（单位亿元）
            industry_col = '行业'
            net_col = '净额'
            if industry_col in df.columns and net_col in df.columns:
                # 直接使用净额（亿元），不再转换
                flow = df.set_index(industry_col)[net_col].astype(float)
                flow_data[p] = flow
            else:
                print(f"警告：{p} 数据缺少必要的列。实际列名：{df.columns.tolist()}")
        except Exception as e:
            print(f"获取 {p} 失败：{e}")

    if not flow_data:
        print("没有获取到任何资金流向数据，无法分析。")
        return

    # 合并为DataFrame，缺失的行业填0
    flow_df = pd.DataFrame(flow_data).fillna(0)
    flow_df = flow_df[periods]  # 确保列顺序

    # 计算各周期的排名（降序，净流入越大排名越靠前）
    rank_df = flow_df.rank(axis=0, ascending=False, method='min').astype(int)

    # ---------- 1. 趋势确认 ----------
    print("\n" + "="*60)
    print("【1. 趋势确认】")
    # 强势：所有周期净流入 > 0
    strong_mask = (flow_df > 0).all(axis=1)
    strong_df = flow_df[strong_mask].copy()
    # 检查是否递增（允许小幅波动，此处简单判断 3日 <= 5日 <= 10日 <= 20日）
    inc_mask = (strong_df['3日排行'] <= strong_df['5日排行'] * 1.05) & \
               (strong_df['5日排行'] <= strong_df['10日排行'] * 1.05) & \
               (strong_df['10日排行'] <= strong_df['20日排行'] * 1.05)
    strong_inc = strong_df[inc_mask].index.tolist()
    strong_stable = strong_df[~inc_mask].index.tolist()

    # 弱势：所有周期净流入 < 0
    weak_mask = (flow_df < 0).all(axis=1)
    weak_df = flow_df[weak_mask].copy()
    dec_mask = (weak_df['3日排行'] >= weak_df['5日排行'] * 1.05) & \
               (weak_df['5日排行'] >= weak_df['10日排行'] * 1.05) & \
               (weak_df['10日排行'] >= weak_df['20日排行'] * 1.05)
    weak_dec = weak_df[dec_mask].index.tolist()
    weak_other = weak_df[~dec_mask].index.tolist()

    # 分歧：其余
    mixed = flow_df[~(strong_mask | weak_mask)].index.tolist()

    print(f"强势行业（所有周期净流入为正）：")
    if strong_inc:
        print(f"  - 递增/稳定：{', '.join(strong_inc[:10])}{'...' if len(strong_inc)>10 else ''}")
    if strong_stable:
        print(f"  - 基本稳定：{', '.join(strong_stable[:10])}{'...' if len(strong_stable)>10 else ''}")
    print(f"弱势行业（所有周期净流出）：")
    if weak_dec:
        print(f"  - 流出加大：{', '.join(weak_dec[:10])}{'...' if len(weak_dec)>10 else ''}")
    if weak_other:
        print(f"  - 其他：{', '.join(weak_other[:10])}{'...' if len(weak_other)>10 else ''}")
    print(f"分歧行业：")
    if mixed:
        print(f"  {', '.join(mixed[:15])}{'...' if len(mixed)>15 else ''}")

    # ---------- 2. 资金加速/减速 ----------
    print("\n" + "="*60)
    print("【2. 资金加速/减速】")
    # 加速流入：3日 > 5日 > 10日 > 20日
    acc_cond = (flow_df['3日排行'] > flow_df['5日排行']) & \
               (flow_df['5日排行'] > flow_df['10日排行']) & \
               (flow_df['10日排行'] > flow_df['20日排行'])
    acc_industries = flow_df[acc_cond].index.tolist()
    print("资金加速流入：")
    print(f"  {', '.join(acc_industries[:10])}{'...' if len(acc_industries)>10 else ''}" if acc_industries else "  无")

    # 减速流入：3日 < 5日 < 10日 < 20日
    dec_cond = (flow_df['3日排行'] < flow_df['5日排行']) & \
               (flow_df['5日排行'] < flow_df['10日排行']) & \
               (flow_df['10日排行'] < flow_df['20日排行'])
    dec_industries = flow_df[dec_cond].index.tolist()
    print("资金减速流入：")
    print(f"  {', '.join(dec_industries[:10])}{'...' if len(dec_industries)>10 else ''}" if dec_industries else "  无")

    # 加速流出：3日 < 5日 < 10日 < 20日 且 3日 < 0
    acc_out_cond = (flow_df['3日排行'] < flow_df['5日排行']) & \
                   (flow_df['5日排行'] < flow_df['10日排行']) & \
                   (flow_df['10日排行'] < flow_df['20日排行']) & \
                   (flow_df['3日排行'] < 0)
    acc_out_industries = flow_df[acc_out_cond].index.tolist()
    print("资金加速流出：")
    print(f"  {', '.join(acc_out_industries[:10])}{'...' if len(acc_out_industries)>10 else ''}" if acc_out_industries else "  无")

    # ---------- 3. 排名变化 ----------
    print("\n" + "="*60)
    print("【3. 排名变化（识别新热点/退潮）】")
    rank_change = rank_df['20日排行'] - rank_df['3日排行']
    rising = rank_change[rank_change > 0].sort_values(ascending=False)
    falling = rank_change[rank_change < 0].sort_values()

    print("排名上升最快的行业（近3日较20日排名提升）：")
    if not rising.empty:
        for idx, val in rising.head(10).items():
            print(f"  {idx}: 上升 {int(val)} 位")
    else:
        print("  无")

    print("排名下降最快的行业：")
    if not falling.empty:
        for idx, val in falling.head(10).items():
            print(f"  {idx}: 下降 {int(-val)} 位")
    else:
        print("  无")

    # ---------- 4. 市场整体资金流 ----------
    print("\n" + "="*60)
    print("【4. 市场整体资金流】")
    total_flow = flow_df.sum(axis=0)
    for p in periods:
        direction = "净流入" if total_flow[p] > 0 else "净流出"
        print(f"{p}：{direction} {abs(total_flow[p]):.2f} 亿元")

    if (total_flow['3日排行'] > 0) and (total_flow['20日排行'] > 0):
        print("市场整体：中期和短期均为净流入，资金面偏乐观。")
    elif (total_flow['3日排行'] < 0) and (total_flow['20日排行'] < 0):
        print("市场整体：中期和短期均为净流出，资金面偏谨慎。")
    elif (total_flow['3日排行'] > 0) and (total_flow['20日排行'] < 0):
        print("市场整体：短期净流入但长期净流出，可能为超跌反弹，需警惕持续性。")
    elif (total_flow['3日排行'] < 0) and (total_flow['20日排行'] > 0):
        print("市场整体：短期净流出但长期净流入，可能为正常调整，中期趋势未坏。")

    # ---------- 5. 背离分析提示 ----------
    print("\n" + "="*60)
    print("【5. 背离分析提示】")
    print("如需判断背离，请结合行业指数价格走势：")
    print("- 价格新高但中期资金流萎缩 → 顶背离")
    print("- 价格新低但中期资金流改善 → 底背离")

    print("\n" + "="*60)
    print("分析完成。")

    return flow_df, rank_df

####高级版资金分析
def analyze_fund_flow_advanced(
    periods=None,
    width_data: pd.Series = None,
    winsorize_limit: float = 3.0
):
    """
    高级多周期资金流向分析（修复索引重复及回归形状问题）

    参数:
        periods : list of str, 默认 ['3日排行','5日排行','10日排行','20日排行']
        width_data : pd.Series, 可选，市场宽度指标时间序列（如站上20日线比例），用于协同分析
        winsorize_limit : float, 极端值缩尾阈值（倍数标准差）

    返回:
        flow_df : 原始累计净额（亿元）
        daily_avg_df : 分段日均净额（亿元/日）
        summary : 分析结论文本
    """
    if periods is None:
        periods = ['3日排行', '5日排行', '10日排行', '20日排行']

    # 1. 获取原始数据
    flow_data = {}
    for p in periods:
        try:
            df = ak.stock_fund_flow_industry(symbol=p)
            # 根据实际列名调整（常见为 '行业' 和 '净额'）
            if '行业' in df.columns and '净额' in df.columns:
                flow = df.set_index('行业')['净额'].astype(float)
                # 去除重复行业（取第一个）
                flow = flow[~flow.index.duplicated(keep='first')]
                flow_data[p] = flow
            else:
                print(f"警告：{p} 列名异常，实际列名：{df.columns.tolist()}")
        except Exception as e:
            print(f"获取 {p} 失败：{e}")

    if not flow_data:
        print("无数据，无法分析")
        return None, None, ""

    # 2. 统一行业索引
    all_industries = set()
    for ser in flow_data.values():
        all_industries.update(ser.index)
    all_industries = sorted(all_industries)

    aligned_data = {}
    for p, ser in flow_data.items():
        aligned_ser = ser.reindex(all_industries).fillna(0)
        aligned_data[p] = aligned_ser

    flow_df = pd.DataFrame(aligned_data)
    flow_df = flow_df[periods]  # 确保列顺序

    # 3. 计算分段日均净额（亿元/日）
    daily_avg = pd.DataFrame(index=flow_df.index)
    daily_avg['最近3日'] = flow_df['3日排行'] / 3
    daily_avg['4-5日'] = (flow_df['5日排行'] - flow_df['3日排行']) / 2
    daily_avg['6-10日'] = (flow_df['10日排行'] - flow_df['5日排行']) / 5
    daily_avg['11-20日'] = (flow_df['20日排行'] - flow_df['10日排行']) / 10

    # 4. 极端值处理（缩尾）
    def winsorize_series(s):
        mean = s.mean()
        std = s.std()
        lower = mean - winsorize_limit * std
        upper = mean + winsorize_limit * std
        return s.clip(lower, upper)

    daily_avg_winsor = daily_avg.apply(winsorize_series, axis=0)

    # 5. 趋势强度（线性回归，添加原点0）
    x = np.array([0, 3, 5, 10, 20])
    slopes = []
    for industry in flow_df.index:
        y_vals = flow_df.loc[industry, periods].values  # 4个累计值
        y = np.concatenate([[0], y_vals])               # 添加原点，共5个点
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        slopes.append({
            '行业': industry,
            '斜率': slope,
            'R²': r_value**2,
            'p值': p_value,
            '趋势方向': '上升' if slope > 0 else '下降' if slope < 0 else '平稳'
        })
    slope_df = pd.DataFrame(slopes).set_index('行业')

    # 6. 加速/减速动量
    daily_avg_winsor['动量3-5'] = daily_avg_winsor['最近3日'] - daily_avg_winsor['4-5日']
    daily_avg_winsor['动量5-10'] = daily_avg_winsor['4-5日'] - daily_avg_winsor['6-10日']
    daily_avg_winsor['动量10-20'] = daily_avg_winsor['6-10日'] - daily_avg_winsor['11-20日']

    # 7. 排名变化（百分位数）
    rank_pct = flow_df.rank(axis=0, ascending=False, pct=True)
    rank_pct.columns = [f'{col}_百分位' for col in rank_pct.columns]
    rank_change = rank_pct['20日排行_百分位'] - rank_pct['3日排行_百分位']

    rising_threshold = 0.2
    falling_threshold = -0.2
    rising_industries = rank_change[rank_change > rising_threshold].sort_values(ascending=False)
    falling_industries = rank_change[rank_change < falling_threshold].sort_values()

    # 8. 市场整体资金流
    total_flow = flow_df.sum(axis=0)

    # 9. 与市场宽度结合
    width_comment = ""
    if width_data is not None and isinstance(width_data, pd.Series) and not width_data.empty:
        latest_width = width_data.iloc[-1]
        if total_flow['3日排行'] > 0 and latest_width > 0.6:
            width_comment = "资金流入且市场宽度较高（>60%），市场健康上涨，个股普涨。"
        elif total_flow['3日排行'] > 0 and latest_width < 0.4:
            width_comment = "资金流入但宽度较低（<40%），可能为少数权重股拉升，个股普遍下跌，警惕指数失真。"
        elif total_flow['3日排行'] < 0 and latest_width < 0.3:
            width_comment = "资金流出且宽度低迷，市场整体弱势，建议谨慎。"
        elif total_flow['3日排行'] < 0 and latest_width > 0.5:
            width_comment = "资金流出但宽度尚可，可能为结构性调整，需观察。"
        else:
            width_comment = "资金流与宽度无明显协同信号。"

    # 10. 构建分析报告
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("【多周期资金流向深度分析报告】")
    report_lines.append(f"数据时间：最新一期（基于{periods}累计）")
    report_lines.append("")

    report_lines.append("【市场整体资金流】")
    for p in periods:
        direction = "净流入" if total_flow[p] > 0 else "净流出"
        report_lines.append(f"  {p}：{direction} {abs(total_flow[p]):.2f} 亿元")
    report_lines.append("")

    if width_comment:
        report_lines.append("【市场宽度协同】")
        report_lines.append(f"  {width_comment}")
        report_lines.append("")

    strong_trend = slope_df[(slope_df['p值'] < 0.1) & (slope_df['斜率'] > 0)].sort_values('斜率', ascending=False)
    weak_trend = slope_df[(slope_df['p值'] < 0.1) & (slope_df['斜率'] < 0)].sort_values('斜率')
    report_lines.append("【趋势强度】")
    if not strong_trend.empty:
        report_lines.append("  趋势强劲（显著上升）：")
        for idx, row in strong_trend.head(10).iterrows():
            report_lines.append(f"    {idx}：斜率 {row['斜率']:.3f} 亿元/日，R²={row['R²']:.2f}")
    else:
        report_lines.append("  无显著上升趋势行业。")
    if not weak_trend.empty:
        report_lines.append("  趋势显著下降：")
        for idx, row in weak_trend.head(10).iterrows():
            report_lines.append(f"    {idx}：斜率 {row['斜率']:.3f} 亿元/日，R²={row['R²']:.2f}")
    report_lines.append("")

    report_lines.append("【资金加速/减速】")
    acc_cond = (daily_avg_winsor['最近3日'] > daily_avg_winsor['4-5日']) & \
               (daily_avg_winsor['4-5日'] > daily_avg_winsor['6-10日']) & \
               (daily_avg_winsor['6-10日'] > daily_avg_winsor['11-20日'])
    acc_industries = daily_avg_winsor[acc_cond].index.tolist()
    report_lines.append("  资金加速流入行业（各时段日均递增）：")
    if acc_industries:
        report_lines.append(f"    {', '.join(acc_industries[:10])}{'...' if len(acc_industries)>10 else ''}")
    else:
        report_lines.append("    无")

    dec_cond = (daily_avg_winsor['最近3日'] < daily_avg_winsor['4-5日']) & \
               (daily_avg_winsor['4-5日'] < daily_avg_winsor['6-10日']) & \
               (daily_avg_winsor['6-10日'] < daily_avg_winsor['11-20日'])
    dec_industries = daily_avg_winsor[dec_cond].index.tolist()
    report_lines.append("  资金减速流入行业（各时段日均递减）：")
    if dec_industries:
        report_lines.append(f"    {', '.join(dec_industries[:10])}{'...' if len(dec_industries)>10 else ''}")
    else:
        report_lines.append("    无")

    acc_out_cond = (daily_avg_winsor['最近3日'] < daily_avg_winsor['4-5日']) & \
                   (daily_avg_winsor['4-5日'] < daily_avg_winsor['6-10日']) & \
                   (daily_avg_winsor['6-10日'] < daily_avg_winsor['11-20日']) & \
                   (daily_avg_winsor['最近3日'] < 0)
    acc_out_industries = daily_avg_winsor[acc_out_cond].index.tolist()
    report_lines.append("  资金加速流出行业：")
    if acc_out_industries:
        report_lines.append(f"    {', '.join(acc_out_industries[:10])}{'...' if len(acc_out_industries)>10 else ''}")
    else:
        report_lines.append("    无")
    report_lines.append("")

    report_lines.append("【排名变化】")
    report_lines.append(f"  近期（3日vs20日）排名显著上升的行业（百分位提升>{rising_threshold:.0%}）：")
    if not rising_industries.empty:
        for idx, val in rising_industries.head(10).items():
            report_lines.append(f"    {idx}：提升 {val:.1%}")
    else:
        report_lines.append("    无")
    report_lines.append(f"  近期排名显著下降的行业（百分位下降>{abs(falling_threshold):.0%}）：")
    if not falling_industries.empty:
        for idx, val in falling_industries.head(10).items():
            report_lines.append(f"    {idx}：下降 {-val:.1%}")
    else:
        report_lines.append("    无")
    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("注：以上分析基于累计净额折算的日均值，极端值已进行缩尾处理。")
    report_lines.append("斜率回归基于累计值（含原点），p值<0.1视为显著。")

    report = "\n".join(report_lines)
    print(report)

    return flow_df, daily_avg_winsor, report


# 使用示例
if __name__ == "__main__":
    #####初级版资金分析
    analyze_fund_flow_multi_period()

    ####高级版资金分析
    # 如果需要市场宽度数据，可以从你的监控系统中获取
    width_data = None  # 示例：pd.Series(...)
    flow_df, daily_avg, report = analyze_fund_flow_advanced(width_data=width_data)