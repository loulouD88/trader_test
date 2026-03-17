"""
style_rotation.py - 风格轮动监控模块
基于ETF分类（价值/成长、周期/防御等）计算类别指数和相对强度
"""

import pandas as pd
import numpy as np

# ================== 分类映射表 ==================
# 根据你提供的分类表构建映射字典
# 格式: { '小类': {'value_growth': '价值'/'成长'/'混合', 
#                 'cycle_defensive': '周期'/'防御'/'中性',
#                 'cap': '大盘'/'中盘'/'小盘'/'混合',
#                 'chain': '上游'/'中游'/'下游'/'-'} }

CLASSIFICATION = {
    # 宽基/策略
    '沪深300': {'value_growth': '混合', 'cycle_defensive': '中性', 'cap': '大盘', 'chain': '-'},
    '中证500': {'value_growth': '混合', 'cycle_defensive': '中性', 'cap': '中盘', 'chain': '-'},
    '中证1000': {'value_growth': '成长', 'cycle_defensive': '中性', 'cap': '小盘', 'chain': '-'},
    '中证2000': {'value_growth': '成长', 'cycle_defensive': '中性', 'cap': '小盘', 'chain': '-'},
    '上证180': {'value_growth': '混合', 'cycle_defensive': '中性', 'cap': '大盘', 'chain': '-'},
    '上证50ETF': {'value_growth': '价值', 'cycle_defensive': '中性', 'cap': '大盘', 'chain': '-'},
    '深证100': {'value_growth': '混合', 'cycle_defensive': '中性', 'cap': '大盘', 'chain': '-'},
    '创业板': {'value_growth': '成长', 'cycle_defensive': '中性', 'cap': '中大盘', 'chain': '-'},
    '科创板': {'value_growth': '成长', 'cycle_defensive': '中性', 'cap': '中盘', 'chain': '-'},
    '中证A50': {'value_growth': '混合', 'cycle_defensive': '中性', 'cap': '大盘', 'chain': '-'},
    'MSCI': {'value_growth': '混合', 'cycle_defensive': '中性', 'cap': '大盘', 'chain': '-'},
    '策略': {'value_growth': '-', 'cycle_defensive': '-', 'cap': '-', 'chain': '-'},
    '红利': {'value_growth': '价值', 'cycle_defensive': '防御', 'cap': '大盘', 'chain': '-'},
    '自由现金流': {'value_growth': '价值', 'cycle_defensive': '防御', 'cap': '混合', 'chain': '-'},

    # 科技成长
    '半导体': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '中游'},
    '芯片': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '中游'},
    '电子': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '中游'},
    '通信': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '中游'},
    '计算机': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '下游'},
    '人工智能': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '下游'},
    '云计算与大数据': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '下游'},
    '互联网': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '大盘', 'chain': '下游'},
    '动漫游戏': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '小盘', 'chain': '下游'},

    # 医药医疗
    '医药': {'value_growth': '成长', 'cycle_defensive': '防御', 'cap': '混合', 'chain': '-'},
    '生物医药': {'value_growth': '成长', 'cycle_defensive': '防御', 'cap': '混合', 'chain': '-'},
    '创新药': {'value_growth': '成长', 'cycle_defensive': '防御', 'cap': '混合', 'chain': '-'},
    '中药': {'value_growth': '价值', 'cycle_defensive': '防御', 'cap': '混合', 'chain': '-'},
    '医疗': {'value_growth': '成长', 'cycle_defensive': '防御', 'cap': '混合', 'chain': '-'},

    # 消费
    '消费': {'value_growth': '混合', 'cycle_defensive': '防御', 'cap': '大盘', 'chain': '下游'},
    '食品饮料': {'value_growth': '价值', 'cycle_defensive': '防御', 'cap': '大盘', 'chain': '下游'},
    '家用电器': {'value_growth': '价值', 'cycle_defensive': '周期', 'cap': '大盘', 'chain': '下游'},
    '汽车': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '下游'},
    '汽车零部件': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '小盘', 'chain': '中游'},
    '养殖': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '小盘', 'chain': '上游'},
    '农业': {'value_growth': '混合', 'cycle_defensive': '防御', 'cap': '小盘', 'chain': '上游'},

    # 周期
    '有色': {'value_growth': '价值', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '上游'},
    '钢铁': {'value_growth': '价值', 'cycle_defensive': '周期', 'cap': '大盘', 'chain': '上游'},
    '煤炭': {'value_growth': '价值', 'cycle_defensive': '周期', 'cap': '大盘', 'chain': '上游'},
    '化工': {'value_growth': '混合', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '中游'},
    '稀土': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '小盘', 'chain': '上游'},
    '能源': {'value_growth': '价值', 'cycle_defensive': '周期', 'cap': '大盘', 'chain': '上游'},

    # 新能源
    '新能源': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '中游'},
    '光伏': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '中游'},
    '风电': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '中游'},
    '电池': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '中游'},
    '电网设备': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '中游'},
    '新能源汽车': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '下游'},
    '低碳': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '-'},

    # 金融地产
    '金融': {'value_growth': '价值', 'cycle_defensive': '周期', 'cap': '大盘', 'chain': '-'},
    '银行': {'value_growth': '价值', 'cycle_defensive': '周期', 'cap': '大盘', 'chain': '-'},
    '非银': {'value_growth': '价值', 'cycle_defensive': '周期', 'cap': '大盘', 'chain': '-'},
    '证券': {'value_growth': '价值', 'cycle_defensive': '周期', 'cap': '大盘', 'chain': '-'},
    '保险': {'value_growth': '价值', 'cycle_defensive': '周期', 'cap': '大盘', 'chain': '-'},
    '房地产': {'value_growth': '价值', 'cycle_defensive': '周期', 'cap': '大盘', 'chain': '-'},
    '基建': {'value_growth': '价值', 'cycle_defensive': '周期', 'cap': '大盘', 'chain': '-'},

    # 高端制造
    '军工': {'value_growth': '成长', 'cycle_defensive': '防御', 'cap': '混合', 'chain': '-'},
    '航天': {'value_growth': '成长', 'cycle_defensive': '防御', 'cap': '混合', 'chain': '-'},
    '工程机械': {'value_growth': '价值', 'cycle_defensive': '周期', 'cap': '大盘', 'chain': '中游'},
    '工业': {'value_growth': '混合', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '中游'},

    # 其他
    '黄金': {'value_growth': '价值', 'cycle_defensive': '防御', 'cap': '-', 'chain': '上游'},
    '商品': {'value_growth': '价值', 'cycle_defensive': '周期', 'cap': '-', 'chain': '上游'},
    'QDII': {'value_growth': '-', 'cycle_defensive': '-', 'cap': '-', 'chain': '-'},
    '海外': {'value_growth': '-', 'cycle_defensive': '-', 'cap': '-', 'chain': '-'},
    '一带一路': {'value_growth': '价值', 'cycle_defensive': '周期', 'cap': '大盘', 'chain': '-'},
    '国央企': {'value_growth': '价值', 'cycle_defensive': '中性', 'cap': '大盘', 'chain': '-'},
    '高股息': {'value_growth': '价值', 'cycle_defensive': '防御', 'cap': '大盘', 'chain': '-'},
    '环保': {'value_growth': '成长', 'cycle_defensive': '防御', 'cap': '小盘', 'chain': '-'},
    '电力': {'value_growth': '价值', 'cycle_defensive': '防御', 'cap': '大盘', 'chain': '-'},
    '传媒': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '小盘', 'chain': '下游'},
    '材料': {'value_growth': '混合', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '上游'},
    '创新': {'value_growth': '成长', 'cycle_defensive': '周期', 'cap': '混合', 'chain': '-'},
}

# ================== 风格轮动监控函数 ==================
def style_rotation_monitor(price_df, etf_info, window=5, long_window=20):
    """
    计算风格轮动指标
    参数:
        price_df: DataFrame, 索引为日期，列为基金代码，值为收盘价
        etf_info: DataFrame, 包含 '基金代码', '基金简称', '小类' 等列
        window: 短期窗口（用于计算动量）
        long_window: 长期窗口（用于计算趋势）
    返回:
        style_data: dict, 包含各类别指数、相对强度、趋势等
        style_df: DataFrame, 适合写入Excel的汇总表
    """
    # 确保索引为日期
    price_df = price_df.copy()
    
    # 建立基金代码到小类的映射
    code_to_subclass = etf_info.set_index('基金代码')['小类'].to_dict()
    # 建立小类到分类的映射（只取我们关心的维度）
    # 我们主要关注价值/成长、周期/防御、大盘/小盘、上游/下游
    # 为简化，先计算价值/成长和周期/防御两个关键维度
    code_to_vg = {}
    code_to_cd = {}
    code_to_cap = {}
    code_to_chain = {}
    for code, sub in code_to_subclass.items():
        if sub in CLASSIFICATION:
            code_to_vg[code] = CLASSIFICATION[sub]['value_growth']
            code_to_cd[code] = CLASSIFICATION[sub]['cycle_defensive']
            code_to_cap[code] = CLASSIFICATION[sub]['cap']
            code_to_chain[code] = CLASSIFICATION[sub]['chain']
        else:
            # 未分类的ETF忽略
            continue

    # 筛选出有分类的ETF代码
    valid_codes = [c for c in price_df.columns if c in code_to_vg]
    if not valid_codes:
        print("警告：没有找到任何已分类的ETF")
        return None, None

    price_df = price_df[valid_codes]

    # 定义需要监控的风格类别
    style_groups = {
        '价值': [c for c in valid_codes if code_to_vg[c] == '价值'],
        '成长': [c for c in valid_codes if code_to_vg[c] == '成长'],
        '周期': [c for c in valid_codes if code_to_cd[c] == '周期'],
        '防御': [c for c in valid_codes if code_to_cd[c] == '防御'],
        '大盘': [c for c in valid_codes if code_to_cap[c] == '大盘'],
        '小盘': [c for c in valid_codes if code_to_cap[c] == '小盘'],
        '上游': [c for c in valid_codes if code_to_chain[c] == '上游'],
        '中游': [c for c in valid_codes if code_to_chain[c] == '中游'],
        '下游': [c for c in valid_codes if code_to_chain[c] == '下游'],
    }

    # 计算每个类别的等权平均收益率
    daily_ret = price_df.pct_change()
    style_ret = {}
    for name, codes in style_groups.items():
        if len(codes) > 0:
            # 等权平均收益率
            style_ret[name] = daily_ret[codes].mean(axis=1)
        else:
            style_ret[name] = pd.Series(index=daily_ret.index, dtype=float)

    # 将收益率序列转为价格指数（以第一个非空值为基准）
    style_index = {}
    for name, ret in style_ret.items():
        if ret.dropna().empty:
            style_index[name] = pd.Series(index=ret.index, dtype=float)
        else:
            # 从第一个有效值开始计算指数
            first_valid = ret.first_valid_index()
            idx = ret.loc[first_valid:].index
            cum_ret = (1 + ret.loc[first_valid:]).cumprod()
            style_index[name] = cum_ret.reindex(ret.index, method='ffill')

    # 计算关键相对强度
    # 价值/成长比值
    if '价值' in style_index and '成长' in style_index:
        vg_ratio = style_index['价值'] / style_index['成长']
    else:
        vg_ratio = pd.Series(index=price_df.index, dtype=float)

    # 周期/防御比值
    if '周期' in style_index and '防御' in style_index:
        cd_ratio = style_index['周期'] / style_index['防御']
    else:
        cd_ratio = pd.Series(index=price_df.index, dtype=float)

    # 大盘/小盘比值
    if '大盘' in style_index and '小盘' in style_index:
        lsc_ratio = style_index['大盘'] / style_index['小盘']
    else:
        lsc_ratio = pd.Series(index=price_df.index, dtype=float)

    # 上游/下游比值
    if '上游' in style_index and '下游' in style_index:
        ud_ratio = style_index['上游'] / style_index['下游']
    else:
        ud_ratio = pd.Series(index=price_df.index, dtype=float)

    # 计算比值的短期趋势（斜率）
    def slope(series, win):
        if len(series) < win:
            return np.nan
        y = series.iloc[-win:].values
        x = np.arange(len(y))
        if np.std(y) == 0:
            return 0
        slope_val, _ = np.polyfit(x, y, 1)
        return slope_val

    latest_date = price_df.index[-1]

    # 构建结果DataFrame（最新一期）
    result = {}
    for name, idx_series in style_index.items():
        if not idx_series.empty and latest_date in idx_series.index:
            result[f'{name}指数'] = idx_series.loc[latest_date]
        else:
            result[f'{name}指数'] = np.nan

    # 相对强度最新值及趋势
    ratios = {
        '价值/成长': vg_ratio,
        '周期/防御': cd_ratio,
        '大盘/小盘': lsc_ratio,
        '上游/下游': ud_ratio
    }
    for name, ratio_series in ratios.items():
        if not ratio_series.empty and latest_date in ratio_series.index:
            result[f'{name}'] = ratio_series.loc[latest_date]
            result[f'{name}_短期斜率'] = slope(ratio_series, window)
            result[f'{name}_长期斜率'] = slope(ratio_series, long_window)
        else:
            result[f'{name}'] = np.nan
            result[f'{name}_短期斜率'] = np.nan
            result[f'{name}_长期斜率'] = np.nan

    # 生成风格偏好结论
    conclusion = []
    if not np.isnan(result.get('价值/成长', np.nan)):
        if result['价值/成长_短期斜率'] > 0.001:
            conclusion.append("价值相对走强")
        elif result['价值/成长_短期斜率'] < -0.001:
            conclusion.append("成长相对走强")
    if not np.isnan(result.get('周期/防御', np.nan)):
        if result['周期/防御_短期斜率'] > 0.001:
            conclusion.append("周期相对走强")
        elif result['周期/防御_短期斜率'] < -0.001:
            conclusion.append("防御相对走强")
    if not np.isnan(result.get('大盘/小盘', np.nan)):
        if result['大盘/小盘_短期斜率'] > 0.001:
            conclusion.append("大盘相对走强")
        elif result['大盘/小盘_短期斜率'] < -0.001:
            conclusion.append("小盘相对走强")
    if not np.isnan(result.get('上游/下游', np.nan)):
        if result['上游/下游_短期斜率'] > 0.001:
            conclusion.append("上游相对走强")
        elif result['上游/下游_短期斜率'] < -0.001:
            conclusion.append("下游相对走强")

    result['风格结论'] = '；'.join(conclusion) if conclusion else '无明显风格'

    # 转换为DataFrame（一行）
    result_df = pd.DataFrame([result])
    result_df.insert(0, '最新日期', latest_date)

    return result_df, style_index