import pandas as pd
import numpy as np

def analyze_etfs_by_category(
    input_path='data/etf_with_metrics.csv',
    output_path='data/etf_top1_by_category.csv',
    name_col='基金简称',
    category_col='小类',
    ret_col='年化收益率',
    vol_col='年化波动率',
    sharpe_col='夏普比率',
    alpha_col='Alpha',
    beta_col='Beta',
    status_col='计算状态',
    vol_target=(15, 30),
    beta_target=None,
    top_n=1,                       # 改为 1，只取每个类别的第一名
    weights=None
):
    """
    按类别对 ETF 进行综合评分，输出每个类别前 top_n 只。
    """
    # 默认权重
    if weights is None:
        weights = {
            ret_col: 0.20,
            sharpe_col: 0.30,
            alpha_col: 0.20,
            'vol_score': 0.15,
            'beta_score': 0.15
        }

    # 读取数据
    df = pd.read_csv(input_path)
    print(f"原始数据行数: {len(df)}")
    
    # 清洗列名：去除两端空格
    df.columns = df.columns.str.strip()
    print("清洗后的列名：")
    print(df.columns.tolist())

    # 仅保留计算状态为“成功”的 ETF
    df = df[df[status_col] == '成功'].copy()
    print(f"成功计算行数: {len(df)}")

    # 确保必要列存在
    required_cols = [name_col, category_col, ret_col, vol_col, sharpe_col, alpha_col, beta_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"数据缺少必要列: {col}")

    # 剔除缺失值
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    print(f"剔除缺失值后行数: {len(df)}")

    # 定义波动率得分函数
    def vol_score(x):
        low, high = vol_target
        if pd.isna(x):
            return 0
        if low <= x <= high:
            return 1.0
        elif x < low:
            return max(0, 1 - (low - x) / low)
        else:
            return max(0, 1 - (x - high) / high)

    # 定义 Beta 得分函数
    def beta_score(beta, group):
        if beta_target is not None:
            max_dist = group[beta_col].apply(lambda x: abs(x - beta_target)).max()
            if max_dist == 0:
                return 1.0
            return 1 - abs(beta - beta_target) / max_dist
        else:
            min_b = group[beta_col].min()
            max_b = group[beta_col].max()
            if max_b == min_b:
                return 0.5
            return (beta - min_b) / (max_b - min_b)

    # 按类别分组计算得分
    def score_group(group):
        g = group.copy()
        # 正向指标标准化
        for col in [ret_col, sharpe_col, alpha_col]:
            min_val = g[col].min()
            max_val = g[col].max()
            if max_val > min_val:
                g[col + '_score'] = (g[col] - min_val) / (max_val - min_val)
            else:
                g[col + '_score'] = 0.5

        # 波动率得分
        g['vol_score'] = g[vol_col].apply(vol_score)

        # Beta 得分
        g['beta_score'] = g[beta_col].apply(lambda x: beta_score(x, g))

        # 计算总分
        total = 0.0
        for key, w in weights.items():
            if key in [ret_col, sharpe_col, alpha_col]:
                total += w * g[key + '_score']
            elif key == 'vol_score':
                total += w * g['vol_score']
            elif key == 'beta_score':
                total += w * g['beta_score']
            else:
                pass
        g['综合得分'] = total
        return g

    # 使用 group_keys=True 使得分组键成为索引，然后 reset_index 将其转为列
    df_scored = df.groupby(category_col, group_keys=True).apply(score_group).reset_index()
    
    # 删除可能多余的索引列
    extra_cols = ['index', 'level_0']
    for col in extra_cols:
        if col in df_scored.columns:
            df_scored = df_scored.drop(columns=[col])

    print("df_scored 列名:", df_scored.columns.tolist())
    print(f"类别列 '{category_col}' 的值示例:", df_scored[category_col].head(10).tolist())

    # 在每个类别内按综合得分降序排序，并取前 top_n
    top_etfs = df_scored.sort_values([category_col, '综合得分'], ascending=[True, False]) \
                        .groupby(category_col).head(top_n)

    top_etfs = top_etfs.reset_index(drop=True)

    # 保存结果
    top_etfs[['基金代码', '基金简称', '小类', '综合得分']].to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存至 {output_path}")
    print(f"共选出 {len(top_etfs)} 只 ETF，涉及 {top_etfs[category_col].nunique()} 个类别")

    return top_etfs

# ==================== 使用示例 ====================
if __name__ == "__main__":
    result = analyze_etfs_by_category(
        input_path='data/etf_with_metrics.csv',
        output_path='data/etf_top1_by_category.csv',
        name_col='基金简称',
        category_col='小类',
        ret_col='年化收益率',
        vol_col='年化波动率',
        sharpe_col='夏普比率',
        alpha_col='Alpha',
        beta_col='Beta',
        status_col='计算状态',
        vol_target=(15, 30),
        beta_target=None,
        top_n=1,                     # 只取每个类别第一名
        weights={
            '年化收益率': 0.20,
            '夏普比率': 0.30,
            'Alpha': 0.20,
            'vol_score': 0.15,
            'beta_score': 0.15
        }
    )

    print("\n各小类第一名示例：")
    print(result[['基金代码', '基金简称', '小类', '综合得分']].head(10))