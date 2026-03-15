import pandas as pd
import numpy as np
import akshare as ak
import os
import time
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FundFlowTimeSeriesAnalyzer:
    """
    基于时间序列的资金流向分析器
    每天获取行业资金流数据（3日、5日、10日、20日累计），保存到本地CSV，并生成分析报告。
    数据不足时自动跳过滚动计算，不影响程序运行。
    """

    def __init__(self, data_dir='fund_flow_data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.paths = {
            '3日': os.path.join(data_dir, 'flow_3d.csv'),
            '5日': os.path.join(data_dir, 'flow_5d.csv'),
            '10日': os.path.join(data_dir, 'flow_10d.csv'),
            '20日': os.path.join(data_dir, 'flow_20d.csv'),
        }
        self.data = self._load_all_data()

    def _load_all_data(self):
        data = {}
        for period, path in self.paths.items():
            if os.path.exists(path):
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                data[period] = df
            else:
                data[period] = pd.DataFrame()
        return data

    def _save_data(self, period, df):
        path = self.paths[period]
        df.to_csv(path)

    def fetch_today_data(self, date=None):
        if date is None:
            date = datetime.now()
        date_str = date.strftime('%Y%m%d')
        periods = ['3日排行', '5日排行', '10日排行', '20日排行']
        result = {}
        for p in periods:
            try:
                df = ak.stock_fund_flow_industry(symbol=p)
                if '行业' in df.columns and '净额' in df.columns:
                    series = df.set_index('行业')['净额'].astype(float)
                    series = series[~series.index.duplicated(keep='first')]
                    period_key = p.replace('排行', '')
                    result[period_key] = series
                else:
                    print(f"警告：{p} 列名异常，实际列名：{df.columns.tolist()}")
            except Exception as e:
                print(f"获取 {p} 失败：{e}")
        return result

    def update_daily(self, date=None, delay=0.5):
        print("正在获取最新行业资金流向数据...")
        today_data = self.fetch_today_data(date)
        if not today_data:
            print("获取数据失败，跳过更新。")
            return

        for period, series in today_data.items():
            df_new = series.to_frame().T
            df_new.index = [pd.to_datetime(date) if date else datetime.now()]
            df_new.index.name = 'date'

            if period in self.data and not self.data[period].empty:
                df_old = self.data[period]
                combined = pd.concat([df_old, df_new])
                combined = combined[~combined.index.duplicated(keep='last')].sort_index()
                self.data[period] = combined
            else:
                self.data[period] = df_new.sort_index()

            self._save_data(period, self.data[period])

        print(f"数据已更新至 {datetime.now().strftime('%Y-%m-%d')}")

    def _rolling_slope(self, series, window):
        def slope_func(y):
            if len(y) < 2 or np.std(y) == 0:
                return np.nan
            x = np.arange(len(y))
            slope, _, _, _, _ = stats.linregress(x, y)
            return slope
        return series.rolling(window, min_periods=window).apply(slope_func, raw=False)

    def analyze(self, period='3日', window=20, short_window=5, long_window=20, rank_window=5, verbose=False):
        """
        对指定周期（如3日）的资金流时间序列进行分析
        返回 (result_df, report_text)
        """
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            self._analyze_internal(period, window, short_window, long_window, rank_window)
        report_text = f.getvalue()

        result_df = self._get_analyze_result(period, window, short_window, long_window, rank_window)

        if verbose:
            print(report_text)

        return result_df, report_text

    def _analyze_internal(self, period, window, short_window, long_window, rank_window):
        if period not in self.data or self.data[period].empty:
            print(f"没有 {period} 数据，请先更新。")
            return
        df = self.data[period].copy()
        if df.shape[0] < 2:
            print("数据不足，无法分析。")
            return

        latest_date = df.index[-1]
        slope = self._rolling_slope(df, window).loc[latest_date] if df.shape[0] >= window else pd.Series(index=df.columns, dtype=float)
        slope_short = self._rolling_slope(df, short_window).loc[latest_date] if df.shape[0] >= short_window else pd.Series(index=df.columns, dtype=float)
        slope_long = self._rolling_slope(df, long_window).loc[latest_date] if df.shape[0] >= long_window else pd.Series(index=df.columns, dtype=float)
        momentum = slope_short - slope_long
        rank = df.rank(axis=1, ascending=False).astype(int)
        latest_rank = rank.loc[latest_date]
        rank_slope = self._rolling_slope(rank, rank_window).loc[latest_date] if rank.shape[0] >= rank_window else pd.Series(index=df.columns, dtype=float)
        total_flow = df.sum(axis=1)
        total_slope = self._rolling_slope(total_flow.to_frame('total'), window).iloc[-1, 0] if df.shape[0] >= window else np.nan

        if '3日' in self.data and '20日' in self.data and not self.data['3日'].empty and not self.data['20日'].empty:
            df_short = self.data['3日'].reindex(df.index)
            df_long = self.data['20日'].reindex(df.index)
            ratio = df_short / df_long.replace(0, np.nan)
            ratio_latest = ratio.loc[latest_date] if not ratio.empty else pd.Series(index=df.columns, dtype=float)
            ratio_slope = self._rolling_slope(ratio, window).loc[latest_date] if df.shape[0] >= window else pd.Series(index=df.columns, dtype=float)
        else:
            ratio_latest = pd.Series(index=df.columns, dtype=float)
            ratio_slope = pd.Series(index=df.columns, dtype=float)

        temp_result = pd.DataFrame(index=df.columns)
        temp_result['趋势斜率'] = slope
        temp_result['动量'] = momentum
        temp_result['排名趋势'] = rank_slope
        temp_result['最新排名'] = latest_rank
        temp_result['最新值'] = df.loc[latest_date]

        print("\n" + "="*70)
        print(f"【{period}资金流时间序列分析报告】")
        print(f"最新日期: {latest_date}")
        print(f"数据天数: {df.shape[0]}")
        print("\n--- 市场整体 ---")
        print(f"全行业总资金流({period}累计): {total_flow.iloc[-1]:.2f} 亿元")
        if not np.isnan(total_slope):
            print(f"总资金流趋势斜率({window}日): {total_slope:.4f} ({'上升' if total_slope>0 else '下降'})")
        else:
            print(f"总资金流趋势斜率({window}日): 数据不足")

        mask = (temp_result['趋势斜率'] > 0) & (temp_result['动量'] > 0) & (temp_result['排名趋势'] > 0)
        candidates = temp_result[mask].sort_values('动量', ascending=False)
        print("\n--- 资金加速流入且排名上升的行业 ---")
        if not candidates.empty:
            print(candidates[['最新值', '趋势斜率', '动量', '最新排名', '排名趋势']].head(10))
        else:
            print("无符合条件的行业（可能数据不足或无显著信号）")

        mask_out = (temp_result['趋势斜率'] < 0) & (temp_result['动量'] < 0)
        out = temp_result[mask_out].sort_values('动量')
        print("\n--- 资金加速流出行业 ---")
        if not out.empty:
            print(out[['最新值', '趋势斜率', '动量']].head(10))
        else:
            print("无")

        rising = temp_result.nlargest(10, '排名趋势')
        print("\n--- 排名上升最快行业 ---")
        print(rising[['排名趋势', '最新排名']])

        falling = temp_result.nsmallest(10, '排名趋势')
        print("\n--- 排名下降最快行业 ---")
        print(falling[['排名趋势', '最新排名']])

        if not ratio_latest.isna().all():
            temp_result['资金流比率'] = ratio_latest
            high_ratio = temp_result.nlargest(10, '资金流比率')
            low_ratio = temp_result.nsmallest(10, '资金流比率')
            print("\n--- 近期资金占比最高行业 (3日/20日) ---")
            print(high_ratio[['资金流比率', '比率趋势']].head(10))
            print("\n--- 近期资金占比最低行业 ---")
            print(low_ratio[['资金流比率', '比率趋势']].head(10))
        else:
            print("\n资金流比率数据不足（需要3日和20日数据）")
        print("="*70)

    def _get_analyze_result(self, period, window, short_window, long_window, rank_window):
        if period not in self.data or self.data[period].empty:
            return None
        df = self.data[period].copy()
        if df.shape[0] < 2:
            return None

        latest_date = df.index[-1]
        slope = self._rolling_slope(df, window).loc[latest_date] if df.shape[0] >= window else pd.Series(index=df.columns, dtype=float)
        slope_short = self._rolling_slope(df, short_window).loc[latest_date] if df.shape[0] >= short_window else pd.Series(index=df.columns, dtype=float)
        slope_long = self._rolling_slope(df, long_window).loc[latest_date] if df.shape[0] >= long_window else pd.Series(index=df.columns, dtype=float)
        momentum = slope_short - slope_long
        rank = df.rank(axis=1, ascending=False).astype(int)
        latest_rank = rank.loc[latest_date]
        rank_slope = self._rolling_slope(rank, rank_window).loc[latest_date] if rank.shape[0] >= rank_window else pd.Series(index=df.columns, dtype=float)

        if '3日' in self.data and '20日' in self.data and not self.data['3日'].empty and not self.data['20日'].empty:
            df_short = self.data['3日'].reindex(df.index)
            df_long = self.data['20日'].reindex(df.index)
            ratio = df_short / df_long.replace(0, np.nan)
            ratio_latest = ratio.loc[latest_date] if not ratio.empty else pd.Series(index=df.columns, dtype=float)
            ratio_slope = self._rolling_slope(ratio, window).loc[latest_date] if df.shape[0] >= window else pd.Series(index=df.columns, dtype=float)
        else:
            ratio_latest = pd.Series(index=df.columns, dtype=float)
            ratio_slope = pd.Series(index=df.columns, dtype=float)

        result = pd.DataFrame(index=df.columns)
        result['最新值'] = df.loc[latest_date].values
        result[f'趋势斜率({window}日)'] = slope.values
        result[f'短期斜率({short_window}日)'] = slope_short.values
        result[f'长期斜率({long_window}日)'] = slope_long.values
        result['动量(短-长)'] = momentum.values
        result['最新排名'] = latest_rank.values
        result[f'排名趋势({rank_window}日)'] = rank_slope.values
        result['资金流比率(3日/20日)'] = ratio_latest.values
        result['比率趋势'] = ratio_slope.values

        return result