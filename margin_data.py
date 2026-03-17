import pandas as pd
import numpy as np
import akshare as ak
import os
import time
from datetime import datetime, timedelta
from typing import Optional

class MarginDataUpdater:
    """
    融资余额数据更新器（带缓存，防止数据丢失）
    """

    def __init__(self, csv_path: str = "data/margin_data.csv", sz_raw_path: str = "data/margin_sz_raw.csv"):
        self.csv_path = csv_path
        self.sz_raw_path = sz_raw_path
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

    def _load_sz_raw(self) -> pd.DataFrame:
        """加载深交所原始缓存数据"""
        if os.path.exists(self.sz_raw_path):
            df = pd.read_csv(self.sz_raw_path, parse_dates=['date'])
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
            return df
        return pd.DataFrame(columns=['date', 'fin_balance_sz'])

    def _save_sz_raw(self, df: pd.DataFrame):
        """保存深交所原始数据（覆盖）"""
        df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
        df.to_csv(self.sz_raw_path, index=False)

    def _append_sz_raw(self, new_records: pd.DataFrame):
        """追加新的深交所记录到缓存文件"""
        if new_records.empty:
            return
        existing = self._load_sz_raw()
        combined = pd.concat([existing, new_records], ignore_index=True)
        combined = combined.drop_duplicates('date').sort_values('date').reset_index(drop=True)
        combined.to_csv(self.sz_raw_path, index=False)

    def _fetch_sh_margin(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取上交所融资余额（区间）"""
        try:
            df = ak.stock_margin_sse(start_date=start_date, end_date=end_date)
            if df.empty:
                return pd.DataFrame()
            df = df.rename(columns={'信用交易日期': 'date', '融资余额': 'fin_balance_sh'})
            df['date'] = pd.to_datetime(df['date'])
            df['fin_balance_sh'] = df['fin_balance_sh'] / 1e8
            df = df[['date', 'fin_balance_sh']].drop_duplicates('date').sort_values('date')
            return df
        except Exception as e:
            print(f"获取上交所数据失败: {e}")
            return pd.DataFrame()

    def _fetch_sz_missing(self, target_dates, delay=0.5, max_retries=3):
        """
        批量获取缺失的深交所数据，并立即存入缓存（带重试机制）
        """
        if not target_dates:
            return
        existing = self._load_sz_raw()
        existing_dates = set(existing['date'].dt.date) if not existing.empty else set()

        to_fetch = [dt for dt in target_dates if dt.date() not in existing_dates]
        if not to_fetch:
            print("  深交所数据已全部存在，无需获取")
            return

        total = len(to_fetch)
        new_records = []
        print(f"  需要获取 {total} 个新日期...")
        for i, dt in enumerate(to_fetch):
            date_str = dt.strftime('%Y%m%d')
            for attempt in range(max_retries):
                try:
                    df = ak.stock_margin_szse(date=date_str)
                    if not df.empty and '融资余额' in df.columns:
                        balance = float(df.iloc[0]['融资余额'])
                        new_records.append({'date': dt, 'fin_balance_sz': balance})
                        break
                    else:
                        new_records.append({'date': dt, 'fin_balance_sz': np.nan})
                        break
                except Exception as e:
                    print(f"  深交所 {date_str} 获取失败 (尝试 {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        new_records.append({'date': dt, 'fin_balance_sz': np.nan})
                        print(f"  深交所 {date_str} 最终失败，记录为NaN")
                    else:
                        time.sleep(delay * 2)
            if i % 10 == 0:
                print(f"  深交所进度: {i}/{total}")
            time.sleep(delay)

        if new_records:
            new_df = pd.DataFrame(new_records)
            print(f"  准备追加 {len(new_df)} 条记录，其中NaN数量: {new_df['fin_balance_sz'].isna().sum()}")
            self._append_sz_raw(new_df)
            print(f"  已追加 {len(new_df)} 条新记录到 {self.sz_raw_path}")

    def update(self, start_date: Optional[str] = None, end_date: Optional[str] = None, delay_sz: float = 0.5):
        """
        更新融资余额数据（带缓存）
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = '20180101'
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        print("获取上交所数据...")
        sh_df = self._fetch_sh_margin(start_date, end_date)
        if sh_df.empty:
            raise ValueError("无法获取上交所数据，程序终止")
        print(f"  上交所数据行数: {len(sh_df)}")

        print("深交所数据检查...")
        self._fetch_sz_missing(sh_df['date'].tolist(), delay=delay_sz)

        sz_raw = self._load_sz_raw()
        sz_raw = sz_raw[(sz_raw['date'] >= start_dt) & (sz_raw['date'] <= end_dt)]

        print(f"合并前 sh_df 行数: {len(sh_df)}, sz_raw 行数: {len(sz_raw)}")
        merged = pd.merge(sh_df, sz_raw, on='date', how='outer').sort_values('date').reset_index(drop=True)
        merged = merged.ffill().fillna(0)
        merged['total_balance'] = merged['fin_balance_sh'] + merged['fin_balance_sz']
        print(f"合并后总行数: {len(merged)}")
        self._save_data(merged)
        print("最终数据已保存")

    def _save_data(self, df: pd.DataFrame):
        df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)
        df.to_csv(self.csv_path, index=False)
        print(f"合并数据已保存至 {self.csv_path}, 共 {len(df)} 行")

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"数据文件 {self.csv_path} 不存在，请先运行 update()")
        return pd.read_csv(self.csv_path, parse_dates=['date'])


def sentiment_from_margin_file(csv_path: str = "data/margin_data.csv", years: int = 1) -> dict:
    """从本地文件读取融资余额数据，计算情绪指标"""
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    if df.empty:
        return {}

    cutoff = df['date'].max() - timedelta(days=int(years*365))
    recent = df[df['date'] >= cutoff].copy()
    if recent.empty:
        recent = df

    latest = recent.iloc[-1]
    total = latest['total_balance']
    quantile = (recent['total_balance'] < total).sum() / len(recent)

    if len(recent) >= 6:
        prev = recent.iloc[-6]['total_balance']
        change_5d = (total / prev) - 1
    else:
        change_5d = np.nan

    score = 0
    if quantile < 0.2:
        score -= 1
    elif quantile > 0.8:
        score += 1
    if not np.isnan(change_5d):
        if change_5d > 0.03:
            score += 1
        elif change_5d < -0.03:
            score -= 1

    if score >= 2:
        conclusion = "偏乐观（融资余额高位或快速上升）"
    elif score >= 1:
        conclusion = "略偏乐观"
    elif score <= -2:
        conclusion = "偏悲观（融资余额低位或快速下降）"
    elif score <= -1:
        conclusion = "略偏悲观"
    else:
        conclusion = "中性"

    return {
        '最新日期': latest['date'].strftime('%Y-%m-%d'),
        '融资余额_合计(亿元)': round(total, 2),
        '融资余额_分位数(最近{}年)'.format(years): round(quantile, 2),
        '融资余额_5日变化': round(change_5d, 4) if not np.isnan(change_5d) else None,
        '情绪总分': score,
        '情绪结论': conclusion
    }


if __name__ == "__main__":
    updater = MarginDataUpdater()
    updater.update(start_date="20180101", delay_sz=0.5)
    sentiment = sentiment_from_margin_file("data/margin_data.csv", years=1)
    print("\n========== 融资余额情绪分析 ==========")
    for k, v in sentiment.items():
        print(f"{k}: {v}")