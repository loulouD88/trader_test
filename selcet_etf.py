import time
import os
import pandas as pd
import akshare as ak
from playwright.sync_api import sync_playwright
import re

# 1. 获取所有场内 ETF 的基本信息
etf_df = ak.fund_etf_fund_daily_em()
etf_df = etf_df[['基金代码', '基金简称']]

results = []

# 2. 使用 Playwright 进行无头浏览器抓取（无需系统级 Chrome 安装）
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # 3. 遍历所有 ETF 页面
    for idx, row in etf_df.iterrows():
        code = row['基金代码']
        name = row['基金简称']
        url = f'https://fundf10.eastmoney.com/cyrjg_{code}.html'

        try:
            page.goto(url, timeout=30000)
            page.wait_for_timeout(1500)  # 等待 JS 加载

            # 提取资产规模
            guimo = page.locator('.bs_gl').inner_text().strip()
            clean_text = guimo.replace(',', '')
            a = re.search(r'资产规模：\s*([\d.]+)', clean_text)
            b = re.search(r'类型：\s*([\u4e00-\u9fff]+)', clean_text)
            if a:
                scale = float(a.group(1))
                if scale <= 2.0:
                    print(f'⏭️ 跳过 {code}：资产规模过小：{scale} 亿')
                    continue
            else:
                print(f'❌ {code} 未找到资产规模')
                continue

            # 提取持有人结构
            summary_text = page.locator('#summary').inner_text().strip()
            percentages = re.findall(r"\d+\.\d+%", summary_text)
            if len(percentages) < 2:
                raise ValueError('百分比数据不足')

            inst_ratio = float(percentages[0].strip('%'))
            person_ratio = float(percentages[1].strip('%'))

            # 只保留机构 > 个人 的 ETF
            if inst_ratio > person_ratio:
                results.append({
                    '基金代码': code,
                    '基金简称': name,
                    '类型': b.group(1),
                    '机构持有比例': f"{inst_ratio:.2f}%",
                    '个人持有比例': f"{person_ratio:.2f}%",
                    '资产规模': f"{scale:.2f} 亿",
                    '原始摘要': summary_text
                })
                print(f'✅ {code} 筛选通过')

            page.wait_for_timeout(300)

        except Exception as e:
            print(f'❌ {code} 失败: {e}')
            continue

    browser.close()

# 4. 保存结果
df_result = pd.DataFrame(results)

# 确保输出目录存在
os.makedirs('data', exist_ok=True)
df_result.to_csv('data/etf_list.csv', index=False, encoding='utf-8-sig')
print("🎉 已筛选并保存机构持有比例更高的 ETF 到 data/etf_list.csv")