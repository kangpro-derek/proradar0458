import os
import pandas as pd
from utils.backtest import get_price_data, run_simple_ttl_backtest

# ✅ 전략별 분할 비율
PRO1_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25]
PRO2_WEIGHTS = [0.10, 0.15, 0.20, 0.25, 0.20, 0.10]
PRO3_WEIGHTS = [1/6] * 6

# ✅ 설정
symbol = "SOXL"
start_year = 2011
end_year = 2024
cache_path = f"cache/statistics_yearly.csv"

def run_or_load_yearly_statistics():
    # ✅ 캐시 파일이 존재하면 바로 로드
    if os.path.exists(cache_path):
        print(f"📁 캐시 파일 사용: {cache_path}")
        return pd.read_csv(cache_path)

    print("🚀 연도별 백테스트 실행 중...")

    # ✅ 전체 가격 데이터 가져오기
    df = get_price_data(symbol, f"{start_year}-01-01", f"{end_year}-12-31")

    # ✅ 연도별 백테스트 결과 저장
    results = []
    for year in range(start_year, end_year + 1):
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        year_df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        year_df = year_df.reset_index(drop=True)  # ✅ 정수 인덱스 보장

        if year_df.empty:
            continue

        r1 = run_simple_ttl_backtest(year_df, PRO1_WEIGHTS)
        r2 = run_simple_ttl_backtest(year_df, PRO2_WEIGHTS)
        r3 = run_simple_ttl_backtest(year_df, PRO3_WEIGHTS)

        results.append({
            "연도": year,
            "Pro1_수익률": r1["수익률"], "Pro1_MDD": r1["MDD"],
            "Pro2_수익률": r2["수익률"], "Pro2_MDD": r2["MDD"],
            "Pro3_수익률": r3["수익률"], "Pro3_MDD": r3["MDD"],
        })

    yearly_df = pd.DataFrame(results)
    yearly_df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"✅ 연도별 백테스트 결과 저장 완료: {cache_path}")
    return yearly_df
