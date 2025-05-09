# utils/backtest.py
# 📌 TTL 전략 백테스트용 클래스와 함수 정의

import requests
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
import pytz
import os
import pandas as pd
import pandas as pd
import os
from utils.recommend import calculate_roc

TIINGO_API_KEY = "87f8380c68d1ae98523f90aa602a84b12a9e7483"  # 👉 Tiingo 웹사이트에서 발급받은 키 입력

# API_KEYS = ["ONB35B97BRJ6G3T8", "AP68Y6LGDXSHYQEP"]
# # API 키 순차적으로 사용하기 위한 인덱스
# api_key_index = 0

# 한국 시간 설정 (UTC+9)
KST = pytz.timezone('Asia/Seoul')

# CALL_COUNT = 0
# MAX_CALLS_PER_DAY = 490  # Alpha Vantage 무료 플랜 기준 (여유 포함)

# def get_next_api_key():
#     global api_key_index
#     api_key_index = (api_key_index + 1) % len(API_KEYS)
#     return API_KEYS[api_key_index]

# def check_api_quota():
#     global CALL_COUNT
#     CALL_COUNT += 1
#     print(f"📈 API 호출 카운트: {CALL_COUNT}/{MAX_CALLS_PER_DAY}")
#     if CALL_COUNT > MAX_CALLS_PER_DAY:
#         raise RuntimeError("📛 Alpha Vantage 일일 호출 한도 초과. 내일 다시 시도하세요.")
        
def is_range_cached(start_dt, end_dt, df):
    if "date" not in df.columns or df.empty:
        return False
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return False
    min_cached = df["date"].min()
    max_cached = df["date"].max()

    # 🔍 로그로 확인
    print(f"🔎 캐시 범위: {min_cached.date()} ~ {max_cached.date()}")
    print(f"📅 요청 범위: {start_dt.date()} ~ {end_dt.date()}")

    return start_dt >= min_cached and end_dt <= max_cached
    
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_price_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = os.path.join(cache_dir, f"{symbol}.csv")
    last_download_file = os.path.join(cache_dir, f"{symbol}_last_downloaded.txt")

    # ✅ 캐시 로드
    if os.path.exists(cache_file):
        cached_df = pd.read_csv(cache_file, parse_dates=["date"])
        cached_df["date"] = pd.to_datetime(cached_df["date"]).dt.date  # tz 제거 및 .date로 통일
    else:
        cached_df = pd.DataFrame(columns=["date", "close", "open", "high", "low", "volume"])

    # ✅ 마지막 다운로드 날짜 로드
    if os.path.exists(last_download_file):
        with open(last_download_file, "r") as f:
            last_downloaded_date = pd.to_datetime(f.read().strip()).date()
    else:
        last_downloaded_date = None

    # ✅ 요청 범위 처리
    start_dt = pd.to_datetime(start).date()
    end_dt = pd.to_datetime(end).date()

    # ✅ 캐시 범위 포함 여부 판단
    if not cached_df.empty:
        min_cached = min(cached_df["date"])
        max_cached = max(cached_df["date"])
        range_cached = (start_dt >= min_cached) and (end_dt <= max_cached)
    else:
        range_cached = False

    # ✅ 디버깅 출력
    print(f"🔍 range_cached = {range_cached}")
    print(f"📅 last_downloaded_date = {last_downloaded_date}")
    print(f"📅 end_dt = {end_dt}")

    # ✅ 다운로드 조건: 캐시 범위 부족 + 마지막 다운로드 기록 미달
    if not range_cached and (last_downloaded_date is None or last_downloaded_date != end_dt):
        print("🌐 Tiingo에서 데이터 다운로드")

        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        headers = {"Content-Type": "application/json"}
        params = {
            "token": TIINGO_API_KEY,
            "startDate": "2010-01-01",  # 전체 데이터 요청
            "resampleFreq": "daily"
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"❌ Tiingo 다운로드 실패: {e}")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"]).dt.date  # tz 제거 후 date 타입 변환
                        
        # adjClose가 없는지 확인
        if "adjClose" not in df.columns or df["adjClose"].isnull().all():
            print(f"⚠️ {symbol} 은 adjClose(조정 종가)를 제공하지 않습니다.")

        df = df.rename(columns={
            "adjClose": "close",
            "adjOpen": "open",
            "adjHigh": "high",
            "adjLow": "low",
            "adjVolume": "volume"
        })
        
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        df = df[["date", "close", "open", "high", "low", "volume"]].sort_values("date")
        
        # 병합 전에 열 이름 정리
        expected_columns = ["date", "close", "open", "high", "low", "volume"]

        # 필터링하여 필요한 열만 유지
        df = df[[col for col in expected_columns if col in df.columns]]
        cached_df = cached_df[[col for col in expected_columns if col in cached_df.columns]]

        # 열 이름 강제 설정 (혹시라도 NaN 있을 경우 대비)
        df.columns = expected_columns[:df.shape[1]]
        cached_df.columns = expected_columns[:cached_df.shape[1]]
        
        # ✅ 캐시 병합 및 저장
        cached_df = pd.concat([cached_df, df], ignore_index=True)
        cached_df = cached_df.drop_duplicates(subset="date").sort_values("date")
        cached_df.to_csv(cache_file, index=False)

        # ✅ 다운로드 기록 갱신
        with open(last_download_file, "w") as f:
            f.write(str(end_dt))

        print(f"✅ 병합된 캐시 저장 완료: {cache_file}")

    # ✅ 요청 구간 필터링
    result_df = cached_df[
        (cached_df["date"] >= start_dt) & (cached_df["date"] <= end_dt)
    ].copy()

    # ✅ 기술적 지표 계산
    result_df['ma20'] = result_df['close'].rolling(window=20).mean()
    result_df['ma60'] = result_df['close'].rolling(window=60).mean()
    result_df["기울기"] = ((result_df["ma20"] - result_df["ma20"].shift(10)) / result_df["ma20"].shift(10)) * 100
    result_df["정배열"] = (result_df["ma20"] > result_df["ma60"]).astype(int)
    result_df["이격도"] = (result_df["close"] / result_df["ma20"] - 1) * 100
    result_df["수익률"] = result_df["close"].pct_change()
    result_df["변동성"] = result_df["수익률"].rolling(window=20).std() * (20 ** 0.5)
    result_df["ROC"] = calculate_roc(result_df["close"], period=12)
    result_df["RSI"] = calculate_rsi(result_df["close"])

    return result_df.reset_index(drop=True)



def log_backtest_debug(name, df, result, initial_capital, buy_discount, sell_premium):
    print(f"\n📘 [디버그: {name}]")
    print(f"  - 시작일: {df['date'].iloc[0].date()}, 종료일: {df['date'].iloc[-1].date()}")
    print(f"  - 초기 자본: ${initial_capital:,.0f}")
    print(f"  - 매수 기준 (discount): {buy_discount * 100:.2f}%, 매도 기준 (premium): {sell_premium * 100:.2f}%")
    print(f"  - 수익률: {result['수익률']}%, MDD: {result['MDD']}%")


# ✅ TierPosition 클래스 정의
class TierPosition:
    def __init__(self, tier_index, weight, capital):
        self.tier_index = tier_index           # 티어 번호
        self.weight = weight                   # 비중 (가중치)
        self.initial_cash = capital            # 초기 자금
        self.cash = capital                    # 현재 보유 현금
        self.reset()

    def reset(self):
        self.active = False                    # 현재 포지션 보유 여부
        self.buy_price = None                  # 매수 가격
        self.shares = 0                        # 보유 주식 수
        self.hold_days = 0                     # 보유 일수
        self.entry_date = None                 # 매수 날짜

    def try_buy(self, trigger_price, actual_close, date):
        trigger_price = round(trigger_price, 2)
        actual_close = round(actual_close, 2)
        if actual_close > trigger_price:
            return False
        # shares = int(self.cash // actual_close)
        shares = int(self.initial_cash // trigger_price)
        if shares == 0:
            return False
        cost = shares * actual_close
        self.active = True
        self.buy_price = actual_close
        self.shares = shares
        self.hold_days = 0
        self.entry_date = date
        self.cash -= cost
        return True

    def update_holding(self):
        if self.active:
            self.hold_days += 1

    def try_sell(self, price, date, sell_premium):
        price = round(price, 2)
        if self.active and price >= round(self.buy_price * (1 + sell_premium), 2):
            proceeds = self.shares * price
            self.cash += proceeds
            self.reset()
            return True
        return False

    def force_sell(self, price, date):
        price = round(price, 2)
        if self.active:
            proceeds = self.shares * price
            self.cash += proceeds
            self.reset()
            return True
        return False

# ✅ TTL 전략 실행 함수
def run_simple_ttl_backtest(df, tier_weights, initial_capital=10000, hold_days_limit=10,
                            buy_discount=0.0001, sell_premium=0.0001):
    TIERS = len(tier_weights)
    tier_initials = [initial_capital * w for w in tier_weights]
    tier_positions = [TierPosition(i, tier_weights[i], tier_initials[i]) for i in range(TIERS)]

    high_watermark = initial_capital
    records = []

    for i in range(len(df)):
        today = df.loc[i, "date"]
        close = round(df.loc[i, "close"], 2)
        prev_close = round(df.loc[i - 1, "close"], 2) if i > 0 else close
        did_profit_sell_today = False

        for pos in tier_positions:
            if pos.active:
                if pos.try_sell(close, today, sell_premium):
                    did_profit_sell_today = True
                else:
                    pos.update_holding()

        for pos in tier_positions:
            if pos.active and pos.hold_days >= hold_days_limit:
                pos.force_sell(close, today)

        if i > 0 and not did_profit_sell_today:
            for pos in tier_positions:
                if not pos.active:
                    target_price = round(prev_close * (1 - buy_discount), 2)
                    if pos.try_buy(target_price, close, today):
                        break

        valuation = sum([pos.shares * close for pos in tier_positions])
        total_cash = sum([pos.cash for pos in tier_positions])
        total_value = valuation + total_cash
        high_watermark = max(high_watermark, total_value)
        drawdown = (total_value - high_watermark) / high_watermark

        records.append({
            "date": today,
            "portfolio_value": total_value,
            "drawdown": drawdown
        })
        
        # # ✅ 상태 로그 출력 (매일)
        # tier_statuses = [f"티어{pos.tier_index}: ${round(pos.cash + pos.shares * close)}" for pos in tier_positions]
        # tier_summary = ", ".join(tier_statuses)
        # print(f"📅 {today.date()} | {tier_summary} | 총 자산: ${round(total_value)} | DD: {round(drawdown * 100, 2)}%")


    start_value = records[0]["portfolio_value"]
    end_value = records[-1]["portfolio_value"]
    max_drawdown = min([r["drawdown"] for r in records])

    # # ✅ 로그 출력
    # start_date = df["date"].iloc[0]
    # end_date = df["date"].iloc[-1]
    # print(f"🧪 [백테스트 실행 로그] 기간: {start_date.date()} ~ {end_date.date()}, 수익률: {round((end_value - start_value) / start_value * 100, 2)}%")

    return {
        "수익률": round((end_value - start_value) / start_value * 100, 1),
        "MDD": round(max_drawdown * 100, 1),
        "기록": records  # 나중에 시각화용
    }

from datetime import timedelta
import pandas as pd

def run_one_rolling_test(price_df, all_dates, i, test_days,
                         PRO1_WEIGHTS, PRO2_WEIGHTS, PRO3_WEIGHTS,
                         run_simple_ttl_backtest):

    test_start = all_dates[i]
    test_end_est = test_start + timedelta(days=test_days - 1)
    test_end_index = price_df.index.get_indexer([test_end_est], method='bfill')[0]
    if test_end_index == -1:
        return None
    test_end = price_df.index[test_end_index]

    ma_window = 60
    extended_start_index = max(0, i - ma_window)
    extended_range = price_df.iloc[extended_start_index:test_end_index + 1]
    test_range = price_df.loc[test_start:test_end]

    extended_range = extended_range.reset_index()
    test_df = test_range.reset_index()

    result1 = run_simple_ttl_backtest(test_df, PRO1_WEIGHTS)
    result2 = run_simple_ttl_backtest(test_df, PRO2_WEIGHTS)
    result3 = run_simple_ttl_backtest(test_df, PRO3_WEIGHTS)

    scores = {
        "Pro1": result1["수익률"] - 0.75 * abs(result1["MDD"]),
        "Pro2": result2["수익률"] - 0.75 * abs(result2["MDD"]),
        "Pro3": result3["수익률"] - 0.75 * abs(result3["MDD"])
    }

    best_strategy = max(scores, key=scores.get)

    return {
        "시작일": test_start.date(),
        "종료일": test_end.date(),
        "Pro1_수익률": result1["수익률"],
        "Pro1_mdd": result1["MDD"],
        "Pro2_수익률": result2["수익률"],
        "Pro2_mdd": result2["MDD"],
        "Pro3_수익률": result3["수익률"],
        "Pro3_mdd": result3["MDD"],
        "우수한전략": best_strategy
    }

def run_daily_rolling_backtest(price_df: pd.DataFrame, start_date: str, test_days: int):
    from .backtest import run_simple_ttl_backtest

    PRO1_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25]
    PRO2_WEIGHTS = [0.10, 0.15, 0.20, 0.25, 0.20, 0.10]
    PRO3_WEIGHTS = [1/6] * 6

    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.drop_duplicates(subset="date").set_index("date").sort_index()

    all_dates = price_df.index
    start_idx = all_dates.get_indexer([pd.to_datetime(start_date)], method='bfill')[0]

    results = []
    for i in range(start_idx, len(all_dates)):
        result = run_one_rolling_test(
            price_df, all_dates, i, test_days,
            PRO1_WEIGHTS, PRO2_WEIGHTS, PRO3_WEIGHTS,
            run_simple_ttl_backtest
        )
        if result is not None:
            results.append(result)

    return pd.DataFrame(results)
