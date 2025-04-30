# utils/backtest.py
# 📌 TTL 전략 백테스트용 클래스와 함수 정의

import pandas as pd
import numpy as np
from datetime import timedelta
import yfinance as yf
from datetime import datetime

def get_price_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    yfinance를 이용해 주가 데이터를 받아오는 함수
    :param symbol: 종목 코드 (예: 'SOXL')
    :param start: 시작일자 (예: '2021-01-01')
    :param end: 종료일자 (예: '2023-12-31')
    :return: 날짜별 종가가 포함된 DataFrame
    """
    df = yf.download(symbol, start=start, end=end, progress=False)[["Close"]].dropna()
    df = df.reset_index()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"])
    
    # ✅ 지표 계산
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma60'] = df['close'].rolling(window=60).mean()
    df["기울기"] = ((df["ma20"] - df["ma20"].shift(10)) / df["ma20"].shift(10)) * 100
    df["정배열"] = (df["ma20"] > df["ma60"]).astype(int)
    df["이격도"] = (df["close"] / df["ma20"] - 1) * 100
    df["수익률"] = df["close"].pct_change()
    df["변동성"] = df["수익률"].rolling(window=20).std() * (20 ** 0.5)
    df["상승비율"] = df["수익률"].rolling(window=20).apply(lambda x: (x > 0).mean(), raw=True)

    # ✅ RSI 계산 함수 필요 시:
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df["RSI"] = calculate_rsi(df["close"], period=14)

    return df


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
        shares = int(self.cash // actual_close)
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

    start_value = records[0]["portfolio_value"]
    end_value = records[-1]["portfolio_value"]
    max_drawdown = min([r["drawdown"] for r in records])

    return {
        "수익률": round((end_value - start_value) / start_value * 100, 2),
        "MDD": round(max_drawdown * 100, 2),
        "기록": records  # 나중에 시각화용
    }
