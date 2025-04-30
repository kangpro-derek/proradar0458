# utils/backtest.py

def run_backtest_for_user(initial_cash: float = 10000):
    """
    사용자 입력을 받아 간단한 테스트 백테스트를 수행합니다.
    추후 yfinance 기반 데이터 백테스트로 교체 예정.
    """
    x = ["Day 1", "Day 2", "Day 3"]
    y = [initial_cash, initial_cash * 1.03, initial_cash * 1.07]
    return x, y
