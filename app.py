from flask import Flask, render_template, request, redirect, url_for
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
from datetime import timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os
import pyarrow.feather as feather
import time
from utils.backtest import run_simple_ttl_backtest, get_price_data
from utils.backtest import get_price_data, run_daily_rolling_backtest
from utils.recommend import recommend_best_strategy, calculate_rsi
from utils.backtest import run_simple_ttl_backtest
from utils.recommend import calculate_roc

TIERS = 6

PRO1_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25]
PRO2_WEIGHTS = [0.10, 0.15, 0.20, 0.25, 0.20, 0.10]
PRO3_WEIGHTS = [1 / TIERS] * TIERS

app = Flask(__name__)

# 홈 → 백테스트 페이지로 이동
@app.route("/")
def home():
    return redirect(url_for("backtest"))

# ✅ 백테스트 페이지
@app.route("/backtest", methods=["GET", "POST"])

def backtest():
    graph_html = None
    result_text = ""
    
    # ✅ 오늘 날짜 변수 추가
    today = datetime.today().strftime("%Y-%m-%d")
    # ✅ 모든 경우를 대비해 초기값 선언 (GET용 기본값)
    selected_symbol = "SOXL"
    selected_start = "2025-01-01"
    selected_end = today

    if request.method == "POST":
        print("✅ POST 요청 도착!")  # 터미널에서 확인

        symbol = request.form.get("symbol", "SOXL")
        selected_symbol = symbol  # ✅ 유지용 변수 저장
        
        start = request.form.get("start", "2025-01-01")
        selected_start = start  # ✅ 사용자가 입력한 시작일 유지

        end = request.form.get("end", datetime.today().strftime("%Y-%m-%d"))
        # ✅ 종료일을 포함하도록 하루 추가
        # yf_end = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")        
        end = min(end, datetime.today().strftime("%Y-%m-%d"))
        selected_end = end  # ✅ 사용자가 입력한 값을 유지할 변수

        print("📅 selected_end:", selected_end)

        # ✅ 이평선 계산을 위한 여유 데이터 확보 (최대 90일 전)
        start_date_obj = datetime.strptime(start, "%Y-%m-%d")
        extended_start = (start_date_obj - timedelta(days=90)).strftime("%Y-%m-%d")

        # ✅ 확장 구간으로 데이터 요청
        df = get_price_data(symbol, extended_start, end)
        # print(f"📊 확장된 데이터 행 수: {len(df)}")

        # ✅ 이동평균선을 위한 계산
        df["ma20"] = df["close"].rolling(window=20).mean()
        df["ma60"] = df["close"].rolling(window=60).mean()
        
        # ✅ 사용자가 요청한 실제 구간만 추출
        test_df = df[df["date"] >= start].reset_index(drop=True)

        initial_cash = 10000.0
        
        # ✅ 구간 차트용 데이터
        chart_df = df.copy()
        chart_df["ma20"] = chart_df["close"].rolling(window=20).mean()
        chart_df["ma60"] = chart_df["close"].rolling(window=60).mean()

        plot_df = test_df.copy()  # ← test_df를 그대로 사용
        chart_fig = go.Figure()
        chart_fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["close"], name="종가", line=dict(color="white")))
        chart_fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["ma20"], name="MA20", line=dict(color="orange")))
        chart_fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["ma60"], name="MA60", line=dict(color="green")))


        chart_fig.update_layout(
            title="📈 기간 차트 (log scale)",
            xaxis=dict(title=''),  # 하단 라벨 제거
            yaxis=dict(title="주가", type="log"),  # ✅ y축을 로그 스케일로 설정
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=20, t=40, b=0),  # ✅ 마진 조정: 상하좌우
            # ✅ 범례를 안쪽으로 넣기
            legend=dict(
                x=0.01,        # 왼쪽 여백 (0 ~ 1)
                y=0.99,        # 위쪽 정렬
                bgcolor="rgba(0,0,0,0)",  # 배경 투명
                borderwidth=0
            )
        )

        chart_html = pio.to_html(chart_fig, full_html=False, config={"staticPlot": True})


        # ✅ 최신 지표 추출
        latest = df.iloc[-1]
        feature_summary = {
            "정배열": "✅" if latest["정배열"] else "❌",
            "기울기": f"{latest['기울기']:.2f}%",
            "이격도": f"{latest['이격도']:.2f}%",
            "변동성": f"{latest['변동성']:.4f}",
            "ROC": f"{latest['ROC']/100:.2%}",
            "RSI": f"{latest['RSI']:.2f}"
        }

        # ✅ 전략별 가중치 정의
        strategies = {
            "Pro1": PRO1_WEIGHTS,
            "Pro2": PRO2_WEIGHTS,
            "Pro3": PRO3_WEIGHTS
        }

        results = {}

        # ✅ 전략별 결과 임시 저장용
        raw_results = {}
        all_y_values = []
        all_dd_values = []

        for name, weights in strategies.items():
            result = run_simple_ttl_backtest(test_df, weights, initial_capital=initial_cash)
            
            last_value = result["기록"][-1]["portfolio_value"]

            x_all = [r["date"] for r in result["기록"]]
            y_all = [r["portfolio_value"] for r in result["기록"]]

            # ✅ 누적 낙폭 (MDD 시계열)
            high_water = -float("inf")
            drawdowns = []
            for val in y_all:
                if val > high_water:
                    high_water = val
                dd = (val - high_water) / high_water
                drawdowns.append(dd)

            # ✅ 값 누적
            all_y_values.extend(y_all)
            all_dd_values.extend(drawdowns)

            # ✅ 임시 저장
            raw_results[name] = {
                "result": result,
                "last_value": last_value,
                "x_all": x_all,
                "y_all": y_all,
                "drawdowns": drawdowns,
                "weights": weights
            }

        # ✅ y축 범위 계산
        y2_max = max(all_y_values) * 1.1
        y1_min = min(all_dd_values)

        results = {}

        for name, info in raw_results.items():
            result = info["result"]
            x_all = info["x_all"]
            y_all = info["y_all"]
            drawdowns = info["drawdowns"]
            weights = info["weights"]
            last_value = info["last_value"]

            split_ratio_str = " / ".join(f"{round(w * 100, 1)}%" for w in weights)

            fig = go.Figure()

            # MDD 먼저
            fig.add_trace(go.Scatter(
                x=x_all,
                y=drawdowns,
                name="MDD",
                fill="tozeroy",
                mode="lines",
                line=dict(color="rgba(255,100,100,0.5)"),
                yaxis="y1"
            ))

            # 포트폴리오 선 나중
            fig.add_trace(go.Scatter(
                x=x_all,
                y=y_all,
                name="자산",
                line=dict(color="royalblue", width=3),
                mode="lines",
                yaxis="y2"
            ))

            adjusted_y1_min = y1_min * 1.33

            fig.update_layout(
                title=f"{name} 자산 및 MDD 차트",
                template="plotly_dark",
                xaxis=dict(title=None),
                yaxis=dict(title=None, range=[adjusted_y1_min, 0], side="left"),
                yaxis2=dict(title=None, range=[0, y2_max], overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=80, b=30)  # ✅ 마진 조정: 상하좌우
            )

            graph_html = pio.to_html(fig, full_html=False, config={"staticPlot": True})

            results[name] = {
                "final_value": last_value,
                "수익률": result["수익률"],
                "MDD": result["MDD"],
                "분할비율": split_ratio_str,
                "graph": graph_html
            }


        # (원한다면 그래프 하나 만들기 가능)
        graph_html = None

        # ✅ 템플릿에 today 포함해서 전달
        return render_template(
            "backtest.html",
            graph_html=graph_html,
            today=today,
            request=request,
            results=results,
            feature_summary=feature_summary,
            selected_start=selected_start,
            selected_end=selected_end,
            selected_symbol=selected_symbol,
            chart_html=chart_html,
        )

    # ✅ GET 요청일 때도 기본적으로 렌더링 해줘야 함!
    return render_template(
        "backtest.html",
        graph_html=None,
        result_text="",
        today=today,
        request=request,
        results=None,
        feature_summary=None,
        selected_start=selected_start,
        selected_end=selected_end,
        selected_symbol=selected_symbol,
        chart_html=None
    )

def find_mdd_period(records):
    peak_idx = 0
    trough_idx = 0
    max_drawdown = 0

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            peak = records[i]["portfolio_value"]
            trough = records[j]["portfolio_value"]
            drawdown = (trough - peak) / peak
            if drawdown < max_drawdown:
                max_drawdown = drawdown
                peak_idx = i
                trough_idx = j

    return peak_idx, trough_idx

def run_recommendation_logic(target_date):
    
    end_date = (datetime.strptime(target_date, "%Y-%m-%d")).strftime("%Y-%m-%d")
    extended_start = (datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")

    df = get_price_data("SOXL", extended_start, end_date)
    # ✅ 빈 데이터프레임 예외 처리
    if df.empty:
        return render_template(
            "backtest.html",
            graph_html=None,
            result_text="📭 데이터를 불러오지 못했습니다. 날짜 범위나 종목명을 확인해주세요.",
            today=today,
            request=request,
            results=None,
            feature_summary=None,
            selected_start=selected_start,
            selected_end=selected_end,
            selected_symbol=selected_symbol,
            chart_html=None
        )
            
    # ✅ 이동평균선 계산
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["ma60"] = df["close"].rolling(window=60).mean()
    df["기울기"] = ((df["ma20"] - df["ma20"].shift(10)) / df["ma20"].shift(10)) * 100
    df["정배열"] = (df["ma20"] > df["ma60"]).astype(int)
    df["이격도"] = (df["close"] / df["ma20"] - 1) * 100
    df["수익률"] = df["close"].pct_change()
    df["변동성"] = df["수익률"].rolling(window=20).std() * (20 ** 0.5)
    df["ROC"] = calculate_roc(df["close"], period=12)
    df["RSI"] = calculate_rsi(df["close"], period=14)

    # ✅ 해당일 기준 최근 30일
    recent_df = df[df["date"] <= target_date].tail(30).reset_index(drop=True)
    if len(recent_df) < 30:
        return {"error": "해당 날짜 기준으로 최근 30일 데이터가 부족합니다."}

    # ✅ 전략 백테스트
    strategies = {
        "Pro1": PRO1_WEIGHTS,
        "Pro2": PRO2_WEIGHTS,
        "Pro3": PRO3_WEIGHTS
    }

    results = {}
    for name, weights in strategies.items():
        result = run_simple_ttl_backtest(recent_df, weights, initial_capital=10000)
        score = result["수익률"] - 0.75 * abs(result["MDD"])
        results[name] = {
            "score": round(score, 2),
            "수익률": result["수익률"],
            "MDD": result["MDD"]
        }

    best = max(results.items(), key=lambda x: x[1]["score"])

    return {
        "기준일": target_date,
        "추천전략": best[0],
        "전략들": results
    }
    
def run_performance_backtests(df, start_date, end_date):
    """
    성과 확인 구간에 대해 3가지 전략을 실행하여 수익률과 MDD 반환
    - df: 전체 가격 데이터
    - start_date: 성과 확인 시작일 (datetime 형식)
    - end_date: 성과 확인 종료일 (datetime 형식)
    """

    # ✅ 전략별 가중치
    PRO1_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.25]
    PRO2_WEIGHTS = [0.10, 0.15, 0.20, 0.25, 0.20, 0.10]
    PRO3_WEIGHTS = [1 / 6] * 6

    # ✅ 이평선 계산을 위해 60일 전부터 데이터 확보
    extended_start = start_date - timedelta(days=60)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= extended_start) & (df["date"] <= end_date)].reset_index(drop=True)

    # ✅ 성과 확인 기간만 추출
    test_df = df[df["date"] >= start_date].reset_index(drop=True)
    
    # ✅ 지표 제거: 수익률에 영향 없도록
    test_df = test_df.drop(columns=["ma20", "ma60", "기울기", "정배열", "이격도", "RSI", "변동성", "ROC"], errors="ignore")

    # ✅ 전략별 백테스트 실행
    result1 = run_simple_ttl_backtest(test_df, PRO1_WEIGHTS)
    result2 = run_simple_ttl_backtest(test_df, PRO2_WEIGHTS)
    result3 = run_simple_ttl_backtest(test_df, PRO3_WEIGHTS)

    return {
        "Pro1": {"수익률": result1["수익률"], "MDD": result1["MDD"]},
        "Pro2": {"수익률": result2["수익률"], "MDD": result2["MDD"]},
        "Pro3": {"수익률": result3["수익률"], "MDD": result3["MDD"]}
    }


def evaluate_strategy(row, df):
    row_dict = row._asdict()
    성과시작 = pd.to_datetime(row_dict["종료일"]) + timedelta(days=1)
    성과종료 = 성과시작 + timedelta(days=30)
    performance = run_performance_backtests(df, 성과시작, 성과종료)

    return {
        "시작일": row_dict["시작일"],
        "종료일": row_dict["종료일"],
        "Pro1_수익률": performance["Pro1"]["수익률"],
        "Pro1_mdd": performance["Pro1"]["MDD"],
        "Pro2_수익률": performance["Pro2"]["수익률"],
        "Pro2_mdd": performance["Pro2"]["MDD"],
        "Pro3_수익률": performance["Pro3"]["수익률"],
        "Pro3_mdd": performance["Pro3"]["MDD"]
    }
    
def load_or_run_rolling_cache(symbol, df, start_date, test_days):
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{symbol}_rolling.csv")
    last_bt_path = os.path.join(cache_dir, f"{symbol}_rolling_last.txt")  # ← 마지막 백테스트 기록

    # ✅ 기존 캐시 로딩
    if os.path.exists(cache_path):
        existing_df = pd.read_csv(cache_path, parse_dates=["종료일"])
        print(f"📁 [롤링 캐시] 기존 캐시 로딩: {cache_path}")
    else:
        existing_df = pd.DataFrame()
        print(f"📁 [롤링 캐시] 새 캐시 파일 생성 예정: {cache_path}")

    # ✅ 가격 데이터의 마지막 날짜 확인
    df["date"] = pd.to_datetime(df["date"])
    latest_price_date = df["date"].max()

    # ✅ 마지막 백테스트 종료일 불러오기
    if os.path.exists(last_bt_path):
        with open(last_bt_path, "r") as f:
            last_bt_str = f.read().strip()
            last_backtested_date = pd.to_datetime(last_bt_str)
            print(f"📜 마지막 백테스트 종료일: {last_backtested_date.date()}")

        # ✅ 백테스트가 최신까지 되어 있지 않다면 → 다시 실행
        if last_backtested_date < latest_price_date:
            start_bt_date = (latest_price_date - timedelta(days=30)).strftime("%Y-%m-%d")
            print(f"🆕 새로운 데이터 있음 → {start_bt_date}부터 백테스트 실행")
        else:
            print("📭 최신까지 백테스트 완료 → 생략")
            return existing_df
    else:
        last_backtested_date = pd.to_datetime(start_date)
        start_bt_date = start_date
        print(f"🆕 기록 없음 → 시작일 사용: {start_bt_date}")

    # ✅ 백테스트 수행
    t0 = time.perf_counter()
    full_new_df = run_daily_rolling_backtest(df, start_date=start_bt_date, test_days=test_days)
    t1 = time.perf_counter()
    print(f"⏱️ run_daily_rolling_backtest 소요 시간: {t1 - t0:.2f}초")

    # ✅ 새로운 데이터가 없다면 종료
    if full_new_df.empty:
        print("📄 [롤링 캐시] 새로운 백테스트 결과 없음")
        return existing_df

    # ✅ 캐시 병합 및 저장
    updated_df = pd.concat([existing_df, full_new_df], ignore_index=True)
    updated_df["종료일"] = pd.to_datetime(updated_df["종료일"], errors="coerce")
    updated_df = updated_df.drop_duplicates(subset=["종료일"]).sort_values("종료일")
    updated_df.to_csv(cache_path, index=False)
    print(f"✅ [캐시 저장] 병합된 캐시 저장 완료: {cache_path}")

    # ✅ 마지막 백테스트 종료일 갱신
    latest_done = updated_df["종료일"].max()
    with open(last_bt_path, "w") as f:
        f.write(latest_done.strftime("%Y-%m-%d"))
    print(f"📌 [기록 갱신] 마지막 백테스트 종료일 → {latest_done.date()}")

    return updated_df


    
# ✅ 전략 추천 페이지
@app.route("/recommend", methods=["GET", "POST"])

def recommend():
    selected_date = datetime.today().strftime("%Y-%m-%d")
    recommend_result = None

    if request.method == "POST":
        date_mode = request.form.get("date_mode")
        if date_mode == "today":
            selected_date = datetime.today().strftime("%Y-%m-%d")
        else:
            selected_date = request.form.get("custom_date") or selected_date

        target_date = pd.to_datetime(selected_date)
        df = get_price_data("SOXL", start="2012-01-01", end=(target_date).strftime("%Y-%m-%d"))
        if df.empty:
            return render_template("backtest.html", graph_html=None, result_text="📭 데이터를 불러오지 못했습니다. 날짜 범위나 종목명을 확인해주세요.", today=datetime.today().strftime("%Y-%m-%d"), request=request, results=None, feature_summary=None, selected_start=None, selected_end=None, selected_symbol="SOXL", chart_html=None)

        df["date"] = pd.to_datetime(df["date"])
        df["종료일"] = df["date"].dt.date

        recent_window = df[df["date"] <= target_date].tail(30).reset_index(drop=True)
        if len(recent_window) < 30:
            return render_template("recommend.html", error="최근 30일 데이터가 부족합니다.", selected_date=selected_date)

        rolling_df = load_or_run_rolling_cache("SOXL", df, start_date="2012-01-01", test_days=30)
        # cutoff_date = (target_date - timedelta(days=30)).date()
        # past_df = rolling_df[rolling_df["종료일"] < cutoff_date].copy()
        cutoff_date = pd.to_datetime(target_date - timedelta(days=30))
        past_df = rolling_df[rolling_df["종료일"] < cutoff_date].copy()

        merge_cols = ["종료일", "기울기", "정배열", "이격도", "ROC", "변동성", "RSI"]
        
        # 🔑 '종료일' 컬럼 타입 일치시키기
        df["종료일"] = pd.to_datetime(df["종료일"], errors="coerce")
        past_df["종료일"] = pd.to_datetime(past_df["종료일"], errors="coerce")

        past_df = pd.merge(past_df, df[merge_cols], on="종료일", how="left")

        top_matches_df = recommend_best_strategy(recent_window, past_df)
        # print("🔍 유사 구간 개수:", len(top_matches_df))
        # print(top_matches_df.head())  # 첫 몇 개 출력
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            score_rows = list(executor.map(lambda row: evaluate_strategy(row, df), top_matches_df.itertuples(index=False, name="Row")))

        # print("📊 추천 유사 구간 평가 결과:")
        # for row in score_rows:
        #     print(row)
    
        score_df = pd.DataFrame(score_rows)
        # ✅ 방식 4번: 점수 = 수익률 × exp(MDD)
        def calc_exp_score(row, prefix):
            r = row[f"{prefix}_수익률"] / 100
            m = row[f"{prefix}_mdd"] / 100
            return r * (2.718 ** m) * 100  # 수익률은 % 환산, 점수는 다시 %로

        scores = {
            "Pro1": score_df.apply(lambda row: calc_exp_score(row, "Pro1"), axis=1).mean(),
            "Pro2": score_df.apply(lambda row: calc_exp_score(row, "Pro2"), axis=1).mean(),
            "Pro3": score_df.apply(lambda row: calc_exp_score(row, "Pro3"), axis=1).mean()
        }
        best_strategy = max(scores, key=scores.get)

        # ✅ 날짜 형식 통일
        score_df["시작일"] = pd.to_datetime(score_df["시작일"]).dt.date
        score_df["종료일"] = pd.to_datetime(score_df["종료일"]).dt.date

        top_details = []
        for display_index, (_, row) in enumerate(top_matches_df.iterrows(), start=1):
            similarity = round(row["similarity"], 2)
            성과시작 = pd.to_datetime(row["종료일"]) + timedelta(days=1)
            성과종료 = 성과시작 + timedelta(days=30)

            matched_row = score_df[
                (score_df["시작일"] == pd.to_datetime(row["시작일"]).date()) &
                (score_df["종료일"] == pd.to_datetime(row["종료일"]).date())
            ]
            
            if matched_row.empty:
                continue
            matched_row = matched_row.iloc[0]

            # ✅ 유사 구간 차트 생성
           # 차트 범위: 유사구간 시작 ~ 성과구간 종료
            plot_start = pd.to_datetime(row["시작일"])
            plot_end = 성과종료  # 이미 datetime 형식일 것

            match_plot_df = df[(df["date"] >= plot_start) & (df["date"] <= plot_end)].copy()

            # 차트 생성
            match_chart = go.Figure()
            match_chart.add_trace(go.Scatter(x=match_plot_df["date"], y=match_plot_df["close"], name="종가", line=dict(color="white")))
            match_chart.add_trace(go.Scatter(x=match_plot_df["date"], y=match_plot_df["ma20"], name="MA20", line=dict(color="orange")))
            match_chart.add_trace(go.Scatter(x=match_plot_df["date"], y=match_plot_df["ma60"], name="MA60", line=dict(color="green")))

            # ✅ 성과 시작일에 세로선 추가
            match_chart.add_shape(
                type="line",
                x0=성과시작,
                x1=성과시작,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="red", width=2, dash="dot")
            )

            match_chart.add_vrect(
                x0=성과시작, x1=성과종료,
                fillcolor="rgba(255, 100, 100, 0.2)",  # 붉은색 반투명
                layer="below",
                line_width=0,
            )

            match_chart.update_layout(
                xaxis=dict(title='', range=[plot_start, plot_end]),
                yaxis_title="주가 (로그)",
                yaxis_type="log",
                template="plotly_dark",
                height=250,
                margin=dict(l=30, r=20, t=30, b=0),
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", borderwidth=0)
            )
            
            match_chart_html = pio.to_html(match_chart, full_html=False, config={"staticPlot": True})
            
            top_details.append({
                "순번": f"Top{display_index}",
                "시작일": row["시작일"],
                "종료일": row["종료일"],
                "정배열": "✅" if row["정배열"] else "❌",
                "기울기": f"{row['기울기']:.2f}%",
                "이격도": f"{row['이격도']:.2f}%",
                "변동성": f"{row['변동성']:.4f}",
                "ROC": f"{row['ROC']/100:.2%}",
                "RSI": f"{row['RSI']:.2f}",
                "유사도": f"{similarity}%",
                "성과시작": 성과시작.strftime("%Y-%m-%d"),
                "성과종료": 성과종료.strftime("%Y-%m-%d"),
                "Pro1": {"수익률": f"{matched_row['Pro1_수익률']:.1f}%", "MDD": f"{matched_row['Pro1_mdd']:.1f}%"},
                "Pro2": {"수익률": f"{matched_row['Pro2_수익률']:.1f}%", "MDD": f"{matched_row['Pro2_mdd']:.1f}%"},
                "Pro3": {"수익률": f"{matched_row['Pro3_수익률']:.1f}%", "MDD": f"{matched_row['Pro3_mdd']:.1f}%"},
                "차트": match_chart_html
            })
                        
        # 📌 날짜를 datetime 객체로 변환하여 템플릿에서 strftime 안전하게 사용
        for item in top_details:
            item["시작일"] = pd.to_datetime(item["시작일"])
            item["종료일"] = pd.to_datetime(item["종료일"])
            
        # ✅ 분석 구간 차트 생성
        plot_df = recent_window.copy()
        chart_fig = go.Figure()
        chart_fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["close"], name="종가", line=dict(color="white")))
        chart_fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["ma20"], name="MA20", line=dict(color="orange")))
        chart_fig.add_trace(go.Scatter(x=plot_df["date"], y=plot_df["ma60"], name="MA60", line=dict(color="green")))

        chart_fig.update_layout(
            xaxis=dict(title=''),
            yaxis_title="주가 (로그)",
            yaxis_type="log",
            template="plotly_dark",
            height=300,
            margin=dict(l=40, r=20, t=40, b=0),
            legend=dict(
                x=0.01, y=0.99,
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0
            )
        )
        chart_html = pio.to_html(chart_fig, full_html=False, config={"staticPlot": True})

        # ✅ 전략 파라미터 설명 텍스트 추가 (실제 상수값 기반으로 계산)
        def weights_to_str(weights):
            return " / ".join(f"{w * 100:.1f}%" for w in weights)

        strategy_descriptions = {
            "Pro1": {
                "weights": weights_to_str(PRO1_WEIGHTS),
                "details": ["6분할 10일 손절", "손절 시 매수 O", "수익금 재투자 X", "정액매수 X", "매수 기준 -0.01%", "매도 기준 +0.01%"]
            },
            "Pro2": {
                "weights": weights_to_str(PRO2_WEIGHTS),
                "details": ["6분할 10일 손절", "손절 시 매수 O", "수익금 재투자 X", "정액매수 X", "매수 기준 -0.01%", "매도 기준 +0.01%"]
            },
            "Pro3": {
                "weights": weights_to_str(PRO3_WEIGHTS),
                "details": ["6분할 10일 손절", "손절 시 매수 O", "수익금 재투자 X", "정액매수 X", "매수 기준 -0.01%", "매도 기준 +0.01%"]
            }
        }

        strategy_params = strategy_descriptions.get(best_strategy, {})
        
        recent_summary = {
            "기준일": selected_date,
            "시작일": recent_window["date"].iloc[0].strftime("%Y-%m-%d"),
            "종료일": recent_window["date"].iloc[-1].strftime("%Y-%m-%d"),
            "정배열": "✅" if recent_window["정배열"].iloc[-1] else "❌",
            "기울기": f"{recent_window['기울기'].iloc[-1]:.2f}%",
            "이격도": f"{recent_window['이격도'].iloc[-1]:.2f}%",
            "변동성": f"{recent_window['변동성'].iloc[-1]:.4f}",
            "ROC": f"{recent_window['ROC'].iloc[-1]/100:.2%}",
            "RSI": f"{recent_window['RSI'].iloc[-1]:.2f}",
            "유사구간": top_matches_df.to_dict(orient="records"),
            "점수": scores,
            "추천": best_strategy,
            "유사구간상세": top_details,
            "추천전략파라미터": strategy_params
        }

        return render_template(
            "recommend.html", 
            selected_date=selected_date, 
            today=datetime.today().strftime("%Y-%m-%d"),  
            recommend_result=recent_summary, 
            chart_html=chart_html,
            date_mode=date_mode)

    return render_template(
        "recommend.html", 
        selected_date=selected_date, 
        recommend_result=None, 
        chart_html=None,        
        date_mode="today"  # ✅ 기본값 설정
        )


from utils.stat import run_or_load_yearly_statistics  # 👈 아까 만든 함수 위치

@app.route("/stats")
def stats():    
         
    # ✅ 연도별 백테스트 실행 or 캐시 로드
    yearly_df = run_or_load_yearly_statistics()

    pro1_stats = yearly_df[["연도", "Pro1_수익률", "Pro1_MDD"]].rename(
        columns={"Pro1_수익률": "수익률", "Pro1_MDD": "MDD"}
    ).to_dict(orient="records")

    pro2_stats = yearly_df[["연도", "Pro2_수익률", "Pro2_MDD"]].rename(
        columns={"Pro2_수익률": "수익률", "Pro2_MDD": "MDD"}
    ).to_dict(orient="records")

    pro3_stats = yearly_df[["연도", "Pro3_수익률", "Pro3_MDD"]].rename(
        columns={"Pro3_수익률": "수익률", "Pro3_MDD": "MDD"}
    ).to_dict(orient="records")

    return render_template("stats.html",
                           pro1_stats=pro1_stats,
                           pro2_stats=pro2_stats,
                           pro3_stats=pro3_stats)


@app.route("/api/stats/yearly/<int:year>")
def api_yearly_chart(year):
    cache_path = f"cache/stats_{year}.csv"
    if not os.path.exists(cache_path):
        return {"error": "해당 연도의 캐시 파일이 없습니다."}, 404

    df = pd.read_csv(cache_path, parse_dates=["date"])
    df["portfolio_value"] = df["portfolio_value"].round(2)
    df["drawdown"] = (df["drawdown"] * 100).round(2)

    return df.to_dict(orient="records")


@app.route("/info")
def info():
    return render_template("info.html")




if __name__ == "__main__":
    app.run(debug=True)
    
    
