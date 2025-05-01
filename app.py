from flask import Flask, render_template, request, redirect, url_for
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
from datetime import timedelta
import pandas as pd
from utils.backtest import run_simple_ttl_backtest, get_price_data
from utils.backtest import get_price_data, run_daily_rolling_backtest
from utils.recommend import recommend_best_strategy, calculate_rsi
from utils.backtest import run_simple_ttl_backtest

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
        yf_end = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        end = min(end, datetime.today().strftime("%Y-%m-%d"))
        selected_end = end  # ✅ 사용자가 입력한 값을 유지할 변수

        print("📅 selected_end:", selected_end)

        # ✅ 이평선 계산을 위한 여유 데이터 확보 (최대 90일 전)
        start_date_obj = datetime.strptime(start, "%Y-%m-%d")
        extended_start = (start_date_obj - timedelta(days=90)).strftime("%Y-%m-%d")

        # ✅ 확장 구간으로 데이터 요청
        df = get_price_data(symbol, extended_start, yf_end)
        print(f"📊 확장된 데이터 행 수: {len(df)}")

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

        chart_html = pio.to_html(chart_fig, full_html=False)


        # ✅ 최신 지표 추출
        latest = df.iloc[-1]
        feature_summary = {
            "정배열": "✅" if latest["정배열"] else "❌",
            "기울기": f"{latest['기울기']:.2f}%",
            "이격도": f"{latest['이격도']:.2f}%",
            "변동성": f"{latest['변동성']:.4f}",
            "상승비율": f"{latest['상승비율']:.2%}",
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

            graph_html = pio.to_html(fig, full_html=False)

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
    # ✅ 종료일 포함되도록 1일 더
    end_date = (datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    extended_start = (datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")

    df = get_price_data("SOXL", extended_start, end_date)

    # ✅ 이동평균선 계산
    df["ma20"] = df["close"].rolling(window=20).mean()
    df["ma60"] = df["close"].rolling(window=60).mean()
    df["기울기"] = ((df["ma20"] - df["ma20"].shift(10)) / df["ma20"].shift(10)) * 100
    df["정배열"] = (df["ma20"] > df["ma60"]).astype(int)
    df["이격도"] = (df["close"] / df["ma20"] - 1) * 100
    df["수익률"] = df["close"].pct_change()
    df["변동성"] = df["수익률"].rolling(window=20).std() * (20 ** 0.5)
    df["상승비율"] = df["수익률"].rolling(window=20).apply(lambda x: (x > 0).mean(), raw=True)
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
    test_df = test_df.drop(columns=["ma20", "ma60", "기울기", "정배열", "이격도", "RSI", "변동성", "상승비율"], errors="ignore")

    # ✅ 전략별 백테스트 실행
    result1 = run_simple_ttl_backtest(test_df, PRO1_WEIGHTS)
    result2 = run_simple_ttl_backtest(test_df, PRO2_WEIGHTS)
    result3 = run_simple_ttl_backtest(test_df, PRO3_WEIGHTS)

    return {
        "Pro1": {"수익률": result1["수익률"], "MDD": result1["MDD"]},
        "Pro2": {"수익률": result2["수익률"], "MDD": result2["MDD"]},
        "Pro3": {"수익률": result3["수익률"], "MDD": result3["MDD"]}
    }

# ✅ 전략 추천 페이지
@app.route("/recommend", methods=["GET", "POST"])
@app.route("/recommend", methods=["GET", "POST"])
def recommend():

    # ✅ 초기값
    selected_date = datetime.today().strftime("%Y-%m-%d")
    recommend_result = None

    if request.method == "POST":
        # ✅ 라디오 버튼 기준으로 날짜 결정
        date_mode = request.form.get("date_mode")
        if date_mode == "today":
            selected_date = datetime.today().strftime("%Y-%m-%d")
        else:
            selected_date = request.form.get("custom_date") or selected_date

        target_date = pd.to_datetime(selected_date)

        # 데이터 가져오기 (여유 기간 포함)
        df = get_price_data("SOXL", start="2011-10-01", end=(target_date + timedelta(days=1)).strftime("%Y-%m-%d"))
        df["date"] = pd.to_datetime(df["date"])

        # ✅ 지표 계산
        df["ma20"] = df["close"].rolling(window=20).mean()
        df["ma60"] = df["close"].rolling(window=60).mean()
        df["기울기"] = ((df["ma20"] - df["ma20"].shift(10)) / df["ma20"].shift(10)) * 100
        df["정배열"] = (df["ma20"] > df["ma60"]).astype(int)
        df["이격도"] = (df["close"] / df["ma20"] - 1) * 100
        df["수익률"] = df["close"].pct_change()
        df["변동성"] = df["수익률"].rolling(window=20).std() * (20 ** 0.5)
        df["상승비율"] = df["수익률"].rolling(window=20).apply(lambda x: (x > 0).mean(), raw=True)
        df["RSI"] = calculate_rsi(df["close"])

        # ✅ 최근 구간 추출 (종가 포함 마지막 30일)
        recent_window = df[df["date"] <= target_date].tail(30).reset_index(drop=True)
        if len(recent_window) < 30:
            return render_template("recommend.html",
                                   error="최근 30일 데이터가 부족합니다.",
                                   selected_date=selected_date)

        # ✅ rolling 백테스트 실행
        # 롤링 백테스트는 2012년부터 시작
        rolling_df = run_daily_rolling_backtest(df, start_date="2012-01-01", test_days=30)        
        cutoff_date = (target_date - timedelta(days=30)).date()
        past_df = rolling_df[rolling_df["종료일"] < cutoff_date].copy()

        # ✅ 지표 merge
        df["종료일"] = df["date"].dt.date
        merge_cols = ["종료일", "기울기", "정배열", "이격도", "상승비율", "변동성", "RSI"]
        past_df = pd.merge(past_df, df[merge_cols], on="종료일", how="left")

        # ✅ 유사 구간 top 3 추출
        top_matches_df = recommend_best_strategy(recent_window, past_df)
        
        # ✅ 유사 구간의 성과 확인 및 점수 계산
        score_rows = []
        for i, row in top_matches_df.iterrows():
            성과시작 = pd.to_datetime(row["종료일"]) + timedelta(days=1)
            성과종료 = 성과시작 + timedelta(days=30)

            # 성과 백테스트 실행
            performance = run_performance_backtests(df, 성과시작, 성과종료)

            score_rows.append({
                "시작일": row["시작일"],
                "종료일": row["종료일"],
                "Pro1_수익률": performance["Pro1"]["수익률"],
                "Pro1_mdd": performance["Pro1"]["MDD"],
                "Pro2_수익률": performance["Pro2"]["수익률"],
                "Pro2_mdd": performance["Pro2"]["MDD"],
                "Pro3_수익률": performance["Pro3"]["수익률"],
                "Pro3_mdd": performance["Pro3"]["MDD"]
            })

        score_df = pd.DataFrame(score_rows)

        # ✅ 점수 계산
        scores = {
            "Pro1": (score_df["Pro1_수익률"].mean() - 0.75 * abs(score_df["Pro1_mdd"].mean())) * 10,
            "Pro2": (score_df["Pro2_수익률"].mean() - 0.75 * abs(score_df["Pro2_mdd"].mean())) * 10,
            "Pro3": (score_df["Pro3_수익률"].mean() - 0.75 * abs(score_df["Pro3_mdd"].mean())) * 10,
        }
        best_strategy = max(scores, key=scores.get)
        
        # ✅ 상세 결과 리스트 구성
        top_details = []
        for display_index, (_, row) in enumerate(top_matches_df.iterrows(), start=1):
            similarity = round(row["similarity"], 2)

            # 성과 확인 구간
            성과시작 = pd.to_datetime(row["종료일"]) + timedelta(days=1)
            성과종료 = 성과시작 + timedelta(days=30)

            # matching row 찾기
            matched_row = score_df[
                (score_df["시작일"] == pd.to_datetime(row["시작일"]).date()) &
                (score_df["종료일"] == pd.to_datetime(row["종료일"]).date())
            ]

            if matched_row.empty:
                continue
            matched_row = matched_row.iloc[0]

            top_details.append({
                "순번": f"Top{display_index}",
                "시작일": row["시작일"],
                "종료일": row["종료일"],
                "정배열": "✅" if row["정배열"] else "❌",
                "기울기": f"{row['기울기']:.2f}%",
                "이격도": f"{row['이격도']:.2f}%",
                "변동성": f"{row['변동성']:.4f}",
                "상승비율": f"{row['상승비율']:.2%}",
                "RSI": f"{row['RSI']:.2f}",
                "유사도": f"{similarity}%",
                "성과시작": 성과시작.strftime("%Y-%m-%d"),
                "성과종료": 성과종료.strftime("%Y-%m-%d"),
                "Pro1": {
                    "수익률": f"{matched_row['Pro1_수익률']:.1f}%",
                    "MDD": f"{matched_row['Pro1_mdd']:.1f}%"
                },
                "Pro2": {
                    "수익률": f"{matched_row['Pro2_수익률']:.1f}%",
                    "MDD": f"{matched_row['Pro2_mdd']:.1f}%"
                },
                "Pro3": {
                    "수익률": f"{matched_row['Pro3_수익률']:.1f}%",
                    "MDD": f"{matched_row['Pro3_mdd']:.1f}%"
                }
            })



        # ✅ 점수 계산
        scores = {
            "Pro1": (score_df["Pro1_수익률"].mean() - 0.75 * abs(score_df["Pro1_mdd"].mean())) * 10,
            "Pro2": (score_df["Pro2_수익률"].mean() - 0.75 * abs(score_df["Pro2_mdd"].mean())) * 10,
            "Pro3": (score_df["Pro3_수익률"].mean() - 0.75 * abs(score_df["Pro3_mdd"].mean())) * 10,
        }
        best_strategy = max(scores, key=scores.get)
        
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
        chart_html = pio.to_html(chart_fig, full_html=False)


        # ✅ 최근 구간 지표 요약
        recent_summary = {
            "기준일": selected_date,
            "시작일": recent_window["date"].iloc[0].strftime("%Y-%m-%d"),
            "종료일": recent_window["date"].iloc[-1].strftime("%Y-%m-%d"),
            "정배열": "✅" if recent_window["정배열"].iloc[-1] else "❌",
            "기울기": f"{recent_window['기울기'].iloc[-1]:.2f}%",
            "이격도": f"{recent_window['이격도'].iloc[-1]:.2f}%",
            "변동성": f"{recent_window['변동성'].iloc[-1]:.4f}",
            "상승비율": f"{recent_window['상승비율'].iloc[-1]:.2%}",
            "RSI": f"{recent_window['RSI'].iloc[-1]:.2f}",
            "유사구간": top_matches_df.to_dict(orient="records"),
            "점수": scores,
            "추천": best_strategy,
            "유사구간상세": top_details,
        }

        return render_template("recommend.html",
                               selected_date=selected_date,
                               recommend_result=recent_summary,                               
                               chart_html=chart_html)

    # ✅ GET 요청
    return render_template("recommend.html",
                           selected_date=selected_date,
                           recommend_result=None,
                           chart_html=None)


if __name__ == "__main__":
    app.run(debug=True)
