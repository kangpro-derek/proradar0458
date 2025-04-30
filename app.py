from flask import Flask, render_template, request, redirect, url_for
import plotly.graph_objects as go
import plotly.io as pio
from utils.backtest import run_simple_ttl_backtest, get_price_data
from datetime import datetime
from datetime import timedelta

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
            title="📈 기간 차트",
            xaxis=dict(title=''),  # 하단 라벨 제거
            yaxis_title="주가",
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
                name=f"{name} 가치",
                line=dict(color="royalblue", width=3),
                mode="lines",
                yaxis="y2"
            ))

            fig.update_layout(
                title=f"{name} 포트폴리오 가치 & MDD",
                template="plotly_dark",
                xaxis=dict(title="날짜"),
                yaxis=dict(title="MDD (%)", range=[y1_min, 0], side="left"),
                yaxis2=dict(title="자산 가치", range=[0, y2_max], overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
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



# ✅ 전략 추천 페이지 (비어 있음)
@app.route("/recommend", methods=["GET"])
def recommend():
    return render_template("recommend.html", request=request)



if __name__ == "__main__":
    app.run(debug=True)
