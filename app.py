# app.py - ProRadar0458 Flask 메인 서버

from flask import Flask, render_template, request
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    graph_html = None

    if request.method == "POST":
        # 사용자가 입력한 초기 투자금 가져오기
        initial_cash = float(request.form.get("initial_cash", 10000))

        # 간단한 테스트용 더미 데이터
        x = ["Day 1", "Day 2", "Day 3"]
        y = [initial_cash, initial_cash * 1.05, initial_cash * 1.10]

        # Plotly 그래프 생성
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='포트폴리오 가치'))
        fig.update_layout(title="테스트 백테스트 결과", xaxis_title="날짜", yaxis_title="가치")

        # HTML로 변환
        graph_html = pio.to_html(fig, full_html=False)

    # index.html 템플릿 렌더링 (그래프 포함)
    return render_template("index.html", graph_html=graph_html)

# 로컬 테스트용 실행 (Render에서는 gunicorn 사용)
if __name__ == "__main__":
    app.run(debug=True)
