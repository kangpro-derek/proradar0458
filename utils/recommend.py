import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def recommend_best_strategy(recent_df: pd.DataFrame, history_df: pd.DataFrame, golden_cross=None):
    # ✅ 필요한 컬럼 체크
    required = ["기울기", "ma20", "ma60", "정배열", "이격도", "변동성", "ROC", "RSI"]
    for col in required:
        if col not in recent_df.columns:
            raise ValueError(f"최근 데이터에 '{col}' 컬럼이 없습니다.")

    # ✅ 정배열 자동 계산
    if golden_cross is None:
        golden_cross = int(recent_df["ma20"].iloc[-1] > recent_df["ma60"].iloc[-1])

    # ✅ 최근 특성 벡터
    recent_features = pd.DataFrame([{
        "정배열": golden_cross,
        "기울기": recent_df["기울기"].iloc[-1],
        "이격도": recent_df["이격도"].iloc[-1],
        "RSI": recent_df["RSI"].iloc[-1],
        "ROC": recent_df["ROC"].iloc[-1],
        "변동성": recent_df["변동성"].iloc[-1]
    }])

    # ✅ 과거 유사 구간 필터링
    history_df = history_df.copy()
    features = history_df[["정배열", "기울기", "이격도", "RSI", "ROC", "변동성"]].dropna()
    features = features[features["정배열"] == golden_cross]
    valid_idx = features.index

    if len(features) == 0:
        raise ValueError("⚠️ 유사 구간이 없습니다.")

    # ✅ 유사도 계산 (가중 코사인 거리 유사도)
    scaler = StandardScaler()
    scaled_history = scaler.fit_transform(features.drop("정배열", axis=1))
    scaled_recent = scaler.transform(recent_features.drop("정배열", axis=1))

    weights = np.array([2.0, 1.5, 1.2, 0.8, 0.6])  # 중요도 반영
    diff = scaled_history - scaled_recent
    distances = np.sqrt(np.sum((diff * weights) ** 2, axis=1))

    # ✅ 유사도 + 정렬
    result = history_df.loc[valid_idx].copy()
    result["distance"] = distances
    result["distance_norm"] = distances / distances.max()
    result["similarity"] = (1 - result["distance_norm"]) * 100
    # result["similarity"] = (1 - np.sqrt(result["distance_norm"])) * 100


    # ✅ 중복 제거 (시작일 기준 7일 이내 제거)
    result = result.sort_values("distance").reset_index(drop=True)
    top_matches = []
    for _, row in result.iterrows():
        if all(abs(pd.to_datetime(row["시작일"]) - pd.to_datetime(r["시작일"])).days > 7 for r in top_matches):
            top_matches.append(row)
        if len(top_matches) == 3:
            break

    return pd.DataFrame(top_matches)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_roc(series: pd.Series, period: int = 12) -> pd.Series:
    prev = series.shift(period)
    roc = (series - prev) / prev.replace(0, np.nan) * 100

    # # 디버깅 출력 - 처음 10개만 출력
    # debug_df = pd.DataFrame({
    #     "현재가": series,
    #     f"{period}일 전 가격": prev,
    #     "ROC": roc
    # })
    # print("🔍 ROC 계산 샘플 (상위 10개):")
    # print(debug_df.tail(10))

    return roc
