"""실제 계산 수치에 근거한, 과장 없는 자동 EDA 요약 문장."""

import pandas as pd


def overview_insight(df):
    numeric = df.select_dtypes(include="number")
    return (f"전체 {len(df):,}행·{len(df.columns):,}열 중 수치형은 {len(numeric.columns)}개입니다. "
            "고유값 비율이 높은 컬럼은 식별자 성격인지 확인하세요.")


def quality_insight(df):
    missing = df.isna().mean().sort_values(ascending=False)
    duplicates = int(df.duplicated().sum())
    if missing.empty or missing.iloc[0] == 0:
        return f"결측치는 발견되지 않았습니다. 완전 중복 행은 {duplicates:,}건입니다."
    return (f"가장 결측률이 높은 컬럼은 '{missing.index[0]}' ({missing.iloc[0]*100:.2f}%)입니다. "
            f"완전 중복 행은 {duplicates:,}건입니다. 삭제 전 업무 의미를 확인하세요.")


def univariate_insight(df, numeric=None, categorical=None):
    messages = []
    if numeric:
        values = pd.to_numeric(df[numeric], errors="coerce").dropna()
        if len(values) > 2:
            skew = values.skew()
            messages.append(f"'{numeric}' 평균 {values.mean():,.2f}, 중앙값 {values.median():,.2f}, 왜도 {skew:.2f}입니다.")
    if categorical:
        counts = df[categorical].fillna("(결측)").astype(str).value_counts()
        if not counts.empty:
            messages.append(f"'{categorical}' 최빈 범주는 '{counts.index[0]}'로 {counts.iloc[0]/len(df)*100:.1f}%를 차지합니다.")
    return " ".join(messages) if messages else "선택한 컬럼의 자료형과 유효값을 확인하세요."


def target_insight(df, target, task):
    value = df[target].dropna()
    if value.empty:
        return "타깃이 모두 결측치입니다. 지도학습이 어렵습니다."
    if task == "분류":
        ratios = value.astype(str).value_counts(normalize=True)
        return (f"유효 타깃 {len(value):,}건, 클래스 {len(ratios)}개입니다. "
                f"가장 작은 클래스 '{ratios.index[-1]}'가 {ratios.iloc[-1]*100:.2f}%를 차지합니다.")
    v = pd.to_numeric(value, errors="coerce").dropna()
    if v.empty:
        return "선택한 회귀 타깃이 수치형으로 변환되지 않습니다."
    return f"회귀 타깃의 중앙값은 {v.median():,.2f}, 평균은 {v.mean():,.2f}입니다."


def compare_insight(table):
    numeric = table.dropna(subset=["PSI"]).sort_values("PSI", ascending=False)
    if not numeric.empty:
        return (f"수치형 변수 중 PSI가 가장 큰 컬럼은 '{numeric.iloc[0]['컬럼']}' "
                f"({numeric.iloc[0]['PSI']:.3f})입니다. 실제 분포 그래프도 함께 확인하세요.")
    return "유효한 PSI 결과가 없으므로 미지 범주·결측 비율을 중심으로 비교하세요."


def model_insight(result):
    metrics = ", ".join(f"{name}={value:.4f}" for name, value in result["metrics"].items())
    return f"홀드아웃 평가: {metrics}. 학습 내부 교차검증과 점수 차이를 확인하세요."
