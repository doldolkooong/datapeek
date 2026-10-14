"""Plotly 기반 시각화 기능."""

import pandas as pd
import plotly.express as px


def make_histogram(df: pd.DataFrame, column: str):
    """선택한 수치형 컬럼의 분포를 보여줍니다."""
    fig = px.histogram(df, x=column, nbins=25, title=f"{column} 분포")
    fig.update_layout(bargap=0.05)
    return fig


def make_category_chart(df: pd.DataFrame, column: str):
    """범주형 컬럼에서 많이 등장한 상위 10개 값을 보여줍니다."""
    values = df[column].fillna("(결측)").astype(str)
    counts = values.value_counts().head(10)
    chart_data = pd.DataFrame({"값": counts.index, "건수": counts.values})
    return px.bar(chart_data, x="값", y="건수", title=f"{column} 상위 10개")


def make_correlation_chart(df: pd.DataFrame):
    """상관관계 분석이 가능한 수치형 컬럼이 2개 이상일 때만 차트를 반환합니다."""
    numeric_df = df.select_dtypes(include="number")
    numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) > 1]

    if len(numeric_df.columns) < 2:
        return None

    correlation = numeric_df.corr()
    fig = px.imshow(
        correlation,
        title="수치형 변수 상관관계",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        text_auto=".2f",
    )
    return fig


def make_boxplot(df: pd.DataFrame, column: str):
    """수치형 변수의 사분위 범위와 특이값을 보여줍니다."""
    return px.box(df, y=column, title=f"{column} 박스플롯", points="outliers")


def make_target_chart(df: pd.DataFrame, target: str, task: str):
    """분류는 클래스 빈도, 회귀는 수치 분포로 보여줍니다."""
    if task == "분류":
        counts = df[target].dropna().astype(str).value_counts().reset_index()
        counts.columns = ["클래스", "건수"]
        return px.bar(counts, x="클래스", y="건수", title=f"{target} 클래스 분포")
    return px.histogram(df, x=target, nbins=30, title=f"{target} 타깃 분포")


def make_relationship_chart(df: pd.DataFrame, target: str, feature: str, task: str):
    """타깃과 피처 타입에 맞춰 산점도·상자그림·막대그래프를 생성합니다."""
    frame = df[[feature, target]].dropna().head(10000)
    numeric = pd.api.types.is_numeric_dtype(frame[feature])
    if task == "분류":
        if numeric:
            return px.box(frame, x=target, y=feature, title=f"{feature} vs {target}")
        subset = frame.copy()
        subset[feature] = subset[feature].astype(str)
        top = subset[feature].value_counts().head(12).index
        subset = subset[subset[feature].isin(top)]
        grouped = subset.groupby([feature, target]).size().reset_index(name="건수")
        return px.bar(grouped, x=feature, y="건수", color=target, barmode="group", title=f"{feature}별 타깃 구성")
    if numeric:
        return px.scatter(frame, x=feature, y=target, opacity=0.5, title=f"{feature} vs {target}")
    top = frame[feature].astype(str).value_counts().head(12).index
    subset = frame[frame[feature].astype(str).isin(top)]
    return px.box(subset, x=feature, y=target, title=f"{feature}별 {target} 분포")
