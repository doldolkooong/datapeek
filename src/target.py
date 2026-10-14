"""타깃 유형과 타깃-피처 관계를 분석합니다."""

import pandas as pd


def infer_task(series: pd.Series) -> str:
    """추정일 뿐이며, 앱 화면에서 사람이 직접 변경할 수 있습니다."""
    unique = series.nunique(dropna=True)
    if not pd.api.types.is_numeric_dtype(series):
        return "분류"
    if unique <= 12 and unique / max(series.notna().sum(), 1) <= 0.2:
        return "분류"
    return "회귀"


def target_distribution(df: pd.DataFrame, target: str, task: str) -> pd.DataFrame:
    values = df[target].dropna()
    if task == "분류":
        counts = values.astype(str).value_counts(dropna=False)
        return pd.DataFrame({"클래스": counts.index, "건수": counts.values,
                             "비율(%)": (counts.values / max(len(values), 1) * 100).round(2)})
    return values.describe().rename_axis("통계량").reset_index(name="값")


def feature_target_table(df: pd.DataFrame, target: str, feature: str, task: str) -> pd.DataFrame:
    frame = df[[feature, target]].dropna()
    if frame.empty or feature == target:
        return pd.DataFrame()
    numeric_feature = pd.api.types.is_numeric_dtype(frame[feature])
    if task == "분류":
        if numeric_feature:
            return frame.groupby(target, dropna=False)[feature].agg(["count", "mean", "median", "std"]).round(3).reset_index()
        return pd.crosstab(frame[feature].astype(str), frame[target].astype(str), normalize="index").head(20).round(3)
    if numeric_feature:
        spearman = frame[feature].corr(pd.to_numeric(frame[target], errors="coerce"), method="spearman")
        return pd.DataFrame({"피처": [feature], "Spearman 상관계수": [round(float(spearman), 3)]})
    return frame.groupby(feature, dropna=False)[target].agg(["count", "mean", "median"]).round(3).sort_values("count", ascending=False).head(20)
