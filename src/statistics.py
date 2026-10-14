"""DataPeek v3: 표로 확인하는 고급 통계 지표."""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name in df.select_dtypes(include="number"):
        s = pd.to_numeric(df[name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        records.append({"컬럼": name, "유효값": len(s), "평균": s.mean(), "중앙값": s.median(),
                        "표준편차": s.std(), "최솟값": s.min(), "Q1": s.quantile(.25),
                        "Q3": s.quantile(.75), "최댓값": s.max(), "왜도": s.skew(), "첨도": s.kurt()})
    return pd.DataFrame(records)


def categorical_profile(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name in df.select_dtypes(exclude="number"):
        s = df[name].dropna().astype(str)
        freq = s.value_counts()
        records.append({"컬럼": name, "유효값": len(s), "고유값": s.nunique(),
                        "최빈값": freq.index[0] if len(freq) else "", "최빈값 비율(%)":
                            round(float(freq.iloc[0] / len(s) * 100), 2) if len(freq) else np.nan,
                        "결측률(%)": round(float(df[name].isna().mean() * 100), 2)})
    return pd.DataFrame(records)


def top_correlations(df: pd.DataFrame, method: str = "pearson", limit: int = 30) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    numeric = numeric.loc[:, numeric.nunique() > 1].iloc[:, :40]
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["피처 A", "피처 B", "상관계수", "절댓값"])
    corr = numeric.corr(method=method)
    rows = []
    columns = corr.columns
    for i, a in enumerate(columns):
        for b in columns[i+1:]:
            score = corr.loc[a, b]
            if pd.notna(score):
                rows.append({"피처 A": a, "피처 B": b, "상관계수": float(score), "절댓값": abs(float(score))})
    return pd.DataFrame(rows).sort_values("절댓값", ascending=False).head(limit).reset_index(drop=True)


def cramers_v(df: pd.DataFrame, a: str, b: str) -> float:
    """범주형 두 변수 간 연관성: 편향 보정 Cramér's V (표본 부족 시 NaN)."""
    frame = df[[a, b]].dropna().astype(str)
    if len(frame) < 10:
        return float("nan")
    cross = pd.crosstab(frame[a], frame[b])
    if min(cross.shape) < 2 or max(cross.shape) > 100:
        return float("nan")
    chi2 = chi2_contingency(cross, correction=False)[0]
    n = cross.values.sum()
    phi2 = max(0.0, chi2 / n - ((cross.shape[0] - 1) * (cross.shape[1] - 1)) / (n - 1))
    r = cross.shape[0] - ((cross.shape[0] - 1)**2) / (n - 1)
    k = cross.shape[1] - ((cross.shape[1] - 1)**2) / (n - 1)
    if min(r - 1, k - 1) <= 0:
        return float("nan")
    return float(np.sqrt(phi2 / min(r - 1, k - 1)))


def target_associations(df: pd.DataFrame, target: str, task: str, limit: int = 30) -> pd.DataFrame:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import LabelEncoder
    work = df.dropna(subset=[target]).copy()
    if len(work) < 30 or work[target].nunique() < 2:
        return pd.DataFrame()
    selected = [c for c in work.select_dtypes(include="number").columns if c != target and work[c].nunique() > 1][:40]
    if not selected:
        return pd.DataFrame()
    x = work[selected].replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median()).fillna(0)
    if task == "분류":
        y = LabelEncoder().fit_transform(work[target].astype(str))
        method = mutual_info_classif
    else:
        y = pd.to_numeric(work[target], errors="coerce")
        valid = y.notna()
        x, y = x.loc[valid], y.loc[valid]
        method = mutual_info_regression
    if len(x) > 4000:
        x = x.sample(4000, random_state=42)
        y = pd.Series(y, index=work.index).loc[x.index]
    try:
        scores = method(x, y, random_state=42)
    except ValueError:
        return pd.DataFrame()
    return pd.DataFrame({"피처": selected, "Mutual Information": scores}).sort_values("Mutual Information", ascending=False).head(limit)
