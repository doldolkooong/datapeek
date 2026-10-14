"""Train / Test의 분포 차이와 미지 범주를 점검합니다."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def population_stability_index(train: pd.Series, test: pd.Series, bins: int = 10) -> float:
    """학습 데이터 분위수 기준 PSI. 작은 표본/상수 컬럼은 NaN을 반환합니다."""
    a = pd.to_numeric(train, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    b = pd.to_numeric(test, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(a) < 10 or len(b) < 10 or np.unique(a).size < 2:
        return float("nan")
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    a_hist = np.histogram(a, bins=edges)[0].astype(float)
    b_hist = np.histogram(b, bins=edges)[0].astype(float)
    p = np.clip(a_hist / len(a), 1e-5, 1)
    q = np.clip(b_hist / len(b), 1e-5, 1)
    return float(np.sum((p - q) * np.log(p / q)))


def compare_train_test(train: pd.DataFrame, test: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    rows = []
    common = [col for col in train.columns if col in test.columns and col != target]
    for column in common:
        a, b = train[column], test[column]
        a_missing = round(float(a.isna().mean() * 100), 2)
        b_missing = round(float(b.isna().mean() * 100), 2)
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            a_valid = pd.to_numeric(a, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            b_valid = pd.to_numeric(b, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(a_valid) > 1 and len(b_valid) > 1:
                stat, pvalue = ks_2samp(a_valid.sample(min(len(a_valid), 10000), random_state=42),
                                        b_valid.sample(min(len(b_valid), 10000), random_state=42))
                psi = population_stability_index(a_valid, b_valid)
                rows.append({"컬럼": column, "유형": "수치형", "Train 결측(%)": a_missing,
                             "Test 결측(%)": b_missing, "KS 통계량": round(float(stat), 4),
                             "KS p값": round(float(pvalue), 5),
                             "PSI": round(psi, 4) if np.isfinite(psi) else np.nan, "새 범주 수": None})
            else:
                rows.append({"컬럼": column, "유형": "수치형", "Train 결측(%)": a_missing,
                             "Test 결측(%)": b_missing, "KS 통계량": None, "KS p값": None,
                             "PSI": None, "새 범주 수": None})
        else:
            a_categories = set(a.dropna().astype(str).unique())
            b_categories = set(b.dropna().astype(str).unique())
            rows.append({"컬럼": column, "유형": "범주형", "Train 결측(%)": a_missing,
                         "Test 결측(%)": b_missing, "KS 통계량": None, "KS p값": None,
                         "PSI": None, "새 범주 수": len(b_categories - a_categories)})
    return pd.DataFrame(rows, columns=["컬럼", "유형", "Train 결측(%)", "Test 결측(%)", "KS 통계량", "KS p값", "PSI", "새 범주 수"])
