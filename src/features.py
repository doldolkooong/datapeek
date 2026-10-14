"""학습 전에 수행해도 누수를 만들지 않는, 행별 결정적 파생변수 생성."""

import numpy as np
import pandas as pd


def make_features(df: pd.DataFrame, date_columns: list[str] | None = None,
                  text_columns: list[str] | None = None,
                  log_columns: list[str] | None = None,
                  ratio: tuple[str, str] | None = None) -> pd.DataFrame:
    result = df.copy()
    for column in date_columns or []:
        dt = pd.to_datetime(result[column], errors="coerce")
        result[f"{column}_year"] = dt.dt.year.astype(float)
        result[f"{column}_month"] = dt.dt.month.astype(float)
        result[f"{column}_dayofweek"] = dt.dt.dayofweek.astype(float)
        result = result.drop(columns=column)
    for column in text_columns or []:
        result[f"{column}_length"] = result[column].astype("string").str.len().astype(float)
    for column in log_columns or []:
        numeric = pd.to_numeric(result[column], errors="coerce")
        # log1p는 음수 원본을 보존하기 위해 음수 구간에서는 NaN 처리합니다.
        result[f"{column}_log1p"] = np.log1p(numeric.where(numeric >= 0))
    if ratio is not None and ratio[0] in result and ratio[1] in result:
        top = pd.to_numeric(result[ratio[0]], errors="coerce")
        bottom = pd.to_numeric(result[ratio[1]], errors="coerce").replace(0, np.nan)
        result[f"{ratio[0]}_over_{ratio[1]}"] = top / bottom
    result = result.replace([np.inf, -np.inf], np.nan)
    for column in result.select_dtypes(exclude="number").columns:
        result[column] = result[column].astype("string").fillna("__MISSING__").astype(str)
    return result
