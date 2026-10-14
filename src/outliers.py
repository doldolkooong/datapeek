"""IQR과 Isolation Forest를 이용해 이상치 후보를 표시합니다."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def detect_outliers(df: pd.DataFrame, columns: list[str], method: str = "Isolation Forest",
                    contamination: float = 0.05) -> pd.DataFrame:
    if not columns:
        raise ValueError("수치형 컬럼을 하나 이상 선택해 주세요.")
    values = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    usable = [name for name in columns if values[name].notna().any()]
    if not usable:
        raise ValueError("모든 값이 결측치인 컬럼은 분석할 수 없습니다.")
    if len(df) < 5:
        raise ValueError("이상치 탐지에는 행이 5개 이상 필요합니다.")
    values = values[usable]
    if method == "IQR":
        flags = pd.Series(False, index=df.index)
        for column in usable:
            series = values[column]
            q1, q3 = series.quantile([0.25, 0.75])
            spread = q3 - q1
            if spread > 0:
                flags = flags | ((series < q1 - 1.5 * spread) | (series > q3 + 1.5 * spread)).fillna(False)
        return pd.DataFrame({"이상치 여부": flags.astype(bool), "이상 점수": np.nan}, index=df.index)
    if method != "Isolation Forest":
        raise ValueError("지원하지 않는 탐지 방법입니다.")
    if not 0 < contamination <= 0.5:
        raise ValueError("이상치 예상 비율은 0보다 크고 0.5 이하여야 합니다.")
    processor = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    x = processor.fit_transform(values)
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
    flags = model.fit_predict(x) == -1
    scores = -model.decision_function(x)  # 클수록 이상한 정도가 높음
    return pd.DataFrame({"이상치 여부": flags, "이상 점수": scores}, index=df.index)
