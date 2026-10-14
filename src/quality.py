"""결측치, 특이값과 누수 의심 컬럼을 탐색하는 함수들."""

import numpy as np
import pandas as pd


def iqr_summary(df: pd.DataFrame) -> pd.DataFrame:
    """각 수치형 컬럼의 IQR 기준 이상치 후보 수를 계산합니다."""
    records = []
    for column in df.select_dtypes(include="number").columns:
        series = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            count = 0
        else:
            count = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        records.append({"컬럼": column, "IQR 이상치 후보": count, "비율(%)": round(count / len(series) * 100, 2),
                        "왜도": round(float(series.skew()), 3) if len(series) > 2 else np.nan})
    return pd.DataFrame(records)


def leakage_warnings(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """확정 판정이 아닌, 검토할 필요가 있는 잠재적 누수/식별자 신호만 탐색합니다."""
    warnings = []
    for column in df.columns:
        if column == target:
            continue
        series = df[column]
        unique_ratio = series.nunique(dropna=True) / max(series.notna().sum(), 1)
        lower = str(column).lower()
        if (lower == "id" or lower.endswith("_id") or lower.startswith("id_") or lower in {"uuid", "puuid"}):
            warnings.append({"컬럼": column, "검토 신호": "ID 형태 컬럼명", "점검할 내용": "식별자로만 쓰이는지 확인하고 기본 모델에서는 제외 권장"})
        elif unique_ratio > 0.98 and (pd.api.types.is_string_dtype(series) or series.dtype == object):
            warnings.append({"컬럼": column, "검토 신호": "고유값이 많은 문자열", "점검할 내용": "고유 식별자 또는 자유 텍스트인지 확인"})
        if target is not None:
            same = series.eq(df[target]) & series.notna() & df[target].notna()
            compared = (series.notna() & df[target].notna()).sum()
            if compared > 0 and same.sum() == compared:
                warnings.append({"컬럼": column, "검토 신호": "타깃과 값이 완전히 동일", "점검할 내용": "직접 타깃 누수 가능성이 높으므로 제거 검토"})
            if (pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(df[target])
                    and compared >= 10 and series.nunique() > 1 and df[target].nunique() > 1):
                correlation = series.corr(df[target])
                if pd.notna(correlation) and abs(correlation) > 0.995:
                    warnings.append({"컬럼": column, "검토 신호": "타깃과 거의 완전한 선형 관계", "점검할 내용": "예측 시점에 실제로 알 수 있는 변수인지 확인"})
    return pd.DataFrame(warnings, columns=["컬럼", "검토 신호", "점검할 내용"])


def quality_issues(df: pd.DataFrame, target: str | None = None) -> list[str]:
    issues = []
    if df.isna().any().any():
        issues.append(f"결측치가 {int(df.isna().sum().sum()):,}개 있습니다.")
    duplicates = int(df.duplicated().sum())
    if duplicates:
        issues.append(f"완전히 중복된 행이 {duplicates:,}개 있습니다. 단, 반복 거래 등 정상 데이터인지 확인하세요.")
    empty = df.columns[df.isna().all()].tolist()
    if empty:
        issues.append(f"전체가 결측치인 컬럼: {', '.join(map(str, empty[:10]))}")
    constant = [str(c) for c in df.columns if df[c].nunique(dropna=False) == 1]
    if constant:
        issues.append(f"값이 변하지 않는 컬럼: {', '.join(constant[:10])}")
    if target and df[target].isna().any():
        issues.append("타깃 결측 행은 지도학습 전에 제외해야 합니다.")
    if not issues:
        issues.append("기본 검사에서 뚜렷한 품질 경고가 발견되지 않았습니다.")
    return issues
