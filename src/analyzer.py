"""CSV 읽기와 기초 EDA 분석을 담당하는 모듈."""

from io import StringIO

import pandas as pd


def read_csv(file_bytes: bytes) -> pd.DataFrame:
    """한국어 CSV를 고려하여 UTF-8, CP949 순서로 읽습니다."""
    for encoding in ["utf-8-sig", "cp949"]:
        try:
            csv_text = file_bytes.decode(encoding)
            return pd.read_csv(StringIO(csv_text))
        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 인코딩을 인식할 수 없습니다. UTF-8 또는 CP949로 저장해 주세요.")


def get_overview(df: pd.DataFrame) -> dict:
    """데이터셋 전체 요약 수치를 반환합니다."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼별 자료형, 결측치 수, 고유값 수를 정리합니다."""
    rows = []

    for column in df.columns:
        rows.append(
            {
                "컬럼명": column,
                "자료형": str(df[column].dtype),
                "결측치 수": int(df[column].isna().sum()),
                "고유값 수": int(df[column].nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


def get_missing_info(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼별 결측치 개수와 비율(%)을 계산합니다."""
    missing_count = df.isna().sum()
    missing_rate = missing_count / len(df) * 100

    result = pd.DataFrame(
        {
            "컬럼명": df.columns,
            "결측치 수": missing_count.values,
            "결측 비율(%)": missing_rate.values.round(2),
        }
    )
    return result.sort_values("결측치 수", ascending=False).reset_index(drop=True)


def get_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """수치형 컬럼의 기초 통계량을 반환합니다."""
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty or len(numeric_df.columns) == 0:
        return pd.DataFrame()

    return numeric_df.describe().T.round(2)
