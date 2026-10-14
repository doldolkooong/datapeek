"""외부 라이브러리 없이 실행 가능한 기본 테스트."""

import unittest

import pandas as pd

from src.analyzer import get_missing_info, get_numeric_summary, get_overview, read_csv
from src.report import build_html_report
from src.visualizer import make_correlation_chart


class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "지역": ["서울", "부산", "부산", "부산"],
                "매출": [100, 200, 200, None],
                "수량": [1, 2, 2, 4],
            }
        )

    def test_overview(self):
        result = get_overview(self.df)
        self.assertEqual(result["rows"], 4)
        self.assertEqual(result["columns"], 3)
        self.assertEqual(result["missing_cells"], 1)
        self.assertEqual(result["duplicate_rows"], 1)

    def test_missing(self):
        missing = get_missing_info(self.df)
        self.assertEqual(int(missing.iloc[0]["결측치 수"]), 1)
        self.assertEqual(float(missing.iloc[0]["결측 비율(%)"]), 25.0)

    def test_cp949_file(self):
        csv_bytes = "도시,값\n서울,1\n부산,2\n".encode("cp949")
        result = read_csv(csv_bytes)
        self.assertEqual(result.loc[0, "도시"], "서울")

    def test_numeric(self):
        summary = get_numeric_summary(self.df)
        self.assertIn("매출", summary.index)
        self.assertNotIn("지역", summary.index)

    def test_correlation(self):
        self.assertIsNotNone(make_correlation_chart(self.df))

    def test_report(self):
        html = build_html_report(self.df, "샘플<매출>.csv")
        self.assertIn("DataPeek 분석 리포트", html)
        self.assertIn("샘플&lt;매출&gt;.csv", html)


if __name__ == "__main__":
    unittest.main()
