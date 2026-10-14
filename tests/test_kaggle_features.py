"""확장 분석 모듈에 대한 통합 검증 (python -m unittest discover -s tests -v)."""

import unittest

import numpy as np
import pandas as pd

from src.compare import compare_train_test, population_stability_index
from src.features import make_features
from src.modeling import evaluate_baseline
from src.outliers import detect_outliers
from src.quality import iqr_summary, leakage_warnings
from src.report import build_html_report
from src.target import feature_target_table, infer_task, target_distribution


class TestAdvancedAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generator = np.random.default_rng(12)
        rows = 240
        price = generator.uniform(1, 400, rows)
        usage = generator.integers(1, 60, rows)
        cls.df = pd.DataFrame({
            "customer_id": [f"C{i:04d}" for i in range(rows)],
            "price": price,
            "usage": usage,
            "segment": generator.choice(["A", "B", "C"], rows),
            "last_seen": pd.date_range("2026-01-01", periods=rows, freq="D").strftime("%Y-%m-%d"),
            "target": ((usage < 25) | (price > 320)).astype(int),
        })
        cls.df.loc[:10, "price"] = np.nan

    def test_target_inference(self):
        self.assertEqual(infer_task(self.df["target"]), "분류")
        self.assertEqual(infer_task(self.df["price"]), "회귀")
        self.assertFalse(target_distribution(self.df, "target", "분류").empty)
        self.assertFalse(feature_target_table(self.df, "target", "usage", "분류").empty)

    def test_quality_and_leakage(self):
        warnings = leakage_warnings(self.df, "target")
        self.assertIn("customer_id", warnings["컬럼"].tolist())
        self.assertIn("usage", iqr_summary(self.df)["컬럼"].tolist())
        doubled = self.df.copy()
        doubled["target_copy"] = doubled["target"]
        self.assertIn("target_copy", leakage_warnings(doubled, "target")["컬럼"].tolist())

    def test_train_test_drift(self):
        test = self.df.drop(columns="target").copy()
        test["usage"] += 15
        test["segment"] = "NEW"
        table = compare_train_test(self.df, test, "target")
        self.assertGreater(int(table.loc[table["컬럼"] == "segment", "새 범주 수"].iloc[0]), 0)
        self.assertGreater(float(table.loc[table["컬럼"] == "usage", "PSI"].iloc[0]), 0)
        self.assertTrue(np.isnan(population_stability_index(pd.Series([1, 1]), pd.Series([2, 2]))))

    def test_features(self):
        frame = make_features(self.df[["last_seen", "price", "usage", "segment"]],
                              date_columns=["last_seen"], text_columns=["segment"],
                              log_columns=["usage"], ratio=("price", "usage"))
        self.assertIn("last_seen_month", frame)
        self.assertNotIn("last_seen", frame)
        self.assertIn("price_over_usage", frame)
        self.assertIn("segment_length", frame)
        self.assertFalse(np.isinf(frame.select_dtypes(include="number").to_numpy()).any())

    def test_outliers(self):
        subset = self.df[["price", "usage"]]
        iso = detect_outliers(subset, ["price", "usage"], contamination=0.07)
        self.assertEqual(len(iso), len(subset))
        self.assertGreater(int(iso["이상치 여부"].sum()), 0)
        simple = detect_outliers(subset, ["price"], method="IQR")
        self.assertEqual(len(simple), len(subset))

    def test_baseline_classification(self):
        result = evaluate_baseline(self.df, "target", "분류", ["usage", "price", "segment"],
                                   "로지스틱 회귀", max_rows=240)
        self.assertIn("F1-macro", result["metrics"])
        self.assertIn("오답 여부", result["predictions"])
        self.assertEqual(len(result["importance"]), 3)
        self.assertEqual(result["n_train"] + result["n_valid"], 240)

    def test_baseline_regression_time(self):
        result = evaluate_baseline(self.df, "price", "회귀", ["usage", "segment", "last_seen"],
                                   "Ridge 회귀", strategy="시간", time_column="last_seen",
                                   date_columns=["last_seen"], max_rows=240)
        self.assertIn("RMSE", result["metrics"])
        self.assertIn("잔차 (실제-예측)", result["predictions"])

    def test_baseline_group(self):
        result = evaluate_baseline(self.df, "target", "분류", ["usage", "price", "segment"],
                                   "랜덤 포레스트", strategy="그룹", group_column="customer_id", max_rows=200)
        self.assertIn("Accuracy", result["metrics"])

    def test_report_extension(self):
        report = build_html_report(self.df, "training<bad>.csv", target="target", task="분류",
                                   test_df=self.df.drop(columns="target"))
        self.assertIn("Train / Test 분포 비교", report)
        self.assertIn("training&lt;bad&gt;.csv", report)


if __name__ == "__main__":
    unittest.main()
