"""DataPeek v3 그래프 생성 스모크 테스트. 그래프를 저장하지 않고 Figure 객체를 확인."""
import unittest

import numpy as np
import pandas as pd
from plotly.graph_objects import Figure

from src.compare import compare_train_test
from src.modeling import evaluate_baseline
from src.outliers import detect_outliers
from src.report import build_html_report
from src.statistics import numeric_profile, categorical_profile, top_correlations, target_associations, cramers_v
from src.visualizations.gallery import (
    CHART_CATALOG, overview_charts, quality_charts, univariate_charts, target_charts,
    relationship_charts, comparison_charts, anomaly_charts, feature_charts, model_charts)

class TestVisualGallery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 설치용 샘플 CSV를 포함하지 않고, 테스트에서만 임의 데이터를 생성합니다.
        rng = np.random.default_rng(42)
        n = 500
        sessions = rng.integers(1, 45, n)
        spend = np.round(rng.normal(85_000, 24_000, n).clip(5_000), 0)
        complaints = rng.integers(0, 5, n)
        cls.train = pd.DataFrame({
            'customer_id': [f'C{i:05d}' for i in range(n)],
            'event_date': pd.date_range('2026-01-01', periods=n).strftime('%Y-%m-%d'),
            'age': rng.integers(18, 75, n),
            'tenure_months': rng.integers(1, 90, n),
            'monthly_spend': spend,
            'sessions': sessions,
            'complaints': complaints,
            'region': rng.choice(['서울', '부산', '인천', '대전'], n),
            'churn': ((sessions < 13) | (complaints >= 4)).astype(int),
        })
        cls.train.loc[::37, 'monthly_spend'] = np.nan
        cls.test = cls.train.drop(columns=['churn']).iloc[:130].copy()
        cls.test['sessions'] += 4

    def check_charts(self, charts, minimum):
        self.assertGreaterEqual(len(charts), minimum)
        for title, fig, interpretation in charts:
            self.assertIsInstance(fig, Figure, title)
            self.assertTrue(interpretation)
            self.assertTrue(title)

    def test_catalog(self):
        self.assertGreaterEqual(sum(len(v) for v in CHART_CATALOG.values()), 60)

    def test_overview_and_quality(self):
        self.check_charts(overview_charts(self.train), 3)
        self.check_charts(quality_charts(self.train), 6)

    def test_univariate(self):
        self.check_charts(univariate_charts(self.train, 'sessions', 'region'), 10)

    def test_target(self):
        self.check_charts(target_charts(self.train, 'churn', '분류'), 3)
        frame = self.train.copy()
        frame['continuous_target'] = frame['monthly_spend']
        self.check_charts(target_charts(frame, 'continuous_target', '회귀'), 3)

    def test_relations(self):
        self.check_charts(relationship_charts(self.train, 'churn', 'sessions', '분류', heavy=True), 8)
        self.check_charts(relationship_charts(self.train, 'churn', 'region', '분류'), 7)
        frame = self.train.copy()
        self.check_charts(relationship_charts(frame, 'monthly_spend', 'sessions', '회귀'), 5)
        self.check_charts(relationship_charts(frame, 'monthly_spend', 'region', '회귀'), 5)

    def test_comparison(self):
        comparison = compare_train_test(self.train, self.test, 'churn')
        self.check_charts(comparison_charts(self.train, self.test, 'sessions', comparison), 5)
        self.check_charts(comparison_charts(self.train, self.test, 'region', comparison), 5)

    def test_anomaly(self):
        cols = ['sessions', 'monthly_spend']
        flags = detect_outliers(self.train, cols, contamination=.05)
        self.check_charts(anomaly_charts(self.train, cols, flags, heavy=True), 6)

    def test_features(self):
        self.check_charts(feature_charts(self.train, 'sessions', 'monthly_spend'), 3)

    def test_model_charts(self):
        classification = evaluate_baseline(self.train, 'churn', '분류',
                                           ['sessions', 'monthly_spend', 'region'], '로지스틱 회귀', max_rows=150)
        self.assertIn('양성 확률', classification['predictions'])
        self.check_charts(model_charts(classification), 7)
        regression = evaluate_baseline(self.train, 'monthly_spend', '회귀',
                                       ['sessions', 'region'], 'Ridge 회귀', max_rows=150)
        self.check_charts(model_charts(regression), 5)

    def test_statistics_and_report(self):
        self.assertFalse(numeric_profile(self.train).empty)
        self.assertFalse(categorical_profile(self.train).empty)
        self.assertFalse(top_correlations(self.train).empty)
        self.assertFalse(target_associations(self.train, 'churn', '분류').empty)
        self.assertGreaterEqual(cramers_v(self.train, 'region', 'churn'), 0)
        html = build_html_report(self.train, 'example<script>.csv', 'churn', '분류', self.test, max_charts=16)
        self.assertIn('DataPeek 분석 리포트', html)
        self.assertIn('example&lt;script&gt;.csv', html)
        self.assertGreater(html.count('plotly-graph-div'), 10)


if __name__ == '__main__':
    unittest.main()
