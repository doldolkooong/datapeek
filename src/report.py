"""통계표 + 다양한 Plotly 그래프가 포함된 단일 HTML 리포트."""

from html import escape

import pandas as pd

from src.analyzer import get_column_info, get_missing_info, get_numeric_summary, get_overview
from src.compare import compare_train_test
from src.quality import leakage_warnings, quality_issues
from src.target import target_distribution
from src.visualizations.gallery import (overview_charts, quality_charts, univariate_charts,
                                        target_charts, relationship_charts, comparison_charts,
                                        anomaly_charts, model_charts, number_columns, category_columns)


def build_html_report(df: pd.DataFrame, filename: str, target: str | None = None,
                      task: str | None = None, test_df: pd.DataFrame | None = None,
                      outliers: pd.DataFrame | None = None, model_result: dict | None = None,
                      max_charts: int = 24) -> str:
    overview = get_overview(df)
    groups = [overview_charts(df), quality_charts(df)]
    numeric = [c for c in number_columns(df) if c != target]
    categorical = [c for c in category_columns(df) if c != target]
    groups.append(univariate_charts(df, numeric[0] if numeric else None, categorical[0] if categorical else None))
    if target is not None and target in df and task:
        groups.append(target_charts(df, target, task))
        features = [c for c in df if c != target]
        if features:
            groups.append(relationship_charts(df, target, features[0], task)[:5])
    if test_df is not None:
        compare = compare_train_test(df, test_df, target)
        common = [c for c in df if c in test_df and c != target]
        if common:
            groups.append(comparison_charts(df, test_df, common[0], compare)[:4])
    if outliers is not None:
        groups.append(anomaly_charts(df, numeric[:2], outliers)[:3])
    if model_result is not None:
        groups.append(model_charts(model_result)[:5])

    # 분야별 그래프가 고르게 포함되도록 라운드 로빈으로 선택합니다.
    charts = []
    while len(charts) < max_charts and any(groups):
        for group in groups:
            if group and len(charts) < max_charts:
                charts.append(group.pop(0))

    sections = []
    for title, fig, tip in charts:
        fig_html = fig.to_html(full_html=False, include_plotlyjs=(len(sections) == 0),
                               config={"displaylogo": False, "responsive": True})
        sections.append(f'<section class="chart"><h3>{escape(title)}</h3>{fig_html}'
                        f'<p class="insight">{escape(tip)}</p></section>')

    table_blocks = [("컬럼 구조", get_column_info(df)), ("결측치 통계", get_missing_info(df)),
                    ("수치형 기초통계", get_numeric_summary(df)),
                    ("누수 의심 신호", leakage_warnings(df, target))]
    if target and task:
        table_blocks.append(("타깃 분포", target_distribution(df, target, task)))
    if test_df is not None:
        table_blocks.append(("Train / Test 분포 비교", compare_train_test(df, test_df, target)))
    if model_result is not None:
        table_blocks.append(("교차검증 성능", model_result["cv"]))
        table_blocks.append(("홀드아웃 점수", pd.DataFrame([model_result["metrics"]])))
    tables = "".join(f"<h2>{escape(name)}</h2><div class='scroll'>{table.to_html(escape=True)}</div>"
                     for name, table in table_blocks if not table.empty)
    issues = "".join(f"<li>{escape(issue)}</li>" for issue in quality_issues(df, target))
    return f'''<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>DataPeek 분석 리포트</title><style>
    body{{font-family:system-ui,-apple-system,'Malgun Gothic',sans-serif;background:#f5f8fc;color:#182638;margin:0}}
    main{{max-width:1140px;margin:30px auto;padding:0 18px}}
    header,section.chart,.panel{{background:white;padding:22px;border:1px solid #e4eaf2;border-radius:16px;margin-bottom:18px}}
    header{{border-top:5px solid #247cab}}h1{{margin:0 0 8px}}h2{{padding-top:12px}}
    h3{{font-size:17px}}.metric{{display:inline-block;background:#eaf3f9;border-radius:8px;padding:12px;margin:6px}}
    .insight{{background:#f0f7fa;padding:12px;border-left:3px solid #14848b;border-radius:5px}}
    table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid #e2e9ef;padding:9px;text-align:left}}
    .scroll{{overflow-x:auto;background:white;border-radius:12px;padding:12px}}.note{{font-size:12px;color:#596777}}
    </style></head><body><main>
    <header><h1>DataPeek 분석 리포트</h1><p>파일: {escape(filename)}</p>
    <div class="metric">행 {overview['rows']:,}</div><div class="metric">컬럼 {overview['columns']:,}</div>
    <div class="metric">결측 셀 {overview['missing_cells']:,}</div>
    <div class="metric">중복 행 {overview['duplicate_rows']:,}</div></header>
    <div class="panel"><h2>데이터 품질 진단</h2><ul>{issues}</ul></div>
    <h2>시각화 분석 ({min(len(charts), max_charts)}개)</h2>{''.join(sections)}
    <h2>상세 통계표</h2>{tables}
    <p class="note">분석 결과는 자동 생성된 탐색 자료입니다. 통계적 연관성은 인과관계나 예측 가능성을 증명하지 않습니다.
    모델에 사용할 전처리값은 학습 폴드에서만 fit해야 합니다.</p></main></body></html>'''
