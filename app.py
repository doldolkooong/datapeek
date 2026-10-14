"""DataPeek Premium — 다크 전용 시각화 + Kaggle 분석 엔진.

실행: python -m streamlit run app.py
"""

from hashlib import sha256
from os.path import splitext

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analyzer import read_csv, get_column_info, get_overview, get_missing_info
from src.ui import set_design, brand, hero, metric_grid, section, insight, mini_cards, chart_theme
from src.compare import compare_train_test
from src.features import make_features
from src.insights import (overview_insight, quality_insight, univariate_insight,
                          target_insight, compare_insight, model_insight)
from src.modeling import evaluate_baseline, explain_with_shap
from src.outliers import detect_outliers
from src.quality import iqr_summary, leakage_warnings, quality_issues
from src.report import build_html_report
from src.statistics import numeric_profile, categorical_profile, top_correlations, target_associations, cramers_v
from src.target import infer_task, target_distribution, feature_target_table
from src.visualizations.gallery import (
    CHART_CATALOG, overview_charts, quality_charts, univariate_charts, target_charts,
    relationship_charts, comparison_charts, anomaly_charts, feature_charts, model_charts,
    number_columns, category_columns,
)

st.set_page_config(page_title="DataPeek | Analytics Studio", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

PAGES = ["◈  대시보드", "▦  데이터 개요", "◉  데이터 품질", "▥  단변량 EDA", "◎  Target 분석",
         "⌘  Feature 관계", "⇆  Train/Test 비교", "◇  이상치 탐지", "⚙  피처 엔지니어링",
         "▣  모델·오답 분석", "▤  HTML 리포트"]


def show_charts(charts, prefix="chart", columns=1, compact=False):
    """모든 그래프를 1행 1개 전체 폭으로 표시합니다.

    기존 함수 호출의 columns 인수는 호환성 때문에 남겨두되 레이아웃에는 사용하지 않습니다.
    """
    if not charts:
        st.info("현재 데이터로 생성할 수 있는 그래프가 없습니다. 컬럼과 분석 유형을 확인하세요.")
        return
    st.markdown(f'<div class="dp-note">그래프 {len(charts)}개 · 마우스를 올리면 상세 값을 볼 수 있습니다.</div>', unsafe_allow_html=True)
    for i, (title, fig, guidance) in enumerate(charts):
        with st.container(border=True):
            st.markdown(f'<div class="dp-chart-label">{html_escape(title)}</div>', unsafe_allow_html=True)
            chart_theme(fig, dashboard=compact)
            st.plotly_chart(fig, use_container_width=True, key=f"{prefix}-{i}", config={"displaylogo": False, "responsive": True})
            st.markdown(f'<div class="dp-chart-hint">✦ {html_escape(guidance)}</div>', unsafe_allow_html=True)


def html_escape(value):
    from html import escape
    return escape(str(value))


def show_insight(text):
    st.markdown(insight(text), unsafe_allow_html=True)


def show_table(label, data):
    with st.expander(f"📋 {label}", expanded=False):
        if data is None or data.empty:
            st.info("해당하는 통계표가 없습니다.")
        else:
            st.dataframe(data, use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False)
def load_csv(raw):
    return read_csv(raw)


@st.cache_data(show_spinner=False)
def calc_comparison(train, test, target):
    return compare_train_test(train, test, target)


with st.sidebar:
    brand()
    st.markdown('<div class="dp-side-heading">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio("분석 메뉴", PAGES, label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="dp-side-heading">데이터셋</div>', unsafe_allow_html=True)
    train_file = st.file_uploader("Train CSV", type="csv")
    test_file = st.file_uploader("Test CSV (선택)", type="csv")
    st.divider()
    set_design()

train_bytes = train_file.getvalue() if train_file else None
test_bytes = test_file.getvalue() if test_file else None
filename = train_file.name if train_file else ""

if train_bytes is None:
    if page == PAGES[0]:
        from src.landing import show_landing
        show_landing(CHART_CATALOG)
    else:
        st.header(page.strip())
        st.warning("📂 데이터셋을 업로드해 주세요.")
        st.info("분석 기능을 사용하려면 왼쪽 사이드바에서 **Train CSV** 파일을 업로드해야 합니다. 업로드 전에는 **대시보드**에서 기능 소개를 확인할 수 있습니다.")
    st.stop()

if len(train_bytes) > 15 * 1024 * 1024 or (test_bytes is not None and len(test_bytes) > 15 * 1024 * 1024):
    st.error("CSV 파일당 최대 15MB까지 지원합니다.")
    st.stop()
try:
    df = load_csv(train_bytes)
    test_df = load_csv(test_bytes) if test_bytes is not None else None
except (UnicodeError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
    st.error(f"CSV 읽기 실패: {exc}")
    st.stop()
if df.empty or df.columns.duplicated().any() or (test_df is not None and (test_df.empty or test_df.columns.duplicated().any())):
    st.error("행이 0개인 데이터 또는 중복된 컬럼명은 지원하지 않습니다.")
    st.stop()

numeric_cols = number_columns(df)
cat_cols = category_columns(df)
data_key = sha256(train_bytes + (test_bytes or b"")).hexdigest()

with st.sidebar:
    with st.expander("⚙ 데이터 분석 설정", expanded=True):
        opts = ["(없음)"] + list(df.columns)
        target_option = st.selectbox("타깃 컬럼", opts, index=0)
        target = target_option if target_option != "(없음)" else None
        automatic_task = infer_task(df[target]) if target else "분류"
        task_choice = st.selectbox("문제 유형", ["자동", "분류", "회귀"], disabled=target is None)
        task = automatic_task if task_choice == "자동" else task_choice
        if target:
            st.caption(f"유형: {task} (데이터 의미 확인 필요)")
    st.caption(f"📄 {filename} · {len(df):,}행")
    st.caption("기본 시각화 최대 3,000행 · 민감정보 업로드 주의")

# 데이터·타깃·문제 유형이 달라지면 이전 검증 결과를 재사용하지 않습니다.
selection = (data_key, target, task)
if st.session_state.get("data_selection") != selection:
    st.session_state["data_selection"] = selection
    for key in ("model_result", "shap_frame", "outlier_result", "report_html", "outlier_settings", "model_settings", "mi_table"):
        st.session_state.pop(key, None)

# 페이지 공통 Hero + KPI 4개. 선택 데이터에서 실제 계산한 값만 표시합니다.
dataset_title = splitext(filename)[0].replace("_", " ") or "데이터 분석 프로젝트"
hero(dataset_title, "데이터를 탐색하고, 패턴을 발견하고, 분석 결과를 시각적으로 해석해 보세요.")
missing_percentage = df.isna().sum().sum() / max(df.size, 1) * 100
memory_megabytes = df.memory_usage(deep=True).sum() / (1024 * 1024)
metric_grid(len(df), len(df.columns), missing_percentage, memory_megabytes)
summary = get_overview(df)

if page == PAGES[0]:
    section("1", "데이터 개요", "데이터의 전체 규모, 타깃 분포와 주요 품질 지표를 한눈에 확인합니다.")
    lead_charts = []
    if target:
        target_set = target_charts(df, target, task)
        lead_charts += [item for item in target_set if "Donut" in item[0]][:1]
        if not lead_charts:
            lead_charts += target_set[:1]
    lead_charts += overview_charts(df)[:1] + quality_charts(df)[:1]
    show_charts(lead_charts, "home-overview", columns=1, compact=False)

    section("2", "데이터 탐색 및 주요 패턴", "변수의 분포와 관계를 살펴보고 추가 분석이 필요한 특징을 찾습니다.")
    numeric_first = next((c for c in numeric_cols if c != target), None)
    feature_first = next((c for c in cat_cols if c != target and 2 <= df[c].nunique(dropna=True) <= 15 and not c.lower().endswith("_id")), None)
    secondary_charts = []
    if numeric_first:
        secondary_charts += univariate_charts(df, numeric_first, None)[:1]
    if len(numeric_cols) >= 2:
        rel = relationship_charts(df, target, numeric_first, task)
        secondary_charts += [item for item in rel if "Heatmap" in item[0]][:1]
    if target and feature_first:
        rel = relationship_charts(df, target, feature_first, task)
        secondary_charts += [item for item in rel if "Normalized Stack" in item[0] or "Box" in item[0]][:1]
    if len(secondary_charts) < 3:
        secondary_charts += overview_charts(df)[1:1+(3-len(secondary_charts))]
    show_charts(secondary_charts[:3], "home-patterns", columns=1, compact=False)

    section("3", "인사이트 및 분석 요약", "지표와 그래프를 함께 해석하고 다음 작업을 결정합니다.")
    target_text = target_insight(df, target, task) if target else "타깃 컬럼을 설정하면 분류·회귀 분석이 활성화됩니다."
    mini_cards([
        ("01 · DATA QUALITY", "데이터 상태", quality_insight(df)),
        ("02 · TARGET", "타깃 진단", target_text),
        ("03 · NEXT STEP", "권장 분석", "사이드바의 Target 분석, Feature 관계, Train/Test 비교를 차례대로 확인해 보세요."),
    ])
    st.caption("※ 자동 분석은 탐색적 진단입니다. 원인·인과관계나 모델의 성능 향상을 보장하지 않습니다.")

elif page == PAGES[1]:
    st.header("01 · 데이터 개요")
    st.write("원본 데이터의 크기·자료형·고유값·메모리를 분석합니다.")
    show_insight(overview_insight(df))
    show_charts(overview_charts(df), "overview")
    show_table("원본 데이터 (상위 100행)", df.head(100))
    show_table("컬럼별 기본 통계", get_column_info(df))
    show_table("수치형 상세 통계", numeric_profile(df))
    show_table("범주형 상세 통계", categorical_profile(df))

elif page == PAGES[2]:
    st.header("02 · 데이터 품질 진단")
    issues = quality_issues(df, target)
    with st.expander(f"품질 검사 결과 ({len(issues)}건)", expanded=True):
        for issue in issues:
            st.write("• " + issue)
    show_insight(quality_insight(df))
    show_charts(quality_charts(df), "quality")
    show_table("컬럼별 결측률", get_missing_info(df))
    show_table("IQR 이상치 및 왜도", iqr_summary(df))
    show_table("누수·ID 후보 경고", leakage_warnings(df, target))
    st.warning("누수 경고는 규칙 기반 후보이며, 실제 누수 여부는 데이터 생성·예측 시점 확인이 필요합니다.")

elif page == PAGES[3]:
    st.header("03 · 단변량 EDA")
    st.write("수치형의 분포·왜도·분위수와 범주형 빈도·누적 점유율을 함께 봅니다.")
    a, b = st.columns(2)
    with a:
        numeric = st.selectbox("분석할 수치형 변수", ["(없음)"] + numeric_cols)
    with b:
        categorical = st.selectbox("분석할 범주형 변수", ["(없음)"] + cat_cols)
    n = None if numeric == "(없음)" else numeric
    c = None if categorical == "(없음)" else categorical
    show_insight(univariate_insight(df, n, c))
    show_charts(univariate_charts(df, n, c), "univariate")
    if n:
        show_table(f"{n} 기술통계", numeric_profile(df[[n]]))
    if c:
        show_table(f"{c} 빈도 상위 20개", df[c].fillna("(결측)").astype(str).value_counts().head(20).rename_axis("범주").reset_index(name="건수"))

elif page == PAGES[4]:
    st.header("04 · Target 분석")
    if target is None:
        st.info("왼쪽 사이드바에서 타깃 컬럼을 선택하세요.")
    else:
        st.write(f"**{target}** · 현재 분석 유형: **{task}**")
        if task == "분류":
            counts = df[target].value_counts(normalize=True, dropna=True)
            if len(counts) > 1:
                x, y = st.columns(2)
                x.metric("클래스 수", len(counts))
                y.metric("최소 클래스 비율", f"{counts.min()*100:.2f}%")
                if counts.min() < .1:
                    st.warning("최소 클래스가 10% 미만입니다. Accuracy 외에 F1-macro / PR Curve도 고려하세요.")
        else:
            col = pd.to_numeric(df[target], errors="coerce")
            x, y, z = st.columns(3)
            x.metric("타깃 평균", f"{col.mean():,.3f}")
            y.metric("타깃 중앙값", f"{col.median():,.3f}")
            z.metric("타깃 왜도", f"{col.skew():.3f}")
        show_insight(target_insight(df, target, task))
        show_charts(target_charts(df, target, task), "target")
        show_table("타깃 상세 통계", target_distribution(df, target, task))

elif page == PAGES[5]:
    st.header("05 · Feature Relationship")
    st.write("피처 간 상관관계 및 피처-타깃 관계를 여러 그래프로 확인합니다.")
    feats = [col for col in df if col != target]
    if not feats:
        st.info("피처 컬럼이 없습니다.")
    else:
        a, b = st.columns(2)
        with a:
            feature = st.selectbox("관찰할 피처", feats)
        with b:
            other = st.selectbox("함께 비교할 수치형 피처", ["(자동)"] + [c for c in numeric_cols if c != feature])
        heavy = st.checkbox("Scatter Matrix 추가 (최대 4개 컬럼 / 400행)")
        show_charts(relationship_charts(df, target, feature, task, None if other == "(자동)" else other, heavy), "relations")
        show_table("Pearson 강한 상관관계 쌍", top_correlations(df, "pearson"))
        show_table("Spearman 강한 상관관계 쌍", top_correlations(df, "spearman"))
        if target is not None:
            show_table("선택 피처와 타깃의 통계 관계", feature_target_table(df, target, feature, task))
            if task == "분류" and not pd.api.types.is_numeric_dtype(df[feature]):
                score = cramers_v(df, feature, target)
                if pd.notna(score):
                    st.metric("Cramér's V (범주형 연관 정도)", f"{score:.3f}")
            if st.button("수치형 피처 Mutual Information 계산"):
                st.session_state["mi_table"] = target_associations(df, target, task)
            if "mi_table" in st.session_state:
                show_table("Mutual Information (탐색용)", st.session_state["mi_table"])
                st.caption("MI 계산은 탐색용이며, 피처 선택 과정은 교차검증 안에서 수행해야 합니다.")

elif page == PAGES[6]:
    st.header("06 · Train / Test Distribution")
    if test_df is None:
        st.info("사이드바에서 Test CSV도 올리면 분포 변화·PSI·KS·새 범주를 분석합니다.")
    else:
        compare = calc_comparison(df, test_df, target)
        if compare.empty:
            st.warning("Train과 Test에 공통된 피처가 없습니다.")
        else:
            common = compare["컬럼"].tolist()
            column = st.selectbox("비교할 컬럼", common)
            st.metric("비교 가능한 컬럼", len(common))
            show_insight(compare_insight(compare))
            show_charts(comparison_charts(df, test_df, column, compare), "compare")
            show_table("분포 차이 통계 (KS/PSI/결측/새 범주)", compare)
            missing = [c for c in df if c not in test_df and c != target]
            if missing:
                st.warning("Test에 없는 피처: " + ", ".join(missing[:15]))
            st.caption("KS p값은 표본 수의 영향을 크게 받습니다. PSI 임계값은 절대적 판정 기준이 아닙니다.")

elif page == PAGES[7]:
    st.header("07 · Outlier Detection")
    candidates = [c for c in numeric_cols if c != target]
    if not candidates:
        st.info("타깃을 제외하고 최소 하나의 수치형 피처가 필요합니다.")
    else:
        columns = st.multiselect("이상치 탐지 피처", candidates, default=candidates[:2])
        a, b = st.columns(2)
        with a:
            method = st.selectbox("탐지 방법", ["Isolation Forest", "IQR"])
        with b:
            contamination = st.slider("예상 이상치 비율", 0.01, 0.30, 0.05, 0.01, disabled=method == "IQR")
        settings = (tuple(columns), method, contamination)
        if st.session_state.get("outlier_settings") != settings:
            st.session_state.pop("outlier_result", None)
            st.session_state["outlier_settings"] = settings
        if st.button("이상치 계산", type="primary"):
            try:
                with st.spinner("이상치 후보를 찾고 있습니다..."):
                    st.session_state["outlier_result"] = detect_outliers(df, columns, method, contamination)
            except (ValueError, TypeError) as exc:
                st.error(str(exc))
        flags = st.session_state.get("outlier_result")
        if flags is not None:
            count = int(flags["이상치 여부"].sum())
            a, b = st.columns(2)
            a.metric("이상치 후보", f"{count:,}건")
            b.metric("탐지 비율", f"{count/len(df)*100:.2f}%")
            heavy = st.checkbox("PCA 2차원 시각화 추가 (필요할 때만 실행)")
            show_charts(anomaly_charts(df, columns, flags, heavy), "anomaly")
            result_data = df.copy()
            result_data["anomaly_flag"] = flags["이상치 여부"].astype(int)
            result_data["anomaly_score"] = flags["이상 점수"]
            show_table("이상치 후보 상위 100행", result_data.loc[result_data["anomaly_flag"] == 1].head(100))
            st.download_button("이상치 표시 결과 CSV 다운로드", result_data.to_csv(index=False).encode("utf-8-sig"),
                               file_name="datapeek_outliers.csv", mime="text/csv")
            st.caption("이상치 탐지는 비지도 모델의 후보 식별이며 오류 판정이 아닙니다.")

elif page == PAGES[8]:
    st.header("08 · Feature Engineering")
    st.write("피처 변환이 분포에 어떤 차이를 만드는지 시각적으로 확인합니다. 원본 CSV는 변경하지 않습니다.")
    nums = [c for c in numeric_cols if c != target]
    if nums:
        numeric = st.selectbox("변환할 수치형 컬럼", nums)
        ratio_to = st.selectbox("비율 분모 (선택)", ["(없음)"] + [c for c in nums if c != numeric])
        show_charts(feature_charts(df, numeric, ratio_to if ratio_to != "(없음)" else None), "feature")
        st.caption("모델 학습 시 인코딩·결측 대체·스케일링은 별도 Pipeline에서 폴드마다 적합합니다.")
    else:
        st.info("수치형 피처가 필요합니다.")
    with st.expander("기존 파생 피처 만들기 함수 사용 예시"):
        dates = [c for c in df.columns if any(x in c.lower() for x in ["date", "time", "날짜", "일자"])]
        st.code('make_features(df, date_columns=["event_date"], log_columns=["sessions"])', language="python")
        if st.button("업로드 데이터 상위 10행 피처 미리보기"):
            try:
                st.dataframe(make_features(df.head(10), date_columns=dates).head(10), use_container_width=True)
            except (ValueError, KeyError) as exc:
                st.error(str(exc))

elif page == PAGES[9]:
    st.header("09 · Baseline Model & Error Analysis")
    if target is None:
        st.info("사이드바에서 타깃 컬럼을 선택하면 베이스라인 모델을 검증할 수 있습니다.")
    else:
        available = [c for c in df if c != target]
        defaults = [c for c in available if c.lower() != "id" and not c.lower().endswith("_id")]
        features = st.multiselect("학습에 사용할 피처 (누수·ID·사후정보 반드시 제외)", available, default=defaults)
        strings = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
        date_defaults = [c for c in strings if any(s in c.lower() for s in ("date", "time", "일자", "날짜"))]
        with st.expander("날짜·문자열·로그·비율 파생변수", expanded=False):
            date_columns = st.multiselect("날짜 컬럼", strings, default=date_defaults)
            text_columns = st.multiselect("문자열 길이", [c for c in strings if c not in date_columns])
            numbers = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
            log_columns = st.multiselect("log1p 변환 (0 이상)", numbers)
            ratio = None
            if len(numbers) >= 2 and st.checkbox("비율 피처 생성"):
                a, b = st.columns(2)
                with a:
                    numerator = st.selectbox("분자", numbers)
                with b:
                    denominator = st.selectbox("분모", [c for c in numbers if c != numerator])
                ratio = (numerator, denominator)
        a, b, c = st.columns(3)
        with a:
            model = st.selectbox("모델", ["랜덤 포레스트", "로지스틱 회귀" if task == "분류" else "Ridge 회귀"])
        with b:
            strategy = st.selectbox("검증 전략", ["랜덤", "그룹", "시간"])
        with c:
            max_rows = st.selectbox("최대 학습 행 수", [1000, 3000, 5000, 10000], index=1)
        group_column, time_column = None, None
        if strategy == "그룹":
            group_column = st.selectbox("같은 그룹 분리 금지 컬럼", available)
        if strategy == "시간":
            time_column = st.selectbox("날짜/시간 정렬 기준", available)
        settings = (tuple(features), model, strategy, max_rows, group_column, time_column,
                    tuple(date_columns), tuple(text_columns), tuple(log_columns), ratio)
        if st.session_state.get("model_settings") != settings:
            st.session_state["model_settings"] = settings
            st.session_state.pop("model_result", None)
            st.session_state.pop("shap_frame", None)
        st.caption("결측 대체·인코딩은 Pipeline 안에서 학습 폴드별 fit · 20% 홀드아웃 분리")
        if st.button("교차검증 + 오답 분석 실행", type="primary"):
            try:
                with st.spinner("Cross Validation · Holdout · Permutation Importance 계산 중..."):
                    st.session_state.model_result = evaluate_baseline(
                        df, target, task, features, model, strategy, group_column, time_column,
                        date_columns, text_columns, log_columns, ratio, max_rows)
                    st.session_state.pop("shap_frame", None)
            except (ValueError, TypeError, KeyError) as exc:
                st.session_state.pop("model_result", None)
                st.error(f"모델 학습/평가 실패: {exc}")
        result = st.session_state.get("model_result")
        if result is not None:
            st.success(f"{model} · {strategy} 검증 · 학습 {result['n_train']:,}행 / 검증 {result['n_valid']:,}행")
            cards = st.columns(min(len(result["metrics"]), 4))
            for i, (key, value) in enumerate(result["metrics"].items()):
                cards[i].metric(key, f"{value:.4f}")
            show_insight(model_insight(result))
            show_charts(model_charts(result), "model")
            show_table("교차검증 점수", result["cv"])
            show_table("Permutation Importance 통계", result["importance"])
            predictions = result["predictions"]
            if task == "분류":
                show_table("오분류 상위 100행", predictions[predictions["오답 여부"]].head(100))
            else:
                show_table("절대오차 상위 100행", predictions.sort_values("절대오차", ascending=False).head(100))
            st.download_button("홀드아웃 예측 결과 CSV", predictions.to_csv(index=False).encode("utf-8-sig"),
                               file_name="datapeek_predictions.csv", mime="text/csv")
            st.subheader("SHAP 분석 (선택 설치)")
            st.caption("랜덤 포레스트에 한하여 홀드아웃 최대 30행·인코딩 피처 최대 150개로 계산합니다.")
            if st.button("SHAP 중요도 계산", disabled=model != "랜덤 포레스트"):
                try:
                    st.session_state.shap_frame = explain_with_shap(result)
                except (ValueError, TypeError) as exc:
                    st.error(str(exc))
            if "shap_frame" in st.session_state:
                shap = st.session_state.shap_frame
                fig = px.bar(shap.sort_values("평균 |SHAP|"), x="평균 |SHAP|", y="피처", orientation="h",
                             title="SHAP 평균 절댓값 중요도")
                chart_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("SHAP은 모델의 출력에 대한 설명이며 인과관계를 보여주지 않습니다.")
        st.warning("⚠️ 홀드아웃을 반복해 특성을 고르면 홀드아웃에도 과적합할 수 있습니다. 캐글 최종 검증에서는 OOF/시간/그룹 분할을 고려하세요.")

elif page == PAGES[10]:
    st.header("10 · HTML 분석 리포트")
    st.write("기초 통계·데이터 품질·타깃·Train/Test·이상치·모델 진단을 그래프와 함께 HTML로 저장합니다.")
    max_charts = st.slider("리포트에 넣을 최대 그래프 수", 6, 36, 24, 2)
    if st.button("종합 HTML 리포트 생성", type="primary"):
        try:
            with st.spinner("분석 차트와 통계표를 하나의 HTML로 만들고 있습니다..."):
                st.session_state.report_html = build_html_report(
                    df, filename, target, task if target else None, test_df,
                    st.session_state.get("outlier_result"), st.session_state.get("model_result"), max_charts)
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
    if "report_html" in st.session_state:
        st.download_button("📥 DataPeek HTML 리포트 다운로드", st.session_state.report_html.encode("utf-8"),
                           file_name="datapeek_visual_report.html", mime="text/html")
    with st.expander("등록된 전체 그래프 유형"):
        for area, names in CHART_CATALOG.items():
            st.markdown(f"**{area}** · {len(names)}종: " + ", ".join(names))
    st.caption("리포트는 오프라인에서도 열 수 있도록 Plotly JavaScript를 포함합니다. 그래프 개수에 따라 파일이 커질 수 있습니다.")
