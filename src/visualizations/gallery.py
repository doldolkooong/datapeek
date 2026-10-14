"""데이터 타입에 맞춰 여러 분석 그래프를 만드는 시각화 갤러리.

각 함수는 (그래프 제목, Plotly Figure, 해석 가이드) 튜플의 리스트를 반환합니다.
데이터가 맞지 않는 그래프는 억지로 그리지 않고 생략합니다.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.quality import iqr_summary

CHART_LIMIT = 3000  # 렌더링 과부하를 막기 위한 그래프 표본 상한


def sample_rows(df, n=CHART_LIMIT):
    return df.sample(n, random_state=42) if len(df) > n else df.copy()


def number_columns(df, exclude=()):
    return [c for c in df.select_dtypes(include="number").columns if c not in exclude]


def category_columns(df, exclude=()):
    return [c for c in df.columns if c not in number_columns(df) and c not in exclude]


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def top_category_series(series, n=12):
    value = series.fillna("(결측)").astype(str)
    allowed = value.value_counts().head(n).index
    return value.where(value.isin(allowed), "기타")


def chart(title, fig, tip):
    fig.update_layout(template="plotly_white", margin=dict(l=12, r=12, t=48, b=24),
                      title=dict(text=title, font=dict(size=15)), legend_title_text="")
    return (title, fig, tip)


def overview_charts(df):
    """01 / 자료형, 고유값, 메모리 사용량: 총 3종."""
    output = []
    column_types = df.dtypes.astype(str).value_counts()
    output.append(chart("자료형 구성 · Donut", px.pie(names=column_types.index, values=column_types.values, hole=0.55),
                        "숫자·문자·날짜 컬럼의 구성을 확인합니다."))
    unique = df.nunique(dropna=True).sort_values(ascending=False).head(25)
    output.append(chart("컬럼별 고유값 개수 · Bar", px.bar(x=unique.index.astype(str), y=unique.values,
                                                          labels={"x": "컬럼", "y": "고유값 수"}),
                        "고유값이 거의 행 개수와 같다면 ID 성격인지 살펴보세요."))
    memory = df.memory_usage(index=False, deep=True).drop("Index", errors="ignore").sort_values(ascending=False).head(20)
    output.append(chart("컬럼별 메모리 사용량 · Horizontal Bar", px.bar(x=memory.values / 1024, y=memory.index,
                              orientation="h", labels={"x": "KiB", "y": "컬럼"}),
                        "메모리 점유가 큰 텍스트 컬럼을 확인합니다."))
    return output


def quality_charts(df):
    """02 / 결측·완성도·왜도·IQR: 총 7종."""
    output = []
    miss = (df.isna().mean() * 100).sort_values(ascending=False).head(30)
    output.append(chart("컬럼별 결측률 · Bar", px.bar(x=miss.index, y=miss.values,
                                  labels={"x": "컬럼", "y": "결측률 (%)"}),
                        "결측률과 결측 발생 원인을 함께 검토하세요."))
    view = sample_rows(df, 200).iloc[:, :40]
    mask = view.isna().astype(int)
    fig = px.imshow(mask.T.to_numpy(), aspect="auto", color_continuous_scale=[[0, "#e7eff7"], [1, "#ed806f"]],
                    labels={"x": "행 (최대 200)", "y": "컬럼", "color": "결측 여부"})
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(view.columns))), ticktext=list(map(str, view.columns)))
    output.append(chart("결측 위치 · Missing Heatmap", fig, "결측이 특정 행이나 컬럼에 모여 발생하는지 확인합니다."))
    row_missing = df.isna().sum(axis=1)
    output.append(chart("행별 결측 컬럼 개수 · Histogram", px.histogram(x=row_missing,
                          labels={"x": "행별 결측 컬럼 수", "y": "행 수"}), "결측이 한 행에 집중되는지 확인합니다."))
    patterns = df.isna().iloc[:, :12].astype(int).astype(str).agg("".join, axis=1).value_counts().head(10)
    labels = ["결측 없음" if "1" not in value else f"패턴 {i+1}" for i, value in enumerate(patterns.index)]
    output.append(chart("주요 결측 패턴 · Bar", px.bar(x=labels, y=patterns.values,
                          labels={"x": "주요 결측 패턴 (앞 12개 컬럼)", "y": "행 수"}),
                        "비슷한 결측 패턴이 반복되면 수집 절차를 확인해 보세요."))
    completeness = ((1 - df.isna().mean(axis=1)) * 100)
    output.append(chart("행별 데이터 완성도 · ECDF", px.ecdf(x=completeness,
                         labels={"x": "행별 입력률 (%)", "y": "누적 비율"}),
                        "데이터가 대부분 채워진 행의 비율을 확인할 수 있습니다."))
    numeric = df.select_dtypes(include="number")
    if len(numeric.columns):
        skew = numeric.skew(numeric_only=True).replace([np.inf, -np.inf], np.nan).dropna().abs().sort_values(ascending=False).head(20)
        if not skew.empty:
            output.append(chart("절댓값 기준 왜도 · Horizontal Bar", px.bar(x=skew.values, y=skew.index,
                                       orientation="h", labels={"x": "|왜도|", "y": "컬럼"}),
                                "왜도가 크면 긴 꼬리·극단값이 존재할 가능성이 있습니다."))
        iqr = iqr_summary(df).sort_values("비율(%)", ascending=False).head(20)
        if not iqr.empty:
            output.append(chart("IQR 이상치 후보 비율 · Bar", px.bar(iqr, x="컬럼", y="비율(%)"),
                                "IQR 기준을 벗어난 값이 곧 잘못된 값이라는 뜻은 아닙니다."))
    return output


def univariate_charts(df, numeric=None, categorical=None):
    """03 / 수치 분포 7종 + 범주 빈도 4종."""
    output = []
    frame = sample_rows(df)
    if numeric and numeric in frame:
        values = clean_numeric(frame[numeric])
        if len(values) >= 3:
            output.append(chart("수치 분포 · Histogram", px.histogram(x=values, nbins=35, labels={"x": numeric}),
                                "치우침, 봉우리 개수와 극단값을 확인합니다."))
            output.append(chart("중앙값·이상치 · Box", px.box(y=values, points="outliers", labels={"y": numeric}),
                                "중앙값과 사분위수, 극단값을 확인합니다."))
            output.append(chart("분포 형태 · Violin", px.violin(y=values, box=True, points=False, labels={"y": numeric}),
                                "넓은 부분은 데이터가 밀집된 구간입니다."))
            output.append(chart("누적 분포 · ECDF", px.ecdf(x=values, labels={"x": numeric}),
                                "특정 값 이하가 전체에서 차지하는 비율을 확인합니다."))
            quant = values.quantile([.01, .05, .10, .25, .5, .75, .90, .95, .99])
            output.append(chart("분위수 프로파일 · Line", px.line(x=quant.index * 100, y=quant.values, markers=True,
                                 labels={"x": "백분위수", "y": numeric}),
                                "하위·상위 분위수의 변화가 비대칭적인지 확인합니다."))
            if values.nunique() > 3 and values.std() > 0:
                x = np.linspace(values.min(), values.max(), 180)
                try:
                    density = stats.gaussian_kde(values.to_numpy())(x)
                    output.append(chart("부드러운 분포 · KDE Line", px.line(x=x, y=density,
                                        labels={"x": numeric, "y": "확률밀도"}),
                                        "KDE는 평활화한 추정 분포이며, 원본 빈도와 다를 수 있습니다."))
                except (ValueError, np.linalg.LinAlgError):
                    pass
                ordered = np.sort(values.to_numpy())
                theoretical = stats.norm.ppf((np.arange(len(ordered)) + .5) / len(ordered))
                output.append(chart("정규성 탐색 · Q-Q Scatter", px.scatter(x=theoretical, y=ordered,
                                   labels={"x": "정규분포 이론 분위수", "y": numeric}),
                                    "점들이 직선 형태에서 크게 벗어나면 정규분포 가정이 맞지 않을 수 있습니다."))
    if categorical and categorical in frame:
        values = top_category_series(frame[categorical])
        counts = values.value_counts()
        table = pd.DataFrame({"범주": counts.index, "건수": counts.values})
        output.append(chart("범주별 건수 · Bar", px.bar(table, x="범주", y="건수"), "상위 범주가 지나치게 많은지 확인합니다."))
        output.append(chart("범주 비중 · Donut", px.pie(table, names="범주", values="건수", hole=.5),
                            "소수의 범주로 분포가 치우치는지 확인합니다."))
        output.append(chart("범주 계층 비중 · Treemap", px.treemap(table, path=["범주"], values="건수"),
                            "사각형의 넓이가 범주별 빈도를 나타냅니다."))
        output.append(chart("누적 범주 점유율 · Pareto Line", px.line(x=counts.index, y=counts.cumsum() / counts.sum() * 100,
                            markers=True, labels={"x": "범주(빈도순)", "y": "누적 비중 (%)"}),
                            "상위 몇 개 범주가 데이터 대부분을 차지하는지 확인합니다."))
    return output


def target_charts(df, target, task):
    """04 / 분류와 회귀 각각 타깃 전용 차트."""
    output = []
    values = df[target].dropna()
    if values.empty:
        return output
    if task == "분류":
        count = values.astype(str).value_counts().head(30)
        table = pd.DataFrame({"클래스": count.index, "건수": count.values, "비율": count.values / len(values) * 100})
        output.append(chart("클래스별 표본 수 · Bar", px.bar(table, x="클래스", y="건수"),
                            "소수 클래스가 적으면 F1-macro 등 불균형 대응 지표를 살펴보세요."))
        output.append(chart("클래스별 비율 · Donut", px.pie(table, names="클래스", values="건수", hole=.5),
                            "다수 클래스만 예측하는 모델은 Accuracy가 높아도 의미가 없을 수 있습니다."))
        output.append(chart("클래스 불균형 순위 · Horizontal Bar", px.bar(table, x="비율", y="클래스", orientation="h"),
                            "각 클래스가 전체의 몇 퍼센트인지 비교합니다."))
    else:
        numeric = clean_numeric(values)
        if len(numeric) < 3:
            return output
        output.append(chart("회귀 타깃 분포 · Histogram", px.histogram(x=numeric, nbins=40),
                            "회귀 타깃의 비대칭과 극단값을 살펴봅니다."))
        output.append(chart("회귀 타깃 극단값 · Box", px.box(y=numeric, points="outliers"),
                            "극단값이 손실 함수에 큰 영향을 줄 수 있습니다."))
        if (numeric >= 0).all() and numeric.nunique() > 3:
            transformed = pd.DataFrame({"원본": numeric, "log1p": np.log1p(numeric)})
            melted = transformed.melt(var_name="버전", value_name="값")
            output.append(chart("타깃 원본 vs 로그 변환 · Facet Histogram",
                                px.histogram(melted, x="값", facet_col="버전", nbins=35),
                                "로그 변환은 0 이상인 값에서만 자동 제안합니다. 평가 지표와 역변환을 확인하세요."))
        if numeric.nunique() > 3 and numeric.std() > 0:
            ordered = np.sort(numeric)
            theoretical = stats.norm.ppf((np.arange(len(ordered)) + .5) / len(ordered))
            output.append(chart("타깃 정규성 · Q-Q Scatter", px.scatter(x=theoretical, y=ordered),
                                "정규성을 강제할 필요는 없지만 타깃 분포의 형태를 이해하는 데 도움이 됩니다."))
    return output



def correlation_heatmap(matrix, method="Pearson"):
    """각 상관계수 셀을 숫자/구분선으로 표현한 Plotly 히트맵.

    r은 비율(%)이 아니라 -1~1 범위의 상관계수입니다.
    """
    names = [str(c) for c in matrix.columns]
    values = matrix.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        z=values, x=names, y=names, zmin=-1, zmax=1,
        colorscale="RdBu_r", xgap=3, ygap=3, hoverongaps=False,
        colorbar=dict(title="상관계수 r", thickness=15, len=.77),
        hovertemplate="%{y} × %{x}<br>상관계수: %{z:.3f}<extra></extra>",
        meta="datapeek_corr",
    ))
    for row, y in enumerate(names):
        for col, x in enumerate(names):
            value = values[row, col]
            if np.isfinite(value):
                fg = "#FFFFFF" if abs(value) >= .56 else "#17223B"
                fig.add_annotation(x=x, y=y, text=f"{value:.2f}", showarrow=False,
                                   font=dict(size=12, color=fg))
            else:
                fig.add_annotation(x=x, y=y, text="N/A", showarrow=False,
                                   font=dict(size=11, color="#8896AF"))
    fig.update_xaxes(side="bottom", tickangle=-38, automargin=True)
    fig.update_yaxes(autorange="reversed", automargin=True)
    return fig


def relationship_charts(df, target, feature, task, second_feature=None, heavy=False):
    """05 / 타깃·피처/피처·피처 관계를 여러 형태로 표시."""
    output = []
    view = sample_rows(df)
    numbers = number_columns(view)
    if len(numbers) >= 2:
        matrix = view[numbers[:20]].corr(method="pearson")
        output.append(chart("수치형 상관관계 · Pearson Heatmap", correlation_heatmap(matrix, "Pearson"),
                            "셀 안 숫자는 상관계수 r(-1~1)입니다. +1은 강한 양의 선형 관계, -1은 강한 음의 선형 관계입니다."))
        output.append(chart("수치형 순위 상관 · Spearman Heatmap",
                            correlation_heatmap(view[numbers[:20]].corr(method="spearman"), "Spearman"),
                            "셀 안 숫자는 순위 상관계수 ρ(-1~1)입니다. 계수는 퍼센트나 인과관계를 의미하지 않습니다."))
        x = feature if feature in numbers else numbers[0]
        y = second_feature if second_feature in numbers and second_feature != x else next((c for c in numbers if c != x), None)
        if y:
            pair = view[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair):
                output.append(chart("피처 간 관계 · Scatter", px.scatter(pair, x=x, y=y, opacity=.55),
                                    "산점도에서 곡선 관계와 군집, 극단값을 확인합니다."))
                output.append(chart("피처 간 관계 + 주변 분포 · Marginal Scatter",
                                    px.scatter(pair, x=x, y=y, marginal_x="histogram", marginal_y="histogram", opacity=.4),
                                    "두 변수의 분포와 관계를 동시에 관찰합니다."))
        if heavy and len(numbers) >= 3:
            output.append(chart("피처 관계 행렬 · Scatter Matrix",
                                px.scatter_matrix(view[numbers[:4]].dropna().head(400), dimensions=numbers[:4]),
                                "최대 4개 변수·400행으로 제한해 브라우저 부하를 줄였습니다."))
    if target is None or feature is None or target == feature:
        return output
    pair = view[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if pair.empty:
        return output
    is_num = pd.api.types.is_numeric_dtype(pair[feature])
    if task == "분류":
        pair[target] = pair[target].astype(str)
        if is_num:
            output.append(chart("클래스별 피처 분포 · Box", px.box(pair, x=target, y=feature),
                                "클래스마다 중앙값과 사분위 범위가 다른지 확인합니다."))
            output.append(chart("클래스별 피처 분포 · Violin", px.violin(pair, x=target, y=feature, box=True),
                                "클래스별 분포 모양과 폭을 비교합니다."))
            output.append(chart("클래스별 피처 히스토그램 · Overlay", px.histogram(pair, x=feature, color=target,
                                barmode="overlay", histnorm="probability density", nbins=30, opacity=.6),
                                "클래스별 표본 수 차이를 감안하여 정규화된 분포를 비교합니다."))
            means = pair.groupby(target)[feature].agg(["mean", "median"]).reset_index()
            output.append(chart("클래스별 평균·중앙값 · Grouped Bar", px.bar(means.melt(id_vars=target, var_name="통계", value_name="값"),
                                x=target, y="값", color="통계", barmode="group"),
                                "평균과 중앙값의 차이가 크면 극단값 영향을 의심할 수 있습니다."))
        else:
            pair[feature] = top_category_series(pair[feature], 10)
            table = pd.crosstab(pair[feature], pair[target])
            long = table.reset_index().melt(id_vars=feature, var_name="클래스", value_name="건수")
            output.append(chart("범주별 클래스 건수 · Grouped Bar", px.bar(long, x=feature, y="건수", color="클래스", barmode="group"),
                                "범주 크기가 다르면 건수뿐 아니라 비율도 비교해야 합니다."))
            output.append(chart("범주별 클래스 구성비 · Normalized Stack", px.bar(long, x=feature, y="건수", color="클래스").update_layout(barnorm="percent"),
                                "범주마다 타깃 비중이 얼마나 달라지는지 확인합니다."))
            output.append(chart("범주 × 타깃 · Crosstab Heatmap", px.imshow(table, aspect="auto"),
                                "특정 범주·클래스 조합이 유독 많은지 확인합니다."))
    else:
        pair[target] = pd.to_numeric(pair[target], errors="coerce")
        pair = pair.dropna()
        if pair.empty:
            return output
        if is_num:
            output.append(chart("피처 vs 회귀 타깃 · Scatter", px.scatter(pair, x=feature, y=target, opacity=.5),
                                "선형뿐 아니라 곡선 패턴과 극단값도 확인합니다."))
            output.append(chart("피처 vs 타깃 · Density Contour", px.density_contour(pair, x=feature, y=target),
                                "관측값이 몰려 있는 구간을 등고선으로 확인합니다."))
        else:
            pair[feature] = top_category_series(pair[feature], 10)
            output.append(chart("범주별 회귀 타깃 · Box", px.box(pair, x=feature, y=target),
                                "범주마다 타깃 분포와 극단값이 다른지 봅니다."))
            output.append(chart("범주별 회귀 타깃 · Violin", px.violin(pair, x=feature, y=target, box=True),
                                "범주별 타깃의 밀집 구간을 비교합니다."))
    return output


def comparison_charts(train, test, column, comparison_table=None):
    """06 / Train-Test 수치 4종, 범주 3종, 전역 비교 2종."""
    output = []
    shared = [c for c in train if c in test]
    if comparison_table is not None and not comparison_table.empty:
        drift = comparison_table.dropna(subset=["PSI"]).sort_values("PSI", ascending=False).head(20)
        if not drift.empty:
            output.append(chart("변수별 PSI · Bar", px.bar(drift, x="컬럼", y="PSI"),
                                "PSI는 분포 변화를 보여주는 참고 지표입니다. 보편적 절대 기준은 아닙니다."))
        missing = comparison_table.melt(id_vars="컬럼", value_vars=["Train 결측(%)", "Test 결측(%)"],
                                        var_name="구분", value_name="결측률").head(60)
        output.append(chart("Train/Test 결측률 비교 · Grouped Bar", px.bar(missing, x="컬럼", y="결측률", color="구분", barmode="group"),
                            "Train과 Test의 결측률이 다르면 전처리 안정성을 점검하세요."))
    if column not in shared:
        return output
    a = sample_rows(train[[column]], 2500).copy()
    b = sample_rows(test[[column]], 2500).copy()
    if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(b[column]):
        joined = pd.concat([a.assign(데이터="Train"), b.assign(데이터="Test")], ignore_index=True)
        joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
        if joined.empty:
            return output
        output.append(chart("Train/Test 분포 · Overlaid Histogram", px.histogram(joined, x=column, color="데이터",
                            barmode="overlay", histnorm="probability density", nbins=35, opacity=.6),
                            "분포의 위치·폭·꼬리 모양을 비교합니다."))
        output.append(chart("Train/Test 극단값 · Box", px.box(joined, x="데이터", y=column),
                            "값 범위와 중앙값이 크게 달라지는지 살펴봅니다."))
        output.append(chart("Train/Test 누적 분포 · ECDF", px.ecdf(joined, x=column, color="데이터"),
                            "누적 분포 간 격차가 분포 변화의 단서가 됩니다."))
        lines = []
        for label, group in joined.groupby("데이터"):
            if group[column].nunique() > 3 and group[column].std() > 0:
                vals = group[column].to_numpy()
                xx = np.linspace(np.min(vals), np.max(vals), 160)
                try:
                    density = stats.gaussian_kde(vals)(xx)
                    lines.append(pd.DataFrame({column: xx, "확률밀도": density, "데이터": label}))
                except (ValueError, np.linalg.LinAlgError):
                    continue
        if lines:
            output.append(chart("Train/Test 확률밀도 · KDE Overlay", px.line(pd.concat(lines), x=column, y="확률밀도", color="데이터"),
                                "두 밀도 곡선의 차이는 데이터 시프트 여부를 살펴보는 데 유용합니다."))
    else:
        a = top_category_series(a[column], 10)
        b = top_category_series(b[column], 10)
        labels = sorted(set(a.unique()) | set(b.unique()))
        frequencies = pd.DataFrame({"범주": labels, "Train": a.value_counts(normalize=True).reindex(labels, fill_value=0).values,
                                    "Test": b.value_counts(normalize=True).reindex(labels, fill_value=0).values})
        long = frequencies.melt(id_vars="범주", var_name="데이터", value_name="비율")
        output.append(chart("범주 비중 비교 · Grouped Bar", px.bar(long, x="범주", y="비율", color="데이터", barmode="group"),
                            "범주별 구성 비율 차이를 봅니다. 신규 범주가 있는지 별도 표에서 확인하세요."))
        output.append(chart("범주별 비율 변화 · Difference Bar", px.bar(frequencies.assign(차이=frequencies["Test"] - frequencies["Train"]),
                            x="범주", y="차이"), "양수는 Test 비율 증가, 음수는 Test 비율 감소입니다."))
        output.append(chart("범주별 Train/Test 비중 · Heatmap", px.imshow(frequencies.set_index("범주")[["Train", "Test"]],
                            aspect="auto"), "동일 범주의 상대 비율을 색상으로 비교합니다."))
    return output


def anomaly_charts(df, columns, flags, heavy=False):
    """07 / 이상 점수와 정상·이상 집단 비교, PCA 투영 포함."""
    output = []
    if flags is None or not columns:
        return output
    select = [c for c in columns if c in df]
    frame = df[select].copy()
    frame["그룹"] = flags["이상치 여부"].map({True: "이상치 후보", False: "일반"})
    frame["이상 점수"] = flags["이상 점수"]
    view = sample_rows(frame)
    count = frame["그룹"].value_counts()
    output.append(chart("이상치 비율 · Donut", px.pie(names=count.index, values=count.values, hole=.5),
                        "이상치 후보는 자동으로 제거하지 않습니다."))
    if view["이상 점수"].notna().any():
        output.append(chart("이상 점수 · Histogram", px.histogram(view, x="이상 점수", color="그룹", nbins=35),
                            "점수가 클수록 Isolation Forest 기준에서 더 특이한 관측값입니다."))
    if len(select) >= 2:
        output.append(chart("이상치 그룹 · Scatter", px.scatter(view, x=select[0], y=select[1], color="그룹", opacity=.65),
                            "두 피처 공간에서 정상·이상 데이터가 어떻게 다른지 확인합니다."))
    if select:
        output.append(chart("그룹별 선택 변수 · Box", px.box(view, x="그룹", y=select[0]),
                            "선택한 변수에서 두 집단의 분포 차이를 비교합니다."))
        output.append(chart("그룹별 선택 변수 · Violin", px.violin(view, x="그룹", y=select[0], box=True),
                            "이상치 후보가 어떤 값 범위에 집중되는지 봅니다."))
    if heavy and len(select) >= 2 and len(df) >= 5:
        matrix = df[select].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if matrix.notna().any().all():
            matrix = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(matrix), columns=select)
            if (matrix.std() > 0).sum() >= 2:
                transformed = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(matrix))
                points = pd.DataFrame({"PC1": transformed[:, 0], "PC2": transformed[:, 1], "그룹": frame["그룹"].to_numpy()})
                output.append(chart("이상치 후보 · PCA 2D Projection", px.scatter(sample_rows(points), x="PC1", y="PC2", color="그룹", opacity=.7),
                                    "PCA는 여러 피처를 2차원에 투영한 결과로, 실제 거리 관계 일부가 왜곡될 수 있습니다."))
    return output


def feature_charts(df, column, ratio_column=None):
    """08 / 파생변수 만들기 전에 원본·변환 분포를 비교하는 3종."""
    output = []
    if column not in df:
        return output
    numeric = clean_numeric(sample_rows(df)[column])
    if numeric.empty:
        return output
    if (numeric >= 0).all():
        before_after = pd.DataFrame({"원본": numeric, "log1p": np.log1p(numeric)})
        output.append(chart("로그 변환 전후 · Facet Histogram", px.histogram(before_after.melt(var_name="버전", value_name="값"),
                          x="값", facet_col="버전", nbins=35),
                            "log1p는 0 이상 수치에 적용합니다. 로그 변환이 항상 성능을 높이지는 않습니다."))
    if numeric.std() > 0:
        standardized = (numeric - numeric.mean()) / numeric.std()
        merged = pd.DataFrame({"원본": numeric, "표준화": standardized})
        output.append(chart("표준화 전후 · Box Comparison", px.box(merged.melt(var_name="버전", value_name="값"),
                          x="버전", y="값"),
                            "모델 전처리 스케일링은 학습 폴드에서만 fit해야 데이터 누수를 막을 수 있습니다."))
    if ratio_column and ratio_column in df and ratio_column != column:
        right = pd.to_numeric(sample_rows(df)[ratio_column], errors="coerce").replace(0, np.nan)
        ratio = (pd.to_numeric(sample_rows(df)[column], errors="coerce") / right).replace([np.inf, -np.inf], np.nan).dropna()
        if not ratio.empty:
            output.append(chart("비율 파생변수 · Histogram", px.histogram(x=ratio.clip(ratio.quantile(.01), ratio.quantile(.99)), nbins=35),
                                "0 분모를 제외했고 그래프 표시 구간을 1~99% 분위수로 제한했습니다. 원본 데이터는 변경하지 않았습니다."))
    return output


def model_charts(result):
    """09~10 / 모델별 CV·중요도·오답 분석 그래프."""
    output = []
    if result is None:
        return output
    cv = result["cv"]
    output.append(chart("교차검증 지표 · Bar", px.bar(cv, x="지표", y="CV 평균", error_y="CV 표준편차"),
                        "각 지표는 값이 클수록 좋은지 방향이 다릅니다. RMSE·MAE는 낮을수록 좋습니다."))
    importance = result["importance"].head(20)
    output.append(chart("Permutation Importance · Horizontal Bar", px.bar(importance.sort_values("중요도"),
                        x="중요도", y="피처", error_x="표준편차", orientation="h"),
                        "검증 성능 감소량으로 계산했습니다. 중요도가 낮다고 항상 불필요한 피처는 아닙니다."))
    pred = result["predictions"].copy()
    if result["task"] == "분류":
        confusion = pd.crosstab(pred["실제값"].astype(str), pred["예측값"].astype(str))
        output.append(chart("Confusion Matrix · Heatmap", px.imshow(confusion, text_auto=True, aspect="auto"),
                            "대각선 밖은 오분류입니다."))
        err = pred.groupby("실제값", dropna=False)["오답 여부"].agg(["mean", "size"]).reset_index()
        output.append(chart("클래스별 오분류 비율 · Bar", px.bar(err, x="실제값", y="mean",
                            hover_data=["size"], labels={"mean": "오답 비율"}),
                            "데이터가 많은 클래스만 잘 맞추는지 확인합니다."))
        if "양성 확률" in pred:
            output.append(chart("클래스별 예측 확률 · Histogram", px.histogram(pred, x="양성 확률", color="실제값",
                                histnorm="probability density", barmode="overlay", nbins=30, opacity=.6),
                                "오답이 집중되는 경계 확률 구간을 찾을 수 있습니다."))
            from sklearn.metrics import precision_recall_curve, roc_curve
            valid = pred["실제값"].astype(str) == str(result["positive_label"])
            if valid.nunique() == 2:
                fpr, tpr, _ = roc_curve(valid, pred["양성 확률"])
                output.append(chart("ROC Curve · Line", px.line(x=fpr, y=tpr, labels={"x": "FPR", "y": "TPR"}),
                                    "ROC-AUC는 양성·음성 구분 능력을 나타냅니다."))
                precision, recall, thresholds = precision_recall_curve(valid, pred["양성 확률"])
                output.append(chart("Precision-Recall Curve · Line", px.line(x=recall, y=precision,
                                    labels={"x": "Recall", "y": "Precision"}),
                                    "불균형 분류에서는 PR Curve도 함께 보는 것이 좋습니다."))
                if len(thresholds):
                    t = pd.DataFrame({"임계값": thresholds, "Precision": precision[:-1], "Recall": recall[:-1]})
                    output.append(chart("임계값별 정밀도·재현율 · Line", px.line(t.melt(id_vars="임계값", var_name="지표", value_name="값"),
                                        x="임계값", y="값", color="지표"),
                                        "임계값 조정은 별도 검증 설계 안에서 해야 하며, 재사용한 홀드아웃에 과적합하지 않도록 주의하세요."))
    else:
        output.append(chart("실제값 vs 예측값 · Scatter", px.scatter(pred, x="실제값", y="예측값", opacity=.6),
                            "대각선에 가까울수록 예측 오차가 작습니다."))
        output.append(chart("예측값 vs 잔차 · Scatter", px.scatter(pred, x="예측값", y="잔차 (실제-예측)", opacity=.6),
                            "잔차가 특정 구간에서 편향되지 않았는지 확인합니다."))
        output.append(chart("잔차 분포 · Histogram", px.histogram(pred, x="잔차 (실제-예측)", nbins=30),
                            "잔차 분포가 한쪽으로 치우치면 체계적 오차 가능성이 있습니다."))
        output.append(chart("절대오차 분포 · ECDF", px.ecdf(pred, x="절대오차"),
                            "전체 데이터의 얼마가 특정 오차 이내인지 파악합니다."))
        sliced = pred.copy()
        if sliced["실제값"].nunique() >= 4:
            sliced["실제값 구간"] = pd.qcut(sliced["실제값"], q=4, duplicates="drop").astype(str)
            output.append(chart("실제값 구간별 잔차 · Box", px.box(sliced, x="실제값 구간", y="잔차 (실제-예측)"),
                                "타깃의 고값·저값 구간에서 오차가 커지는지 확인합니다."))
    return output


# 지원하는 그래프 종류를 문서화한 카탈로그입니다. 화면에서는 데이터에 맞는 것만 렌더링합니다.
CHART_CATALOG = {
    "데이터 개요": ["자료형 Donut", "고유값 Bar", "메모리 Horizontal Bar"],
    "데이터 품질": ["결측률 Bar", "Missing Heatmap", "행별 결측 Histogram", "결측 패턴 Bar", "완성도 ECDF", "왜도 Bar", "IQR 후보 Bar"],
    "단변량 EDA": ["Histogram", "Box", "Violin", "ECDF", "Quantile Line", "KDE", "Q-Q Scatter", "Category Bar", "Category Donut", "Treemap", "Pareto Line"],
    "타깃 분석": ["Class Count Bar", "Class Donut", "Imbalance Bar", "Regression Histogram", "Regression Box", "Log Facet", "Target Q-Q"],
    "피처 관계": ["Pearson Heatmap", "Spearman Heatmap", "Scatter", "Marginal Scatter", "Scatter Matrix", "Class Box", "Class Violin", "Class Overlay Histogram", "Grouped Mean Bar", "Category Grouped Bar", "Normalized Stack", "Crosstab Heatmap", "Regression Scatter", "Density Contour", "Category Regression Box", "Category Regression Violin"],
    "Train/Test": ["PSI Bar", "Missing Grouped Bar", "Overlay Histogram", "Train/Test Box", "Train/Test ECDF", "KDE Overlay", "Category Grouped Bar", "Category Difference Bar", "Category Heatmap"],
    "이상치": ["Anomaly Donut", "Score Histogram", "Anomaly Scatter", "Anomaly Box", "Anomaly Violin", "PCA Projection"],
    "피처 엔지니어링": ["Log Facet Histogram", "Scaled Box", "Ratio Histogram"],
    "모델·오답": ["CV Bar", "Permutation Importance Bar", "Confusion Heatmap", "Class Error Bar", "Probability Histogram", "ROC Curve", "PR Curve", "Threshold Curves", "Prediction Scatter", "Residual Scatter", "Residual Histogram", "Absolute Error ECDF", "Residual by Quartile Box", "SHAP Importance Bar"],
}
