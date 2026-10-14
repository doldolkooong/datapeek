"""교차검증, 홀드아웃 평가, 피처 중요도 및 오답 분석."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     StratifiedKFold, TimeSeriesSplit,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import make_features


def create_pipeline(x: pd.DataFrame, task: str, model_name: str) -> Pipeline:
    numeric = x.select_dtypes(include="number").columns.tolist()
    categorical = [col for col in x.columns if col not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=15)),
        ]), categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    if task == "분류":
        if model_name == "랜덤 포레스트":
            estimator = RandomForestClassifier(n_estimators=80, max_depth=9,
                                               min_samples_leaf=2, n_jobs=-1, random_state=42)
        else:
            estimator = LogisticRegression(max_iter=600, random_state=42)
    else:
        if model_name == "랜덤 포레스트":
            estimator = RandomForestRegressor(n_estimators=80, max_depth=9,
                                              min_samples_leaf=2, n_jobs=-1, random_state=42)
        else:
            estimator = Ridge(alpha=1.0)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def evaluate_baseline(df: pd.DataFrame, target: str, task: str, features: list[str],
                      model_name: str, strategy: str = "랜덤",
                      group_column: str | None = None, time_column: str | None = None,
                      date_columns: list[str] | None = None,
                      text_columns: list[str] | None = None,
                      log_columns: list[str] | None = None,
                      ratio: tuple[str, str] | None = None,
                      max_rows: int = 3000) -> dict:
    if target in features or not features:
        raise ValueError("타깃을 제외한 피처를 한 개 이상 선택해야 합니다.")
    if not set(features).issubset(df.columns):
        raise ValueError("선택한 피처 중 데이터셋에 없는 컬럼이 있습니다.")
    if strategy == "그룹" and not group_column:
        raise ValueError("그룹 분할용 컬럼을 선택해 주세요.")
    if strategy == "시간" and not time_column:
        raise ValueError("시간 분할용 컬럼을 선택해 주세요.")

    valid = df[df[target].notna()].copy()
    if task == "회귀":
        valid[target] = pd.to_numeric(valid[target], errors="coerce")
        valid = valid[valid[target].notna()].copy()
    if len(valid) < 30:
        raise ValueError("지도학습은 타깃이 있는 데이터 30행 이상에서 실행해 주세요.")
    if len(valid) > max_rows:
        if strategy == "시간":
            valid = valid.sort_values(time_column).tail(max_rows).copy()
        else:
            valid = valid.sample(n=max_rows, random_state=42)

    y = valid[target].astype(str) if task == "분류" else valid[target].astype(float)
    if task == "분류" and y.nunique() < 2:
        raise ValueError("분류에는 서로 다른 타깃 클래스가 최소 2개 필요합니다.")
    if task == "분류" and y.value_counts().min() < 3:
        raise ValueError("표본이 3개 미만인 클래스가 있습니다. 클래스 수와 표본 수를 확인해 주세요.")
    x = make_features(valid[features], date_columns, text_columns, log_columns, ratio)
    if not len(x.columns):
        raise ValueError("학습할 피처가 없습니다.")
    if strategy == "시간":
        times = pd.to_datetime(valid[time_column], errors="coerce")
        if times.isna().any():
            raise ValueError("시간 컬럼에 날짜로 변환할 수 없는 값 또는 결측치가 있습니다.")
        order = np.argsort(times.to_numpy(), kind="stable")
        train_index = order[:int(len(order) * 0.8)]
        val_index = order[int(len(order) * 0.8):]
        splitter = TimeSeriesSplit(n_splits=3)
        x_train, y_train = x.iloc[train_index], y.iloc[train_index]
        x_valid, y_valid = x.iloc[val_index], y.iloc[val_index]
        groups_train = None
    elif strategy == "그룹":
        groups = valid[group_column].fillna("__MISSING_GROUP__").astype(str)
        if groups.nunique() < 4:
            raise ValueError("그룹 분할에는 최소 4개 이상의 서로 다른 그룹이 필요합니다.")
        split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_index, val_index = next(split.split(x, y, groups))
        x_train, y_train = x.iloc[train_index], y.iloc[train_index]
        x_valid, y_valid = x.iloc[val_index], y.iloc[val_index]
        groups_train = groups.iloc[train_index]
        splitter = GroupKFold(n_splits=min(4, groups_train.nunique()))
    else:
        stratify = y if task == "분류" else None
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=stratify)
        groups_train = None
        if task == "분류":
            min_count = y_train.value_counts().min()
            splitter = StratifiedKFold(n_splits=min(4, int(min_count)), shuffle=True, random_state=42)
        else:
            splitter = KFold(n_splits=4, shuffle=True, random_state=42)

    if task == "분류" and y_train.nunique() < 2:
        raise ValueError("학습용 분할에 한 종류의 클래스만 남았습니다. 분할 방식을 변경해 주세요.")
    pipe = create_pipeline(x_train, task, model_name)
    if task == "분류":
        scoring = {"Accuracy": "accuracy", "F1-macro": "f1_macro"}
    else:
        scoring = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R²": "r2"}
    cv_scores = cross_validate(pipe, x_train, y_train, cv=splitter, groups=groups_train,
                               scoring=scoring, error_score="raise", n_jobs=1)
    cv_table = []
    for key in scoring:
        values = cv_scores[f"test_{key}"]
        if key in {"RMSE", "MAE"}:
            values = -values
        cv_table.append({"지표": key, "CV 평균": float(np.mean(values)), "CV 표준편차": float(np.std(values))})
    pipe.fit(x_train, y_train)
    predicted = pipe.predict(x_valid)
    metrics = {}
    if task == "분류":
        metrics = {"Accuracy": float(accuracy_score(y_valid, predicted)),
                   "F1-macro": float(f1_score(y_valid, predicted, average="macro", zero_division=0))}
        if y.nunique() == 2 and y_valid.nunique() == 2 and hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(x_valid)[:, 1]
            positive_label = pipe.classes_[1]
            metrics["ROC-AUC"] = float(roc_auc_score(y_valid == positive_label, probs))
    else:
        metrics = {"RMSE": float(np.sqrt(mean_squared_error(y_valid, predicted))),
                   "MAE": float(mean_absolute_error(y_valid, predicted)),
                   "R²": float(r2_score(y_valid, predicted))}

    predictions = pd.DataFrame({"원본 행 번호": x_valid.index, "실제값": y_valid.to_numpy(),
                                "예측값": predicted})
    if task == "분류" and y.nunique() == 2 and hasattr(pipe, "predict_proba"):
        predictions["양성 확률"] = pipe.predict_proba(x_valid)[:, 1]
    if task == "분류":
        predictions["오답 여부"] = predictions["실제값"] != predictions["예측값"]
    else:
        predictions["잔차 (실제-예측)"] = y_valid.to_numpy() - predicted
        predictions["절대오차"] = np.abs(predictions["잔차 (실제-예측)"])
    # 홀드아웃 데이터에 한해서 중요도를 구합니다. 학습 데이터에서 측정하지 않습니다.
    importance = permutation_importance(pipe, x_valid.head(500), y_valid.head(500),
                                        n_repeats=3, random_state=42,
                                        scoring="f1_macro" if task == "분류" else "neg_mean_absolute_error",
                                        n_jobs=1)
    importance_frame = pd.DataFrame({"피처": x.columns,
                                      "중요도": importance.importances_mean,
                                      "표준편차": importance.importances_std}).sort_values("중요도", ascending=False)
    return {"cv": pd.DataFrame(cv_table), "metrics": metrics, "predictions": predictions,
            "importance": importance_frame, "pipeline": pipe,
            "x_valid": x_valid, "y_valid": y_valid,
            "n_train": len(x_train), "n_valid": len(x_valid), "n_total": len(valid),
            "task": task, "model_name": model_name, "strategy": strategy,
            "positive_label": str(pipe.classes_[1]) if task == "분류" and len(pipe.classes_) == 2 else None}


def explain_with_shap(result: dict, max_samples: int = 30) -> pd.DataFrame:
    """선택 설치된 shap으로 트리 모델의 변환된 피처별 평균 |SHAP| 계산."""
    if result["model_name"] != "랜덤 포레스트":
        raise ValueError("SHAP은 이 앱에서 랜덤 포레스트 모델에만 지원됩니다.")
    try:
        import shap
    except ImportError as error:
        raise ValueError("SHAP 사용을 위해 `uv pip install shap`을 실행해 주세요.") from error
    pipe = result["pipeline"]
    preprocessing = pipe.named_steps["preprocess"]
    encoded = preprocessing.transform(result["x_valid"].head(max_samples))
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    if encoded.shape[1] > 150:
        raise ValueError("인코딩된 컬럼이 150개를 초과합니다. 분석 피처를 줄여 주세요.")
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    shap_values = np.asarray(explainer.shap_values(encoded))
    if shap_values.ndim == 3:
        values = np.abs(shap_values).mean(axis=(0, 2))
    elif shap_values.ndim == 2:
        values = np.abs(shap_values).mean(axis=0)
    else:
        raise ValueError("지원하지 않는 SHAP 결과 형태입니다.")
    try:
        names = preprocessing.get_feature_names_out().tolist()
    except Exception:
        names = [f"변환 피처 {i}" for i in range(encoded.shape[1])]
    return pd.DataFrame({"피처": names, "평균 |SHAP|": values}).sort_values("평균 |SHAP|", ascending=False).head(20)
