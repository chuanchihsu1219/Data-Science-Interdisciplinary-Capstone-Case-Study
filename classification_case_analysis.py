from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
GIVEN_DIR = ROOT / "given"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REPORT_PATH = OUTPUT_DIR / "report.md"

CAMERA_COST = 2000
VANDALISM_LOSS = 10000
RANDOM_STATE = 42
TEST_SIZE = 0.2
BUSINESS_THRESHOLD = CAMERA_COST / VANDALISM_LOSS
SENSITIVITY_LEVELS = np.round(np.linspace(0, 1, 11), 1)
BUDGET_SCENARIOS = [50, 100, 200]

PRIMARY_FEATURES = [
    "building_type",
    "indoor",
    "distance_to_store",
    "weekend",
    "club_activity_day",
]

COLUMN_ALIASES = {
    "date": ["date", "day", "transaction_date"],
    "location_id": ["location_id", "location", "machine_id", "machine", "id"],
    "building_type": ["building_type", "building", "buildingcategory", "type"],
    "distance_to_store": [
        "distance_to_store",
        "distancetonearestconveniencestoremeters",
        "distance_to_nearest_convenience_store_meters",
        "distance",
    ],
    "temperature": ["temperature", "temp"],
    "humidity": ["humidity"],
    "weekend": ["weekend", "is_weekend"],
    "club_activity_day": ["club_activity_day", "clubday", "club_activity"],
    "indoor": ["indoor", "inside"],
    "sales": ["sales", "revenue"],
    "broken": ["broken", "vandalized", "damage", "damaged"],
}

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.float_format", lambda value: f"{value:,.4f}")


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    auc_score: float
    brier_score: float
    probabilities: np.ndarray
    metrics_by_threshold: dict[float, dict[str, Any]]
    feature_table: pd.DataFrame
    confusion_matrices: dict[float, np.ndarray]


@dataclass
class AnalysisArtifacts:
    history_location: pd.DataFrame
    history_daily: pd.DataFrame
    candidate_location: pd.DataFrame
    mapping_table: pd.DataFrame
    data_quality: dict[str, Any]
    descriptive_tables: dict[str, pd.DataFrame]
    model_results: dict[str, ModelResult]
    model_comparison: pd.DataFrame
    selected_model_name: str
    selected_candidate_scores: pd.DataFrame
    strategy_comparison: pd.DataFrame
    sensitivity_table: pd.DataFrame
    budget_table: pd.DataFrame
    report_path: Path


def ensure_output_dirs() -> None:
    for directory in (OUTPUT_DIR, FIGURE_DIR, TABLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def normalize_column_name(name: str) -> str:
    return "".join(character for character in str(name).lower() if character.isalnum())


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    normalized_lookup = {normalize_column_name(column): column for column in df.columns}
    mapping: dict[str, str] = {}
    for canonical_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            normalized_alias = normalize_column_name(alias)
            if normalized_alias in normalized_lookup:
                mapping[canonical_name] = normalized_lookup[normalized_alias]
                break
    return mapping


def mode_or_first(series: pd.Series) -> Any:
    non_missing = series.dropna()
    if non_missing.empty:
        return np.nan
    modes = non_missing.mode(dropna=True)
    if modes.empty:
        return non_missing.iloc[0]
    return modes.iloc[0]


def coerce_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "true": 1,
                "false": 0,
                "yes": 1,
                "no": 0,
                "y": 1,
                "n": 0,
            }
        )
    )
    return pd.to_numeric(normalized, errors="coerce")


def rename_to_canonical(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = resolve_columns(df)
    renamed = df.rename(columns={source: canonical for canonical, source in mapping.items()})
    mapping_table = pd.DataFrame(
        {
            "source_dataset": source_name,
            "canonical_name": list(mapping.keys()),
            "original_column": list(mapping.values()),
        }
    )
    return renamed, mapping_table


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    broken_raw = pd.read_csv(GIVEN_DIR / "broken.csv")
    sales_raw = pd.read_csv(GIVEN_DIR / "sales-data.csv")
    candidate_raw = pd.read_csv(GIVEN_DIR / "candidate-location.csv")

    broken_df, broken_mapping = rename_to_canonical(broken_raw, "broken.csv")
    sales_df, sales_mapping = rename_to_canonical(sales_raw, "sales-data.csv")
    candidate_df, candidate_mapping = rename_to_canonical(candidate_raw, "candidate-location.csv")

    mapping_table = pd.concat([broken_mapping, sales_mapping, candidate_mapping], ignore_index=True)

    return broken_df, sales_df, candidate_df, mapping_table


def prepare_raw_frames(
    broken_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for frame in (sales_df, candidate_df):
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if "building_type" in frame.columns:
            frame["building_type"] = frame["building_type"].astype(str).str.strip().str.lower()
        for column in ["indoor", "weekend", "club_activity_day"]:
            if column in frame.columns:
                frame[column] = coerce_binary(frame[column])
        for column in ["distance_to_store", "temperature", "humidity", "sales"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    broken_df["broken"] = coerce_binary(broken_df["broken"]).fillna(0).astype(int)
    broken_df["location_id"] = broken_df["location_id"].astype(str)
    sales_df["location_id"] = sales_df["location_id"].astype(str)
    candidate_df["location_id"] = candidate_df["location_id"].astype(str)

    return broken_df, sales_df, candidate_df


def aggregate_location_features(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    quality = {
        "dataset": dataset_name,
        "missing_values": df.isna().sum().to_dict(),
        "location_count": int(df["location_id"].nunique()),
    }

    if "date" in df.columns:
        quality["date_min"] = df["date"].min()
        quality["date_max"] = df["date"].max()

    per_location_variation = (
        df.groupby("location_id")
        .agg(
            building_type_unique=("building_type", "nunique"),
            indoor_unique=("indoor", pd.Series.nunique),
            distance_unique=("distance_to_store", pd.Series.nunique),
        )
        .reset_index()
    )
    quality["building_type_inconsistent_locations"] = int((per_location_variation["building_type_unique"] > 1).sum())
    quality["indoor_inconsistent_locations"] = int((per_location_variation["indoor_unique"] > 1).sum())
    quality["distance_variable_locations"] = int((per_location_variation["distance_unique"] > 1).sum())

    aggregated = (
        df.groupby("location_id")
        .agg(
            building_type=("building_type", mode_or_first),
            indoor=("indoor", mode_or_first),
            distance_to_store=("distance_to_store", "median"),
            weekend=("weekend", "mean"),
            club_activity_day=("club_activity_day", "mean"),
            days_observed=("location_id", "size"),
        )
        .reset_index()
    )

    aggregated["building_type"] = aggregated["building_type"].fillna("unknown")
    aggregated["indoor"] = aggregated["indoor"].fillna(aggregated["indoor"].mode().iloc[0]).astype(int)
    aggregated["distance_to_store"] = aggregated["distance_to_store"].fillna(aggregated["distance_to_store"].median())
    aggregated["weekend"] = aggregated["weekend"].fillna(aggregated["weekend"].median())
    aggregated["club_activity_day"] = aggregated["club_activity_day"].fillna(aggregated["club_activity_day"].median())

    return aggregated, quality


def build_history_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    broken_df, sales_df, candidate_df, mapping_table = load_raw_data()
    broken_df, sales_df, candidate_df = prepare_raw_frames(broken_df, sales_df, candidate_df)

    history_location, history_quality = aggregate_location_features(sales_df, "historical")
    candidate_location, candidate_quality = aggregate_location_features(candidate_df, "candidate")

    history_location = history_location.merge(broken_df, on="location_id", how="inner", validate="one_to_one")
    history_daily = sales_df.merge(broken_df, on="location_id", how="left", validate="many_to_one")

    overlap_count = candidate_location["location_id"].isin(history_location["location_id"]).sum()
    data_quality = {
        "historical": history_quality,
        "candidate": candidate_quality,
        "candidate_overlap_with_history": int(overlap_count),
    }

    return history_location, history_daily, candidate_location, mapping_table, data_quality


def save_table(df: pd.DataFrame, file_name: str, index: bool = False) -> Path:
    path = TABLE_DIR / file_name
    df.to_csv(path, index=index)
    return path


def percentage(series: pd.Series) -> pd.Series:
    return series * 100


def build_descriptive_tables(history_location: pd.DataFrame) -> dict[str, pd.DataFrame]:
    total_rows = len(history_location)
    broken_counts = history_location["broken"].value_counts().sort_index()
    overview = pd.DataFrame(
        [
            {"metric": "總樣本數", "value": total_rows},
            {"metric": "Broken = 0 筆數", "value": int(broken_counts.get(0, 0))},
            {"metric": "Broken = 0 比例", "value": broken_counts.get(0, 0) / total_rows},
            {"metric": "Broken = 1 筆數", "value": int(broken_counts.get(1, 0))},
            {"metric": "Broken = 1 比例", "value": broken_counts.get(1, 0) / total_rows},
        ]
    )

    binary_summary = (
        history_location.groupby("broken")[["indoor", "weekend", "club_activity_day"]]
        .mean()
        .rename(
            columns={
                "indoor": "Indoor mean",
                "weekend": "Weekend mean",
                "club_activity_day": "Club Activity Day mean",
            }
        )
        .reset_index()
    )

    distance_summary = history_location.groupby("broken")["distance_to_store"].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    quantiles = (
        history_location.groupby("broken")["distance_to_store"]
        .quantile([0.25, 0.75])
        .unstack()
        .rename(columns={0.25: "q1", 0.75: "q3"})
        .reset_index()
    )
    distance_summary = distance_summary.merge(quantiles, on="broken", how="left")

    building_counts = pd.crosstab(history_location["building_type"], history_location["broken"])
    building_share = pd.crosstab(
        history_location["building_type"],
        history_location["broken"],
        normalize="columns",
    ).add_prefix("share_broken_")
    building_rate = history_location.groupby("building_type")["broken"].agg(machine_count="size", broken_rate="mean").reset_index()
    building_distribution = (
        building_counts.rename(columns={0: "broken_0_count", 1: "broken_1_count"})
        .reset_index()
        .merge(building_share.reset_index(), on="building_type", how="left")
        .merge(building_rate, on="building_type", how="left")
    )

    return {
        "overview": overview,
        "binary_summary": binary_summary,
        "distance_summary": distance_summary,
        "building_distribution": building_distribution,
    }


def labeled_bar(ax: plt.Axes, x_values: list[Any], y_values: list[float], as_percent: bool = True) -> None:
    ax.bar(x_values, y_values, color="#2f6690")
    for position, value in enumerate(y_values):
        label = f"{value:.1%}" if as_percent else f"{value:,.0f}"
        ax.text(position, value + (0.01 if as_percent else max(y_values) * 0.01), label, ha="center", va="bottom", fontsize=9)


def save_current_figure(file_name: str) -> Path:
    path = FIGURE_DIR / file_name
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


def plot_descriptive_figures(history_location: pd.DataFrame, history_daily: pd.DataFrame) -> dict[str, Path]:
    figure_paths: dict[str, Path] = {}

    broken_counts = history_location["broken"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    labeled_bar(plt.gca(), ["Broken = 0", "Broken = 1"], (broken_counts / broken_counts.sum()).tolist())
    plt.title("Historical Broken Distribution")
    plt.ylabel("Share of machines")
    figure_paths["broken_distribution"] = save_current_figure("broken_distribution.png")

    building_rate = history_location.groupby("building_type")["broken"].mean().sort_values(ascending=False)
    plt.figure(figsize=(7, 4))
    labeled_bar(plt.gca(), building_rate.index.tolist(), building_rate.tolist())
    plt.title("Broken Rate by Building Type")
    plt.ylabel("Broken rate")
    figure_paths["building_type_broken_rate"] = save_current_figure("building_type_broken_rate.png")

    indoor_rate = history_location.groupby("indoor")["broken"].mean().sort_index()
    plt.figure(figsize=(6, 4))
    labeled_bar(plt.gca(), ["Outdoor", "Indoor"], indoor_rate.tolist())
    plt.title("Broken Rate by Indoor Status")
    plt.ylabel("Broken rate")
    figure_paths["indoor_broken_rate"] = save_current_figure("indoor_broken_rate.png")

    weekend_rate = history_daily.groupby("weekend")["broken"].mean().sort_index()
    plt.figure(figsize=(6, 4))
    labeled_bar(plt.gca(), ["Weekday", "Weekend"], weekend_rate.tolist())
    plt.title("Broken Rate by Weekend Exposure")
    plt.ylabel("Broken rate")
    figure_paths["weekend_broken_rate"] = save_current_figure("weekend_broken_rate.png")

    club_rate = history_daily.groupby("club_activity_day")["broken"].mean().sort_index()
    plt.figure(figsize=(6, 4))
    labeled_bar(plt.gca(), ["No club activity", "Club activity"], club_rate.tolist())
    plt.title("Broken Rate by Club Activity Day")
    plt.ylabel("Broken rate")
    figure_paths["club_activity_broken_rate"] = save_current_figure("club_activity_broken_rate.png")

    plt.figure(figsize=(7, 4))
    data = [
        history_location.loc[history_location["broken"] == 0, "distance_to_store"],
        history_location.loc[history_location["broken"] == 1, "distance_to_store"],
    ]
    plt.boxplot(data, labels=["Broken = 0", "Broken = 1"], patch_artist=True)
    plt.title("Distance to Store by Broken Status")
    plt.ylabel("Distance to nearest store (meters)")
    figure_paths["distance_boxplot"] = save_current_figure("distance_by_broken_boxplot.png")

    building_indoor = pd.pivot_table(
        history_location,
        values="broken",
        index="building_type",
        columns="indoor",
        aggfunc="mean",
    ).sort_index()
    plt.figure(figsize=(6, 4))
    plt.imshow(building_indoor, cmap="Blues", aspect="auto")
    plt.colorbar(label="Broken rate")
    plt.xticks(range(len(building_indoor.columns)), ["Outdoor" if value == 0 else "Indoor" for value in building_indoor.columns])
    plt.yticks(range(len(building_indoor.index)), building_indoor.index)
    plt.title("Broken Rate: Building Type x Indoor")
    for row_index, row_name in enumerate(building_indoor.index):
        for col_index, col_name in enumerate(building_indoor.columns):
            value = building_indoor.loc[row_name, col_name]
            if pd.notna(value):
                plt.text(col_index, row_index, f"{value:.1%}", ha="center", va="center", color="black", fontsize=9)
    figure_paths["building_indoor_heatmap"] = save_current_figure("building_type_indoor_broken_rate_heatmap.png")

    indoor_club = pd.pivot_table(
        history_daily,
        values="broken",
        index="indoor",
        columns="club_activity_day",
        aggfunc="mean",
    ).sort_index()
    plt.figure(figsize=(6, 4))
    plt.imshow(indoor_club, cmap="Oranges", aspect="auto")
    plt.colorbar(label="Broken rate")
    plt.xticks(range(len(indoor_club.columns)), ["No club activity", "Club activity"])
    plt.yticks(range(len(indoor_club.index)), ["Outdoor", "Indoor"])
    plt.title("Broken Rate: Indoor x Club Activity")
    for row_index in range(indoor_club.shape[0]):
        for col_index in range(indoor_club.shape[1]):
            value = indoor_club.iloc[row_index, col_index]
            if pd.notna(value):
                plt.text(col_index, row_index, f"{value:.1%}", ha="center", va="center", color="black", fontsize=9)
    figure_paths["indoor_club_heatmap"] = save_current_figure("indoor_club_broken_rate_heatmap.png")

    return figure_paths


def build_preprocessors(building_categories: list[str]) -> tuple[ColumnTransformer, ColumnTransformer]:
    logistic_preprocessor = ColumnTransformer(
        transformers=[
            (
                "building_type",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                categories=[building_categories],
                                drop="first",
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                ["building_type"],
            ),
            (
                "distance_to_store",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                ["distance_to_store"],
            ),
            (
                "binary_features",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))]),
                ["indoor", "weekend", "club_activity_day"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    forest_preprocessor = ColumnTransformer(
        transformers=[
            (
                "building_type",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                categories=[building_categories],
                                drop="first",
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                ["building_type"],
            ),
            (
                "numeric_features",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                ["distance_to_store", "indoor", "weekend", "club_activity_day"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return logistic_preprocessor, forest_preprocessor


def threshold_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions)
    tn, fp, fn, tp = matrix.ravel()
    realized_cost_value = float(predictions.sum() * CAMERA_COST + ((1 - predictions) * y_true.values * VANDALISM_LOSS).sum())
    expected_cost_value = float((predictions * CAMERA_COST + (1 - predictions) * probabilities * VANDALISM_LOSS).sum())
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1_score": f1_score(y_true, predictions, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "installed_count": int(predictions.sum()),
        "realized_cost": realized_cost_value,
        "expected_cost": expected_cost_value,
        "confusion_matrix": matrix,
    }


def model_feature_table(model_result: Pipeline, feature_names: list[str], model_name: str) -> pd.DataFrame:
    estimator = model_result.named_steps["model"]
    if model_name == "Logistic Regression":
        coefficients = estimator.coef_.ravel()
        table = pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "odds_ratio": np.exp(coefficients),
                "absolute_coefficient": np.abs(coefficients),
            }
        ).sort_values("absolute_coefficient", ascending=False)
        return table.drop(columns="absolute_coefficient")

    importances = estimator.feature_importances_
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    return table


def fit_models(
    history_location: pd.DataFrame,
) -> tuple[dict[str, ModelResult], pd.DataFrame, tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    feature_frame = history_location[PRIMARY_FEATURES].copy()
    target = history_location["broken"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    building_categories = sorted(X_train["building_type"].dropna().astype(str).unique().tolist())
    logistic_preprocessor, forest_preprocessor = build_preprocessors(building_categories)

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", logistic_preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", forest_preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipelines = {
        "Logistic Regression": logistic_pipeline,
        "Random Forest": forest_pipeline,
    }

    model_results: dict[str, ModelResult] = {}
    comparison_rows: list[dict[str, Any]] = []

    for model_name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        auc_score_value = roc_auc_score(y_test, probabilities)
        brier = brier_score_loss(y_test, probabilities)
        metrics_by_threshold = {
            0.5: threshold_metrics(y_test, probabilities, 0.5),
            BUSINESS_THRESHOLD: threshold_metrics(y_test, probabilities, BUSINESS_THRESHOLD),
        }
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
        feature_table = model_feature_table(pipeline, feature_names, model_name)

        result = ModelResult(
            name=model_name,
            pipeline=pipeline,
            auc_score=auc_score_value,
            brier_score=brier,
            probabilities=probabilities,
            metrics_by_threshold=metrics_by_threshold,
            feature_table=feature_table,
            confusion_matrices={threshold: metrics["confusion_matrix"] for threshold, metrics in metrics_by_threshold.items()},
        )
        model_results[model_name] = result

        for threshold, metrics in metrics_by_threshold.items():
            comparison_rows.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "roc_auc": auc_score_value,
                    "brier_score": brier,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "installed_count": metrics["installed_count"],
                    "expected_cost": metrics["expected_cost"],
                    "realized_cost": metrics["realized_cost"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tp": metrics["tp"],
                }
            )

    model_comparison = pd.DataFrame(comparison_rows).sort_values(["threshold", "realized_cost", "roc_auc"])
    return model_results, model_comparison, (X_train, X_test, y_train, y_test)


def plot_model_figures(
    model_results: dict[str, ModelResult],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Path]:
    figure_paths: dict[str, Path] = {}

    plt.figure(figsize=(7, 5))
    for model_name, result in model_results.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, result.probabilities)
        roc_auc_value = auc(false_positive_rate, true_positive_rate)
        plt.plot(false_positive_rate, true_positive_rate, label=f"{model_name} (AUC = {roc_auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve on Test Set")
    plt.legend()
    figure_paths["roc_curve"] = save_current_figure("roc_curve_comparison.png")

    for model_name, result in model_results.items():
        for threshold, matrix in result.confusion_matrices.items():
            plt.figure(figsize=(5, 4))
            display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["Not broken", "Broken"])
            display.plot(cmap="Blues", values_format="d", ax=plt.gca(), colorbar=False)
            plt.title(f"{model_name} Confusion Matrix (threshold = {threshold:.1f})")
            file_stub = model_name.lower().replace(" ", "_")
            threshold_stub = str(threshold).replace(".", "_")
            figure_paths[f"{file_stub}_cm_{threshold_stub}"] = save_current_figure(f"{file_stub}_confusion_matrix_t{threshold_stub}.png")

    forest_table = model_results["Random Forest"].feature_table.head(10).sort_values("importance")
    plt.figure(figsize=(7, 5))
    plt.barh(forest_table["feature"], forest_table["importance"], color="#74a57f")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    figure_paths["random_forest_feature_importance"] = save_current_figure("random_forest_feature_importance.png")

    forest_pipeline = model_results["Random Forest"].pipeline
    permutation = permutation_importance(
        forest_pipeline,
        X_test,
        y_test,
        n_repeats=15,
        random_state=RANDOM_STATE,
        scoring="roc_auc",
        n_jobs=-1,
    )
    permutation_table = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    save_table(permutation_table, "permutation_importance.csv")

    plt.figure(figsize=(7, 4))
    top_permutation = permutation_table.head(10).sort_values("importance_mean")
    plt.barh(top_permutation["feature"], top_permutation["importance_mean"], xerr=top_permutation["importance_std"], color="#ef8354")
    plt.title("Permutation Importance (ROC AUC)")
    plt.xlabel("Mean importance")
    figure_paths["permutation_importance"] = save_current_figure("permutation_importance.png")

    return figure_paths


def choose_selected_model(model_comparison: pd.DataFrame) -> str:
    business_view = model_comparison.loc[model_comparison["threshold"] == BUSINESS_THRESHOLD].copy()
    business_view = business_view.sort_values(["realized_cost", "brier_score", "roc_auc"], ascending=[True, True, False])
    return str(business_view.iloc[0]["model"])


def expected_cost_from_probabilities(probabilities: np.ndarray, install_flags: np.ndarray, effectiveness: float) -> float:
    install_component = install_flags * (CAMERA_COST + (1 - effectiveness) * probabilities * VANDALISM_LOSS)
    no_install_component = (1 - install_flags) * probabilities * VANDALISM_LOSS
    return float((install_component + no_install_component).sum())


def score_candidates(
    candidate_location: pd.DataFrame,
    model_results: dict[str, ModelResult],
    selected_model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_features = candidate_location[PRIMARY_FEATURES].copy()
    scored_frames: list[pd.DataFrame] = []

    for model_name, result in model_results.items():
        probabilities = result.pipeline.predict_proba(candidate_features)[:, 1]
        scored = candidate_location.copy()
        scored["model_name"] = model_name
        scored["predicted_probability"] = probabilities
        scored["install_camera_threshold_0_2"] = (probabilities >= BUSINESS_THRESHOLD).astype(int)
        scored_frames.append(scored)

    all_scores = pd.concat(scored_frames, ignore_index=True)
    selected_scores = all_scores.loc[all_scores["model_name"] == selected_model_name].copy()
    selected_scores["install_camera"] = selected_scores["install_camera_threshold_0_2"]
    selected_scores["risk_segment"] = pd.cut(
        selected_scores["predicted_probability"],
        bins=[-np.inf, 0.1, 0.2, np.inf],
        labels=["Low", "Medium", "High"],
    )
    selected_scores["expected_benefit_e1"] = selected_scores["predicted_probability"] * VANDALISM_LOSS - CAMERA_COST
    selected_scores = selected_scores.sort_values("predicted_probability", ascending=False)

    model_strategy_flags = selected_scores["install_camera"].to_numpy()
    model_expected_cost = expected_cost_from_probabilities(
        selected_scores["predicted_probability"].to_numpy(),
        model_strategy_flags,
        effectiveness=1.0,
    )
    human_rule_flags = (selected_scores["indoor"] == 0).astype(int).to_numpy()
    human_expected_cost = expected_cost_from_probabilities(
        selected_scores["predicted_probability"].to_numpy(),
        human_rule_flags,
        effectiveness=1.0,
    )

    strategy_comparison = pd.DataFrame(
        [
            {
                "strategy": "Model-based",
                "cameras_installed": int(model_strategy_flags.sum()),
                "installation_cost": int(model_strategy_flags.sum()) * CAMERA_COST,
                "expected_vandalism_loss": model_expected_cost - int(model_strategy_flags.sum()) * CAMERA_COST,
                "total_expected_cost": model_expected_cost,
            },
            {
                "strategy": "Outdoor-only human rule",
                "cameras_installed": int(human_rule_flags.sum()),
                "installation_cost": int(human_rule_flags.sum()) * CAMERA_COST,
                "expected_vandalism_loss": human_expected_cost - int(human_rule_flags.sum()) * CAMERA_COST,
                "total_expected_cost": human_expected_cost,
            },
        ]
    )
    strategy_comparison["cost_gap_vs_model"] = strategy_comparison["total_expected_cost"] - model_expected_cost

    risk_profile = (
        selected_scores.groupby("risk_segment", observed=False)
        .agg(
            machine_count=("location_id", "size"),
            avg_probability=("predicted_probability", "mean"),
            avg_distance=("distance_to_store", "mean"),
            indoor_rate=("indoor", "mean"),
            club_activity_share=("club_activity_day", "mean"),
        )
        .reset_index()
    )

    return all_scores, selected_scores, strategy_comparison, risk_profile


def summarize_high_risk(selected_scores: pd.DataFrame) -> pd.DataFrame:
    high_risk = selected_scores.loc[selected_scores["install_camera"] == 1].copy()
    overall = selected_scores.copy()

    summary = pd.DataFrame(
        [
            {
                "metric": "Machine count",
                "high_risk": len(high_risk),
                "overall": len(overall),
            },
            {
                "metric": "Outdoor share",
                "high_risk": 1 - high_risk["indoor"].mean(),
                "overall": 1 - overall["indoor"].mean(),
            },
            {
                "metric": "Average distance to store",
                "high_risk": high_risk["distance_to_store"].mean(),
                "overall": overall["distance_to_store"].mean(),
            },
            {
                "metric": "Weekend share",
                "high_risk": high_risk["weekend"].mean(),
                "overall": overall["weekend"].mean(),
            },
            {
                "metric": "Club activity day share",
                "high_risk": high_risk["club_activity_day"].mean(),
                "overall": overall["club_activity_day"].mean(),
            },
            {
                "metric": "Average predicted probability",
                "high_risk": high_risk["predicted_probability"].mean(),
                "overall": overall["predicted_probability"].mean(),
            },
        ]
    )
    return summary


def plot_business_figures(selected_scores: pd.DataFrame, sensitivity_table: pd.DataFrame) -> dict[str, Path]:
    figure_paths: dict[str, Path] = {}

    high_risk_building = selected_scores.loc[selected_scores["install_camera"] == 1].groupby("building_type").size().sort_values(ascending=False)
    plt.figure(figsize=(7, 4))
    labeled_bar(plt.gca(), high_risk_building.index.tolist(), high_risk_building.tolist(), as_percent=False)
    plt.title("Building Type Distribution of Recommended Cameras")
    plt.ylabel("Machine count")
    figure_paths["high_risk_building_distribution"] = save_current_figure("high_risk_building_distribution.png")

    plt.figure(figsize=(7, 4))
    plt.plot(sensitivity_table["effectiveness"], sensitivity_table["total_expected_cost"], marker="o", color="#2f6690")
    plt.title("Camera Effectiveness vs Total Expected Cost")
    plt.xlabel("Camera effectiveness")
    plt.ylabel("Total expected cost")
    figure_paths["sensitivity_total_cost"] = save_current_figure("sensitivity_total_expected_cost.png")

    plt.figure(figsize=(7, 4))
    plt.plot(sensitivity_table["effectiveness"], sensitivity_table["cameras_installed"], marker="o", color="#74a57f")
    plt.title("Camera Effectiveness vs Cameras Installed")
    plt.xlabel("Camera effectiveness")
    plt.ylabel("Number of cameras installed")
    figure_paths["sensitivity_camera_count"] = save_current_figure("sensitivity_camera_count.png")

    plt.figure(figsize=(7, 4))
    plt.plot(sensitivity_table["effectiveness"], sensitivity_table["threshold"], marker="o", color="#ef8354")
    plt.title("Camera Effectiveness vs Installation Threshold")
    plt.xlabel("Camera effectiveness")
    plt.ylabel("Threshold probability")
    figure_paths["sensitivity_threshold"] = save_current_figure("sensitivity_threshold.png")

    return figure_paths


def run_sensitivity_analysis(selected_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    probabilities = selected_scores["predicted_probability"].to_numpy()

    for effectiveness in SENSITIVITY_LEVELS:
        if effectiveness == 0:
            threshold = np.inf
            install_flags = np.zeros_like(probabilities, dtype=int)
        else:
            threshold = BUSINESS_THRESHOLD / effectiveness
            if threshold > 1:
                install_flags = np.zeros_like(probabilities, dtype=int)
            else:
                install_flags = (probabilities >= threshold).astype(int)
        total_expected_cost = expected_cost_from_probabilities(probabilities, install_flags, effectiveness)
        installation_cost = int(install_flags.sum()) * CAMERA_COST
        rows.append(
            {
                "effectiveness": effectiveness,
                "threshold": threshold if np.isfinite(threshold) else np.nan,
                "cameras_installed": int(install_flags.sum()),
                "installation_cost": installation_cost,
                "remaining_expected_vandalism_loss": total_expected_cost - installation_cost,
                "total_expected_cost": total_expected_cost,
            }
        )

    sensitivity = pd.DataFrame(rows)
    return sensitivity


def budget_scenarios(selected_scores: pd.DataFrame, effectiveness: float = 1.0) -> pd.DataFrame:
    ranked = selected_scores.sort_values("expected_benefit_e1", ascending=False).copy()
    rows: list[dict[str, Any]] = []
    for budget in BUDGET_SCENARIOS:
        chosen = ranked.head(budget)
        install_flags = selected_scores["location_id"].isin(chosen["location_id"]).astype(int).to_numpy()
        total_expected_cost = expected_cost_from_probabilities(
            selected_scores["predicted_probability"].to_numpy(),
            install_flags,
            effectiveness=effectiveness,
        )
        rows.append(
            {
                "budget": budget,
                "selected_machine_count": int(install_flags.sum()),
                "installation_cost": int(install_flags.sum()) * CAMERA_COST,
                "total_expected_cost": total_expected_cost,
                "avg_probability_selected": chosen["predicted_probability"].mean(),
                "min_probability_selected": chosen["predicted_probability"].min(),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, decimals: int = 4) -> str:
    formatted = df.copy()
    for column in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:,.{decimals}f}" if pd.notna(value) else "")
    headers = [str(column) for column in formatted.columns]
    separator = ["---"] * len(headers)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(separator) + " |"]
    for _, row in formatted.iterrows():
        rows.append("| " + " | ".join(str(value) for value in row.tolist()) + " |")
    return "\n".join(rows)


def build_report(
    artifacts: AnalysisArtifacts,
    figure_paths: dict[str, Path],
) -> Path:
    history_location = artifacts.history_location
    descriptive_tables = artifacts.descriptive_tables
    model_results = artifacts.model_results
    selected_scores = artifacts.selected_candidate_scores
    selected_model_name = artifacts.selected_model_name
    strategy_comparison = artifacts.strategy_comparison
    sensitivity_table = artifacts.sensitivity_table
    budget_table = artifacts.budget_table

    logistic_auc = model_results["Logistic Regression"].auc_score
    forest_auc = model_results["Random Forest"].auc_score
    broken_rate = history_location["broken"].mean()
    selected_install_count = int(selected_scores["install_camera"].sum())
    model_strategy_cost = float(strategy_comparison.loc[strategy_comparison["strategy"] == "Model-based", "total_expected_cost"].iloc[0])
    human_strategy_cost = float(strategy_comparison.loc[strategy_comparison["strategy"] == "Outdoor-only human rule", "total_expected_cost"].iloc[0])
    sensitivity_turning_point = sensitivity_table.loc[sensitivity_table["cameras_installed"] > 0, "effectiveness"]
    effective_break_even = sensitivity_turning_point.min() if not sensitivity_turning_point.empty else np.nan

    logistic_table = model_results["Logistic Regression"].feature_table.copy()
    random_forest_table = model_results["Random Forest"].feature_table.copy()
    permutation_table = pd.read_csv(TABLE_DIR / "permutation_importance.csv")
    high_risk_summary = summarize_high_risk(selected_scores)
    risk_profile = (
        selected_scores.groupby("risk_segment", observed=False)
        .agg(
            machine_count=("location_id", "size"),
            avg_probability=("predicted_probability", "mean"),
            avg_distance=("distance_to_store", "mean"),
        )
        .reset_index()
    )

    business_interpretation = [
        "距離便利商店越遠的點位在 Random Forest 中最重要，代表外部監督或人流可見度可能是核心風險因子。",
        "Logistic Regression 的係數顯示 dorm 與 baseline classroom 相比風險略低，而 indoor 的係數接近零，表示單靠室內外不足以完全解釋破壞。",
        "Club activity day 的平均值在高風險群較高，支持活動日人潮或停留行為與破壞風險關聯。",
        "Weekend 在 location-level 幾乎沒有區辨力，因為每個地點面對類似的週末日曆結構，商業上不宜過度解讀。",
    ]

    report = f"""# Camera Installation Decision for Vending Machines

## Executive Summary
- 歷史資料共 {len(history_location):,} 台販賣機，Broken rate 約為 {broken_rate:.1%}，屬於偏高風險場景，在目前成本結構下低門檻保護策略有經濟合理性。
- 兩個模型的整體辨識力都有限，Random Forest 的 ROC AUC 較高 ({forest_auc:.3f} vs {logistic_auc:.3f})，但 Logistic Regression 在 threshold = 0.2 的測試集 realized cost 較低，因此最終商業決策採用 Logistic Regression。
- 在最佳模型下，1,000 個 candidate locations 中共有 {selected_install_count:,} 台建議安裝 camera。
- Model-based strategy 的總預期成本為 ${model_strategy_cost:,.0f}，低於 outdoor-only human rule 的 ${human_strategy_cost:,.0f}。
- Camera effectiveness 至少要達到約 {effective_break_even:.1f} 之後，模型才開始建議安裝任何 camera；之後效果愈好，安裝門檻愈低。
- 距離便利商店、社團活動暴露程度與建築類型是最值得優先管理的訊號，單純 indoor/outdoor 規則不夠精準。
- 在本案成本比下，若預算允許，全面保護或大規模保護策略可能優於只保護 outdoor 的經驗法則。

## 1. Data Overview
- 歷史訓練資料來源為 `given/sales-data.csv`，先依 `location_id` 聚合成 3,000 筆 location-level observations，再與 `given/broken.csv` 合併 target。
- 預測部署對象為 `given/candidate-location.csv` 的 1,000 個 location，先依 location 聚合後再套用最佳模型。
- 標準欄位 mapping 如下：

{markdown_table(artifacts.mapping_table)}

- Broken rate: {broken_rate:.1%}
- 缺失值：三份原始檔案都沒有明顯缺失值，但 candidate data 有 `indoor` 隨日期變動的資料品質問題，因此本分析採用每個 location 的 mode 作為穩健值。
- 重要欄位定義：`weekend` 與 `club_activity_day` 於 location-level 以年度中該事件出現比例表示。

## 2. Question 1: Descriptive Analysis
### 2.1 Summary Statistics by Broken Status
{markdown_table(descriptive_tables['overview'])}

Binary features summary:

{markdown_table(descriptive_tables['binary_summary'])}

Distance summary:

{markdown_table(descriptive_tables['distance_summary'])}

Building type distribution and broken rate:

{markdown_table(descriptive_tables['building_distribution'])}

解讀：
- 單看 unconditional summary statistics，Broken = 1 與 Broken = 0 的平均差異其實不大，說明這份資料的可分性偏弱，不能只靠單一變數做判斷。
- 相對較有訊號的是 building type 與 indoor 的 broken rate 差異，但幅度仍屬溫和；`weekend` 幾乎沒有辨識力。
- 商業上這代表 vandalism risk 比較像是多個場域條件共同作用，而不是由單一規則直接驅動，因此需要模型把多個弱訊號合併起來看。

### 2.2 Visualizations
![Broken distribution](figures/{figure_paths['broken_distribution'].name})

整體 broken rate 不低，意味著如果 camera 能有效降低損失，決策門檻會相對容易被跨過。

![Broken rate by building type](figures/{figure_paths['building_type_broken_rate'].name})

不同 building type 的 broken rate 有可觀差異，代表場域屬性本身反映不同的人流、管理密度與暴露風險。

![Indoor broken rate](figures/{figure_paths['indoor_broken_rate'].name})

Indoor 與 outdoor 有一些差異，但幅度不大，因此只用 outdoor-only rule 做部署仍然過於粗糙。

![Weekend broken rate](figures/{figure_paths['weekend_broken_rate'].name})

Weekend 幾乎沒有區辨力，顯示單純依賴週末時段做長期設備部署不夠有效。

![Club activity broken rate](figures/{figure_paths['club_activity_broken_rate'].name})

Club activity 的單變數差異不算大，但在與其他特徵一起使用時仍可能提供額外排序資訊，適合作為巡檢與監控加強的輔助訊號。

![Distance boxplot](figures/{figure_paths['distance_boxplot'].name})

Distance 的單變數分布差異有限，但在樹模型中仍是最重要特徵，表示它比較像是和其他條件一起作用的交互型訊號。

![Building by indoor heatmap](figures/{figure_paths['building_indoor_heatmap'].name})

Building type 與 indoor 之間存在交互差異，不同建築物在室內外場景下的風險並不一致。

![Indoor by club activity heatmap](figures/{figure_paths['indoor_club_heatmap'].name})

Indoor 與社團活動日的交互圖顯示，活動暴露可能會放大原有風險，值得列入營運排程管理。

## 3. Question 2: Classification Modeling
### 3.1 Logistic Regression
Baseline category: `classroom`

Coefficient and odds ratio table:

{markdown_table(logistic_table.head(10))}

Performance table:

{markdown_table(artifacts.model_comparison.loc[artifacts.model_comparison['model'] == 'Logistic Regression', ['model', 'threshold', 'roc_auc', 'brier_score', 'accuracy', 'precision', 'recall', 'f1_score', 'installed_count', 'expected_cost', 'realized_cost', 'tn', 'fp', 'fn', 'tp']])}

![ROC curve](figures/{figure_paths['roc_curve'].name})

Logistic Regression 在 threshold = 0.2 時 recall 很高，符合「寧可多裝也不要錯過高損失點位」的商業邏輯，但代價是 precision 偏低、決策容易趨近全面安裝。

### 3.2 Random Forest
Feature importance table:

{markdown_table(random_forest_table.head(10))}

Permutation importance table:

{markdown_table(permutation_table.head(10))}

Performance table:

{markdown_table(artifacts.model_comparison.loc[artifacts.model_comparison['model'] == 'Random Forest', ['model', 'threshold', 'roc_auc', 'brier_score', 'accuracy', 'precision', 'recall', 'f1_score', 'installed_count', 'expected_cost', 'realized_cost', 'tn', 'fp', 'fn', 'tp']])}

![Random Forest feature importance](figures/{figure_paths['random_forest_feature_importance'].name})

![Permutation importance](figures/{figure_paths['permutation_importance'].name})

Random Forest 的 AUC 較高，說明它的排序能力略好，但 calibration 與 threshold 0.2 下的 realized cost 不如 Logistic Regression。

### 3.3 Model Comparison
{markdown_table(artifacts.model_comparison[['model', 'threshold', 'roc_auc', 'brier_score', 'accuracy', 'precision', 'recall', 'f1_score', 'expected_cost', 'realized_cost']])}

建議採用 {selected_model_name} 作為最後商業決策模型，原因是本案的成本函數偏向保守防禦，threshold = 0.2 下的實際成本表現比單純 AUC 更重要。雖然 Random Forest 在 AUC 上略勝，但 improvement 不大，且並沒有轉化成更低的測試集 realized cost。

### 3.4 Most Influential Features
- {business_interpretation[0]}
- {business_interpretation[1]}
- {business_interpretation[2]}
- {business_interpretation[3]}

## 4. Question 3: Business Recommendation
### 4.1 Model-based Camera Installation Decision
- 決策邏輯：若 camera effectiveness = 1，當 $p_i > 0.2$ 時應安裝 camera。
- 採用模型：{selected_model_name}
- 建議安裝台數：{selected_install_count:,}
- 完整清單已輸出為 `outputs/tables/camera_recommendations.csv`

Strategy comparison:

{markdown_table(strategy_comparison)}

### 4.2 Characteristics of High-risk Machines
{markdown_table(high_risk_summary)}

Risk segmentation:

{markdown_table(risk_profile)}

![High-risk building distribution](figures/{figure_paths['high_risk_building_distribution'].name})

本案選定模型在 threshold = 0.2 下建議對全部 1,000 個 candidate locations 安裝 camera，因此「高風險子集合」實際上等於整個 candidate pool。這代表在目前成本假設下，模型認為整體風險水位都高於經濟門檻；若管理上無法全面部署，就應改用 budget scenario 與機率排序做優先配置。

### 4.3 Model-based Strategy vs Human Rule
- Outdoor-only rule 只根據 `indoor = 0` 裝設，忽略了部分 indoor 但同樣高風險的 location。
- Model-based strategy 的總預期成本較低，差額為 ${(human_strategy_cost - model_strategy_cost):,.0f}。
- 因此 model 提供了比 human rule 更好的 cost-benefit outcome。

### 4.4 Sensitivity Analysis
{markdown_table(sensitivity_table)}

![Sensitivity total cost](figures/{figure_paths['sensitivity_total_cost'].name})

![Sensitivity camera count](figures/{figure_paths['sensitivity_camera_count'].name})

![Sensitivity threshold](figures/{figure_paths['sensitivity_threshold'].name})

解讀：
- 當 effectiveness 很低時，threshold 會高於 1，因此沒有任何 machine 值得安裝 camera。
- effectiveness 提升時，threshold = 0.2 / e 會下降，表示只要 camera 更有效，就算機器只有中度風險也可能值得投資。
- 一旦 threshold 進入模型機率分布的密集區間，建議安裝數量會快速增加，形成明顯的轉折點。

## 5. Additional Insights and Strategic Recommendations
Budget scenarios:

{markdown_table(budget_table)}

- 若預算有限，應優先安裝在 predicted probability 與 expected benefit 最高的點位，而不是先從所有 outdoor 位置下手。
- 模型支持的建議：優先處理距離便利商店較遠、社團活動暴露較高、且屬於高 broken-rate building type 的位置。
- 合理但需要進一步驗證的建議：調整高風險販賣機位置、在社團活動日增加巡邏、補強照明與現場管理密度。
- 風險分群可以作為營運 SOP：Low risk 以例行巡檢為主，Medium risk 搭配重點巡檢或移機評估，High risk 優先考慮 camera 或實體監控升級。

## 6. Limitations and Next Steps
- 目前 target `Broken` 是過去一年結果，未必能完全代表下一年度行為模式。
- `Weekend` 與 `Club Activity Day` 來自日資料，但 target 是年度結果，因此存在時間尺度不一致問題。
- Candidate data 的 `indoor` 在部分 location 會隨日期切換，顯示原始資料品質可能有定義不一致，需要和資料提供方再確認。
- 模型整體 AUC 不高，代表現有變數只能解釋部分風險；若能加入人流量、照明、樓層、管理員、維修紀錄等變數，效果應可明顯提升。
- 若未來 camera 安裝後改變 vandalism 行為，模型需要重新校正，不能假設歷史資料永遠有效。
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    return REPORT_PATH


def run_analysis() -> AnalysisArtifacts:
    ensure_output_dirs()
    history_location, history_daily, candidate_location, mapping_table, data_quality = build_history_frames()

    descriptive_tables = build_descriptive_tables(history_location)
    for name, table in descriptive_tables.items():
        save_table(table, f"{name}.csv")
    save_table(mapping_table, "column_mapping.csv")

    descriptive_figures = plot_descriptive_figures(history_location, history_daily)

    model_results, model_comparison, split_data = fit_models(history_location)
    save_table(model_comparison, "model_comparison.csv")
    save_table(model_results["Logistic Regression"].feature_table, "logistic_coefficients.csv")
    save_table(model_results["Random Forest"].feature_table, "random_forest_feature_importance.csv")

    X_train, X_test, y_train, y_test = split_data
    model_figures = plot_model_figures(model_results, X_test, y_test)

    selected_model_name = choose_selected_model(model_comparison)
    _, selected_scores, strategy_comparison, risk_profile = score_candidates(
        candidate_location,
        model_results,
        selected_model_name,
    )

    selected_scores.to_csv(TABLE_DIR / "camera_recommendations.csv", index=False)
    save_table(strategy_comparison, "strategy_comparison.csv")
    save_table(risk_profile, "risk_profile.csv")
    high_risk_summary = summarize_high_risk(selected_scores)
    save_table(high_risk_summary, "high_risk_summary.csv")

    sensitivity_table = run_sensitivity_analysis(selected_scores)
    save_table(sensitivity_table, "sensitivity_analysis.csv")

    budget_table = budget_scenarios(selected_scores)
    save_table(budget_table, "budget_scenarios.csv")

    business_figures = plot_business_figures(selected_scores, sensitivity_table)

    all_figures = {}
    all_figures.update(descriptive_figures)
    all_figures.update(model_figures)
    all_figures.update(business_figures)

    artifacts = AnalysisArtifacts(
        history_location=history_location,
        history_daily=history_daily,
        candidate_location=candidate_location,
        mapping_table=mapping_table,
        data_quality=data_quality,
        descriptive_tables=descriptive_tables,
        model_results=model_results,
        model_comparison=model_comparison,
        selected_model_name=selected_model_name,
        selected_candidate_scores=selected_scores,
        strategy_comparison=strategy_comparison,
        sensitivity_table=sensitivity_table,
        budget_table=budget_table,
        report_path=REPORT_PATH,
    )

    build_report(artifacts, all_figures)
    return artifacts


if __name__ == "__main__":
    analysis = run_analysis()
    print(f"Report saved to: {analysis.report_path}")
    print(f"Selected model: {analysis.selected_model_name}")
    print(
        "Recommended cameras:",
        int(analysis.selected_candidate_scores["install_camera"].sum()),
    )
