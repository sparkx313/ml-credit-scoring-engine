"""Train and apply an offline probability-of-default baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_FORMAT_VERSION = 1


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"CSV file does not exist: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"CSV file contains no data rows: {path}")
    return frame


def validate_target(frame: pd.DataFrame, target: str) -> pd.Series:
    if target not in frame.columns:
        raise ValueError(f"Default label column '{target}' is missing from the CSV.")
    labels = pd.to_numeric(frame[target], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(
            f"Target '{target}' must contain numeric 0 (no default) and 1 (default) labels."
        )
    labels = labels.astype("int8")
    counts = labels.value_counts()
    if set(counts.index.tolist()) != {0, 1} or int(counts.min()) < 2:
        raise ValueError(
            "Training needs at least two examples from each class for a stratified holdout."
        )
    return labels


def clean_features(features: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    cleaned = features.copy()
    numeric_set = set(numeric_columns)
    for column in numeric_columns:
        values = pd.to_numeric(cleaned[column], errors="coerce")
        cleaned[column] = values.replace([np.inf, -np.inf], np.nan)
    for column in cleaned.columns:
        if column not in numeric_set:
            values = cleaned[column].astype("string").astype(object)
            cleaned[column] = values.where(pd.notna(values), np.nan)
    return cleaned


def build_model(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    transforms: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transforms.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transforms.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )
    return Pipeline(
        [
            ("features", ColumnTransformer(transforms, remainder="drop")),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def ensure_distinct_paths(input_path: Path, output_paths: list[Path]) -> None:
    resolved = [path.resolve() for path in output_paths]
    if input_path.resolve() in resolved:
        raise ValueError("An output path must not overwrite the input file.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Predictions, metrics, and model paths must be different.")


def train(args: argparse.Namespace) -> int:
    output_paths = [args.predictions, args.metrics, args.model]
    ensure_distinct_paths(args.data, output_paths)
    frame = read_csv(args.data)
    labels = validate_target(frame, args.target)

    id_column = args.id_column
    if id_column == args.target:
        raise ValueError("The ID column and target column must be different.")
    if id_column and id_column not in frame.columns:
        raise ValueError(f"ID column '{id_column}' is missing from the CSV.")
    excluded = [args.target] + ([id_column] if id_column else [])
    features = frame.drop(columns=excluded).dropna(axis=1, how="all")
    if features.empty:
        raise ValueError("No feature columns remain after excluding the target and ID.")
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    features = clean_features(features, numeric_columns)

    try:
        x_train, x_test, y_train, y_test, _train_rows, test_rows = train_test_split(
            features,
            labels,
            frame.index,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=labels,
        )
    except ValueError as exc:
        raise ValueError(f"Could not create the stratified holdout: {exc}") from exc

    usable_columns = [column for column in x_train.columns if x_train[column].notna().any()]
    if not usable_columns:
        raise ValueError("No feature has values in the training split.")
    x_train = x_train.loc[:, usable_columns]
    x_test = x_test.loc[:, usable_columns]
    features = features.loc[:, usable_columns]
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]
    model = build_model(numeric_columns, categorical_columns)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]

    metrics: dict[str, Any] = {
        "model": "logistic regression",
        "target_column": args.target,
        "rows_total": int(len(frame)),
        "rows_train": int(len(x_train)),
        "rows_test": int(len(x_test)),
        "default_rate": float(labels.mean()),
        "seed": int(args.seed),
        "metrics": {
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "brier_score": float(brier_score_loss(y_test, probabilities)),
            "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        },
        "feature_columns": features.columns.tolist(),
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
    }
    holdout_predictions = pd.DataFrame(
        {
            "source_row": np.asarray(test_rows) + 2,
            "actual_default": y_test.to_numpy(),
            "estimated_default_probability": probabilities,
        }
    )
    if id_column:
        holdout_predictions.insert(
            1, id_column, frame.loc[test_rows, id_column].to_numpy()
        )

    model_bundle = {
        "format_version": MODEL_FORMAT_VERSION,
        "pipeline": model,
        "feature_columns": features.columns.tolist(),
        "numeric_columns": numeric_columns,
        "id_column": id_column,
        "target_column": args.target,
    }
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_predictions.sort_values("source_row").to_csv(args.predictions, index=False)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    joblib.dump(model_bundle, args.model)

    print(f"Holdout ROC AUC: {metrics['metrics']['roc_auc']:.4f}")
    print(f"Holdout Brier score: {metrics['metrics']['brier_score']:.4f}")
    print(f"Saved holdout predictions to {args.predictions}")
    print(f"Saved metrics to {args.metrics}")
    print(f"Saved fitted model to {args.model}")
    return 0


def predict(args: argparse.Namespace) -> int:
    ensure_distinct_paths(args.input, [args.output, args.model])
    frame = read_csv(args.input)
    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict) or bundle.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("Unsupported model artifact format.")

    feature_columns = bundle["feature_columns"]
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError("Prediction CSV is missing feature(s): " + ", ".join(missing))
    features = frame.loc[:, feature_columns].copy()
    features = clean_features(features, bundle["numeric_columns"])
    probabilities = bundle["pipeline"].predict_proba(features)[:, 1]
    results = pd.DataFrame(
        {
            "source_row": frame.index.to_numpy() + 2,
            "estimated_default_probability": probabilities,
        }
    )
    id_column = bundle.get("id_column")
    if id_column and id_column in frame.columns:
        results.insert(1, id_column, frame[id_column].to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Scored {len(results)} record(s) to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and apply a local probability-of-default model."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train and evaluate on a holdout split.")
    train_parser.add_argument("--data", required=True, type=Path, help="Labeled applications CSV.")
    train_parser.add_argument("--target", default="defaulted", help="Binary 0/1 target column (default: defaulted).")
    train_parser.add_argument("--id-column", help="Optional record ID excluded from training and included in output.")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Stratified holdout fraction (default: 0.2).")
    train_parser.add_argument("--seed", type=int, default=42, help="Random split seed (default: 42).")
    train_parser.add_argument("--predictions", type=Path, default=Path("credit_holdout_predictions.csv"))
    train_parser.add_argument("--metrics", type=Path, default=Path("credit_metrics.json"))
    train_parser.add_argument("--model", type=Path, default=Path("credit_model.joblib"))
    train_parser.set_defaults(run=train)

    predict_parser = commands.add_parser("predict", help="Estimate default probabilities with a saved model.")
    predict_parser.add_argument("--model", required=True, type=Path, help="Artifact created by the train command.")
    predict_parser.add_argument("--input", required=True, type=Path, help="Applications CSV with the trained feature columns.")
    predict_parser.add_argument("--output", type=Path, default=Path("credit_probabilities.csv"))
    predict_parser.set_defaults(run=predict)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.run(args)
    except (OSError, ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
