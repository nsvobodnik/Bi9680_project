"""model_design.py – Oncogenic‑mutation classification pipeline.

The script reads the mutation table ``train.csv``, cleans it, splits the data,
tunes five sklearn classifiers with GridSearchCV (5‑fold stratified, no shuffle)
and evaluates the best model on both the training and test sets.  
It follows the house‑style defined in *ai_coding_setup.txt* [1]:
Google‑style docstrings, full type hints, `logging` instead of `print`,
descriptive names, blank‑line‑separated logical blocks and a lean CLI entry
point.
"""

# python model_design.py --data train.csv --output-dir artifacts

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier 
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    StratifiedGroupKFold,
    train_test_split,
    GroupKFold,
    cross_val_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------

# Expected column order – matches the header that ships with *train.csv* [2].
_EXPECTED_COLUMNS = [
    "uniprot",
    "pdb_code",
    "mutation",
    "structure",
    "protein_type",
    "essential",
    "domain",
    "essential_residues_all",
    "conservation",
    "msa_data",
    "mut_hydrophobicity",
    "label",
]

# ---------------------------------------------------------------------------


def _assign_names_and_index(df: pd.DataFrame) -> pd.DataFrame:
    """Promote a header row to column names, clean strings and build a stable index.

    * If the first row equals ``_EXPECTED_COLUMNS`` we assume the CSV was read
      with ``header=None`` and promote that row to the real header.
    * Otherwise we enforce the known column order.
    * Whitespace is stripped from string columns and empty strings become ``NaN``.
    * A deterministic identifier ``"<uniprot>|<pdb_code>|<mutation>"`` is set as
      the DataFrame index – useful for debugging and reproducible joins.
    """
    # ---------------------------------------------------------------
    # 1️⃣ Detect a header that was read as data.
    first_row = df.iloc[0].astype(str).str.strip().tolist()
    header_is_data = first_row == _EXPECTED_COLUMNS

    if header_is_data:
        log.info("Header row detected as data – promoting to column names.")
        df = df.iloc[1:].reset_index(drop=True)

    # ---------------------------------------------------------------
    # 2️⃣ Assign proper column names.
    if len(df.columns) != len(_EXPECTED_COLUMNS):
        raise ValueError(
            f"Expected {len(_EXPECTED_COLUMNS)} columns but got "
            f"{len(df.columns)}. Adjust _EXPECTED_COLUMNS or the CSV."
        )
    df.columns = _EXPECTED_COLUMNS

    # ---------------------------------------------------------------
    # 3️⃣ Clean string columns.
    str_cols = df.select_dtypes(include=["object"]).columns
    df[str_cols] = df[str_cols].apply(
        lambda s: s.str.strip().replace("", pd.NA)  # empty strings → NaN
    )

    # ---------------------------------------------------------------
    # 4️⃣ Build a stable identifier index.
    identifier = (
        df["uniprot"].astype(str)
        + "|"
        + df["pdb_code"].astype(str)
        + "|"
        + df["mutation"].astype(str)
    )
    df = df.set_index(identifier, drop=False)

    # ---------------------------------------------------------------
    # 5️⃣ Log missing‑value counts.
    missing = df.isna().sum()
    if missing.any():
        log.info("Missing values after cleaning:\n%s", missing[missing > 0])
    else:
        log.info("No missing values detected after cleaning.")

    # save rows with missing values to a separate file
    rows_with_nan = df[df.isna().any(axis=1)]
    if not rows_with_nan.empty:
        out_path = "missing_rows.csv"
        rows_with_nan.to_csv(out_path, index=False)

        logging.info(
            "Found %d rows with missing values – saved to %s",
            len(rows_with_nan),
            out_path,
    )
    else:
        logging.info("No missing values found; no extra file created.")

    return df


def _load_data(csv_path: Path) -> pd.DataFrame:
    """Read the CSV and return a cleaned DataFrame."""
    raw_df = pd.read_csv(csv_path, header=None, dtype=str)
    log.info("Raw shape (including possible header row): %s", raw_df.shape)
    df = _assign_names_and_index(raw_df)
    log.info("Final shape after column/index fixing: %s", df.shape)
    return df


def _plot_class_counts(df: pd.DataFrame, out_path: Path) -> None:
    """Save a bar chart of the Benign vs Oncogenic label distribution.

    Seaborn ≥ 0.14 requires a ``hue`` argument when a ``palette`` is supplied,
    so we pass ``hue=class_counts.index`` and suppress the legend.
    """
    class_counts = df["label"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=class_counts.index,
        y=class_counts.values,
        hue=class_counts.index,          # satisfies new seaborn API
        palette="viridis",
        legend=False,
    )
    plt.title("Class distribution: Benign vs Oncogenic")
    plt.xlabel("label")
    plt.ylabel("frequency")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Class‑count plot saved to %s", out_path)


def _make_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (binary target) ready for modelling.

    The malignant class (Oncogenic) is encoded as ``0`` and the benign class
    as ``1`` – exactly what the task specification requires.
    """
    feature_cols = [
        "structure",
        "essential",
        "essential_residues_all",
        "conservation",
        "msa_data",
        "mut_hydrophobicity",
        "protein_type",
        "domain",
    ]
    X = df[feature_cols].copy()
    y = (df["label"] == "Benign").astype(int)   # Oncogenic → 0, Benign → 1
    return X, y


# Kyte-Doolittle hydropathy index per amino acid
_HYDRO = {"A":1.8, "C":2.5, "D":-3.5,"E":-3.5,"F":2.8,"G":-0.4,"H":-3.2,"I":4.5,"K":-3.9, "L":3.8, "M":1.9,"N":-3.5,"P":-1.6,"Q":-3.5,"R":-4.5,
          "S":-0.8,"T":-0.7,"V":4.2, "W":-0.9,"Y":-1.3,}
           
def _add_mutation_features(X: pd.DataFrame)-> pd.DataFrame:
    """Fill missing mut_hydrophobicity from the mutant amino acid (Kyte-Doolittle scale)"""
    X = X.copy()
    parts = X["mutation"].astype(str).str.extract(r"^([A-Z])(\d+)([A-Z])$")
    mut = parts[2]
    X["mut_hydrophobicity"] = pd.to_numeric(X["mut_hydrophobicity"], errors="coerce")
    X["mut_hydrophobicity"] = X["mut_hydrophobicity"].fillna(mut.map(_HYDRO))  
    return(X.drop(columns=["mutation"]))   # TODO: decide whether to keep "mutation" for extra features


def _split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80 %/20 % train–test split."""
    return train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )


def _build_preprocessor() -> ColumnTransformer:
    """Preprocess numerics (median‑impute + scaler) and categoricals (mode‑impute + one‑hot)."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe,
             ["structure", "essential", "essential_residues_all",
              "conservation", "msa_data", "mut_hydrophobicity"]),
            ("categorical", categorical_pipe,
             ["protein_type", "domain"]),
        ]
    )

def _baseline_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> str:
    """ trains a DummyClassifier baseline and evaluates its accuracy by using the most frequent class """
    baseline = DummyClassifier(strategy="most_frequent", random_state=42)
    baseline.fit(X_train, y_train)
    baseline_acc = baseline.score(X_test, y_test)
    return(f"Dummy Classifier Test Accuracy (Most Frequent Class): {baseline_acc:.4f}\n")


def _build_model_candidates() -> Dict[str, Tuple[Pipeline, Dict[str, list[object]]]]:
    """Return five candidate pipelines together with their hyper‑parameter grids."""
    preproc = _build_preprocessor()

    candidates: Dict[str, Tuple[Pipeline, Dict[str, list[object]]]] = {
        "logistic_regression": (
            Pipeline([
                ("preprocess", preproc),
                ("classifier", LogisticRegression(max_iter=5_000,
                                                random_state=42)),
            ]),
            {
                "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            Pipeline([
                ("preprocess", preproc),
                ("classifier", RandomForestClassifier(
                    n_jobs=-1, random_state=42)),
            ]),
            {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
                "classifier__max_features": ["sqrt", "log2"],
            },
        ),
        "gradient_boosting": (
            Pipeline([
                ("preprocess", preproc),
                ("classifier", GradientBoostingClassifier(random_state=42)),
            ]),
            {
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [3, 5],
                "classifier__subsample": [1.0],
            },
        ),
        "svm": (
            Pipeline([
                ("preprocess", preproc),
                ("classifier", SVC(random_state=42)),
            ]),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__kernel": ["rbf", "linear"],
                "classifier__gamma": ["scale", "auto"],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "knn": (
            Pipeline([
                ("preprocess", preproc),
                ("classifier", KNeighborsClassifier()),
            ]),
            {
                "classifier__n_neighbors": [3, 5, 7, 9, 11],
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2],
            },
        ),
    }
    return candidates


def _tune_models(
    candidates: Dict[str, Tuple[Pipeline, Dict[str, list[object]]]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
) -> Dict[str, Tuple[Pipeline, float]]:
    """
    Run GridSearchCV for every candidate using a **group‑aware** CV splitter
    (StratifiedGroupKFold).  All models are therefore tuned on identical
    protein‑wise folds, preventing information leakage across mutations of the
    same protein.

    Args:
        candidates: Mapping name → (pipeline, param_grid).
        X_train: Feature matrix of the training split.
        y_train: Target vector of the training split.
        groups: Series that identifies the protein for each row
                (e.g. ``df["uniprot"]``).

    Returns:
        Dictionary name → (best_estimator, best_cv_accuracy).
    """
    # 5‑fold, stratified by the label, but keep all rows of a protein together.
    cv = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=42
    )  # ← replaces the plain StratifiedKFold used before [2]

    best_by_name: Dict[str, Tuple[Pipeline, float]] = {}

    for name, (pipeline, grid) in candidates.items():
        log.info("Tuning %s …", name)

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
        )
        # ``groups`` is forwarded so the CV splitter can enforce the protein
        # boundaries.
        search.fit(X_train, y_train, groups=groups)

        best_by_name[name] = (
            search.best_estimator_,
            float(search.best_score_),
        )
        log.info(
            "%s – best CV accuracy: %.4f (params: %s)",
            name,
            search.best_score_,
            search.best_params_,
        )

    return best_by_name


def _plot_cv_accuracies(
    results: Dict[str, Tuple[Pipeline, float]],
    out_path: Path,
) -> None:
    """Bar chart of the CV accuracies of all five models."""
    names = list(results.keys())
    scores = [acc for _, acc in results.values()]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, scores, color="#55a868")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Best CV accuracy")
    plt.title("Cross‑validated accuracy per model")
    for bar, acc in zip(bars, scores, strict=True):
        plt.annotate(f"{acc:.4f}",
                     (bar.get_x() + bar.get_width() / 2, acc),
                     ha="center", va="bottom")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("CV‑accuracy plot saved to %s", out_path)


def _evaluate_split(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
) -> Dict[str, float]:
    """Return accuracy, precision, recall and F1 for the malignant class (label 0)."""
    preds = estimator.predict(X)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, pos_label=0, zero_division=0)
    rec = recall_score(y, preds, pos_label=0, zero_division=0)
    f1 = f1_score(y, preds, pos_label=0, zero_division=0)
    return {
        "accuracy": acc,
        "precision (malignant)": prec,
        "recall (malignant)": rec,
        "f1 (malignant)": f1,
    }


def _print_metrics(
    model_name: str,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> None:
    """Pretty‑print the metric table for the selected model."""
    print(f"\nSelected model (best CV accuracy): {model_name}")
    header = f"{'Metric':<24}{'Train':>10}{'Test':>10}"
    print(header)
    print("-" * len(header))
    for metric in train_metrics:
        print(
            f"{metric:<24}{train_metrics[metric]:>10.4f}{test_metrics[metric]:>10.4f}"
        )

def _group_check(name, estimator, X, y, groups) -> None:
    """Evaluates accuracy of one model if whole proteins are held out.
    
    All mutations of one protein (identified by uniprot feature) are kept together, either 
    all in training or all in testing. The model is tested on data it has never seen. 
    It is repeated 5 times with different proteins held out."""
    scores = cross_val_score(estimator, X, y, groups=groups,
                             cv=GroupKFold(n_splits=5), scoring="accuracy")
    log.info("Unique proteins: %d", groups.nunique())
    log.info("%s | GroupKFold fold scores: %s", name, [round(float(s), 3) for s in scores])
    log.info("%s | GroupKFold mean accuracy: %.4f ± %.4f", name, scores.mean(), scores.std())


def _print_confusion_matrices(
    estimator: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Print train and test confusion matrices (rows = actual, cols = predicted)."""
    for split, X, y in (("train", X_train, y_train), ("test", X_test, y_test)):
        preds = estimator.predict(X)
        cm = confusion_matrix(y, preds, labels=[0, 1])
        print(f"\nConfusion matrix – {split} set (actual ↔ predicted)")
        print(f"{'':<20}{'pred Oncogenic':>16}{'pred Benign':>13}")
        print(f"{'actual Oncogenic':<20}{cm[0, 0]:>16}{cm[0, 1]:>13}")
        print(f"{'actual Benign':<20}{cm[1, 0]:>16}{cm[1, 1]:>13}")


def _build_parser() -> argparse.ArgumentParser:
    """Create the argument parser (lean CLI entry point – rule 10)."""
    parser = argparse.ArgumentParser(
        description="Compare, tune and evaluate classifiers for oncogenic mutations."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("train.csv"),
        help="Path to the mutation CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory in which plots are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Orchestrate the full end‑to‑end pipeline."""
    # -----------------------------------------------------------------------
    # 1️⃣  CLI parsing & logging configuration
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 2️⃣  Load & clean the data
    df = _load_data(args.data)

    # -----------------------------------------------------------------------
    # 3️⃣  Visualise class distribution
    _plot_class_counts(df, out_dir / "class_counts.png")

    # -----------------------------------------------------------------------
    # 4️⃣  Feature / target preparation and train‑test split

    df = _add_mutation_features(df) # adding computable missing values

    X, y = _make_features_and_target(df)
    X_train, X_test, y_train, y_test = _split_dataset(X, y)
    log.info(
        "Stratified split – %d train rows, %d test rows",
        X_train.shape[0],
        X_test.shape[0],
    )

    # -----------------------------------------------------------------------
    # 5️⃣  Build model candidates and run hyper‑parameter tuning

    baseline_output = _baseline_model(X_train, y_train, X_test, y_test) 
    log.info(baseline_output)

    candidates = _build_model_candidates()
    group_series = df.loc[X_train.index, "uniprot"]  # keep the same order as X_train
    tuned = _tune_models(candidates, X_train, y_train, groups=group_series)

    # -----------------------------------------------------------------------
    # 6️⃣  Select the best model (highest CV accuracy)
    best_name, (best_estimator, best_accuracy) = max(
        tuned.items(), key=lambda item: item[1][1]
    )
    log.info("Best model: %s (CV accuracy %.4f)", best_name, best_accuracy)

    _group_check(best_name, best_estimator, X, y, df["uniprot"]) 
    # -----------------------------------------------------------------------
    # 7️⃣  Evaluate on train and test splits
    train_metrics = _evaluate_split(best_estimator, X_train, y_train)
    test_metrics = _evaluate_split(best_estimator, X_test, y_test)

    # -----------------------------------------------------------------------
    # 8️⃣  Report results
    _print_metrics(best_name, train_metrics, test_metrics)
    _print_confusion_matrices(
        best_estimator, X_train, y_train, X_test, y_test,
    )

    # -----------------------------------------------------------------------
    # 9️⃣  Plot CV accuracies for all models
    _plot_cv_accuracies(tuned, out_dir / "cv_accuracies.png")


if __name__ == "__main__":
    main()