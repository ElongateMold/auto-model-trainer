import matplotlib
matplotlib.use("Agg")

import argparse
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split

from core.eda import EDAAnalyzer
from core.loader import DatasetLoader
from core.model_selector import ModelSelector
from core.preprocessor import DataPreprocessor
from core.transformer import DataCleaner
from evaluation.metrics import ModelEvaluator
from evaluation.visualizer import EvaluationVisualizer
from models.base import get_model_trainer
from models.supervised.tree_models import TreeModelTrainer
from models.unsupervised.kmeans_engine import KMeansEngine


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular Auto-ML Framework (Supervised & Unsupervised Pipeline)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to input dataset (.csv, .tsv, or .parquet).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name (leave omitted for automatic detection).",
    )
    parser.add_argument(
        "--unsupervised",
        action="store_true",
        help="Force Unsupervised K-Means clustering mode.",
    )
    parser.add_argument(
        "--family",
        type=str,
        default="auto",
        choices=["auto", "linear", "tree_based", "spline", "prototype"],
        help="Supervised model family ('auto' lets ModelSelector decide).",
    )
    parser.add_argument(
        "--interactive-clean",
        action="store_true",
        help="Ask user before standardizing inconsistent category spellings.",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory where 300 DPI charts, metrics, and models are saved.",
    )
    return parser.parse_args()


def prompt_interactive_menu() -> tuple[Path, str | None, bool, str, bool]:
    print("\n" + "=" * 56)
    print("   MODULAR MACHINE LEARNING PIPELINE - INTERACTIVE MENU")
    print("=" * 56)

    while True:
        raw_path = input("1. Enter dataset path (e.g., data/customers.csv): ").strip()
        if raw_path and Path(raw_path).is_file():
            file_path = Path(raw_path)
            break
        print("   [!] File not found. Please enter a valid .csv, .tsv, or .parquet path.")

    print("\n2. Select execution mode:")
    print("   [1] Automatic (detect target column or fallback to K-Means)")
    print("   [2] Supervised (specify target column)")
    print("   [3] Unsupervised (force K-Means clustering audit)")
    mode_choice = input("   Choose an option [1/2/3] (default 1): ").strip()

    user_target: str | None = None
    force_unsupervised = False
    model_family = "auto"

    if mode_choice == "2":
        user_target = input("   Enter target column name: ").strip() or None
    elif mode_choice == "3":
        force_unsupervised = True

    if not force_unsupervised:
        print("\n3. Select supervised model family:")
        print("   [1] Auto-select via Structural Diagnostics (recommended)")
        print("   [2] Linear Models (Logistic / Ridge / Lasso)")
        print("   [3] Tree Models (CART / Random Forest)")
        print("   [4] Additive & Spline Models (MARS / GAM)")
        print("   [5] Prototype & Neighbor Models (k-NN / LVQ1 / DANN)")
        fam_map = {
            "1": "auto",
            "2": "linear",
            "3": "tree_based",
            "4": "spline",
            "5": "prototype",
        }
        fam_choice = input("   Choose an option [1-5] (default 1): ").strip()
        model_family = fam_map.get(fam_choice, "auto")

    clean_ans = input("\n4. Enable interactive category cleaning prompts? [y/N]: ").strip().lower()
    interactive_clean = clean_ans == "y"

    return file_path, user_target, force_unsupervised, model_family, interactive_clean


def run_pipeline(
    file_path: Path,
    user_target: str | None = None,
    force_unsupervised: bool = False,
    model_family_override: str = "auto",
    interactive_clean: bool = False,
    reports_dir: Path = Path("reports"),
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Universal Load & Automatic Supervision Detection (core/loader.py)
    loader = DatasetLoader()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        raw_df, detected_target, mode = loader.load(
            path=file_path,
            user_target=user_target,
            force_unsupervised=force_unsupervised,
        )
        for w in caught_warnings:
            print(f"\n[AUTO-ROUTING WARNING] {w.message}")

    print(f"\nLoaded '{file_path.name}' ({raw_df.shape[0]} rows, {raw_df.shape[1]} cols) -> Mode: {mode.upper()}")

    # 2. Category Standardization (core/transformer.py)
    clean_df = DataCleaner(raw_df).clean_categories(interactive=interactive_clean)

    # 3. Automated Structural EDA & 300 DPI Correlation Heatmaps (core/eda.py)
    is_unsupervised = mode == "unsupervised"
    diagnostic = EDAAnalyzer(
        clean_df,
        target_col=detected_target,
        unsupervised=is_unsupervised,
    ).run_structural_diagnostic(save_plots_dir=reports_dir)

    selector = ModelSelector(clean_df, diagnostic)

    # --- FLOW A: UNSUPERVISED K-MEANS CLUSTERING ---
    if is_unsupervised:
        pruned_df, dropped_summary = selector.prune_features()

        preprocessor = DataPreprocessor(scale_numeric=True)
        X_scaled = preprocessor.fit_transform(pruned_df)

        engine = KMeansEngine(k_range=range(2, 9))
        audit_table = engine.run_audit(X_scaled)
        clusters = engine.fit_predict(X_scaled)
        engine.plot_audit_and_pca(X_scaled, save_dir=reports_dir)

        audit_table.to_csv(reports_dir / "kmeans_audit.csv", index=False)
        clustered_df = clean_df.copy()
        clustered_df["cluster"] = clusters
        clustered_df.to_csv(reports_dir / "clustered_data.csv", index=False)

        print("\n=== UNSUPERVISED K-MEANS AUDIT ===")
        print(f"Dropped Columns  : {dropped_summary}")
        print(audit_table.to_string(index=False))
        print(
            f"\nElbow K: {engine.elbow_k} | Best Silhouette K: {engine.best_silhouette_k} "
            f"| Selected K: {engine.selected_k}"
        )
        print(f"Saved 300 DPI plots and cluster assignments to '{reports_dir}/'.")
        return

    # --- FLOW B: SUPERVISED LEARNING ---
    decision = selector.run()
    pruned_df = decision["pruned_df"]
    target_col = decision["target"]
    problem_type = decision["problem_type"]
    chosen_family = (
        decision["model_family"]
        if model_family_override == "auto"
        else model_family_override
    )

    # 4. Train/Test Split BEFORE scaling/encoding to prevent data leakage
    X = pruned_df.drop(columns=[target_col])
    y = pruned_df[target_col]
    stratify_y = y if problem_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_y
    )

    # 5. Conditional Scaling, Imputation & Encoding (core/preprocessor.py)
    needs_scaling = chosen_family != "tree_based"
    preprocessor = DataPreprocessor(scale_numeric=needs_scaling)
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    # 6. Cross-Validated Training & Prediction (models/base.py -> models/supervised/*)
    trainer = get_model_trainer(model_family=chosen_family, problem_type=problem_type)
    trainer.fit(X_train_ready, y_train)
    predictions = trainer.predict(X_test_ready)
    saved_models = trainer.save(reports_dir / "saved_models")

    # 7. Leaderboard Metrics & 300 DPI Evaluation Plots (evaluation/*)
    evaluator = ModelEvaluator(problem_type=problem_type)
    leaderboard = evaluator.evaluate_all(
        y_true=y_test,
        predictions=predictions,
        cv_summary=trainer.cv_results_summary,
    )
    winner = evaluator.get_winner(leaderboard)
    leaderboard.to_csv(reports_dir / "leaderboard.csv", index=False)

    visualizer = EvaluationVisualizer(
        problem_type=problem_type,
        save_dir=reports_dir,
        dpi=300,
    )
    visualizer.plot_all(
        models=trainer.best_models,
        X_train=X_train_ready,
        y_train=y_train,
        X_test=X_test_ready,
        y_test=y_test,
        predictions=predictions,
    )

    print(f"\nTarget Column    : {target_col} ({problem_type.upper()})")
    print(f"Selected Family  : {chosen_family} (StandardScaler={'ON' if needs_scaling else 'OFF'})")
    print(f"Dropped Columns  : {decision['dropped_columns']}")
    print("\n=== MODEL LEADERBOARD ===")
    print(leaderboard.to_string(index=False))
    print(f"\nWinning Model    : {winner}")
    print(f"Saved Models     : {list(saved_models.values())}")

    if isinstance(trainer, TreeModelTrainer):
        importances = trainer.get_feature_importances(
            feature_names=list(X_train_ready.columns),
            model_name=winner,
        )
        print(f"\n=== FEATURE IMPORTANCES ({winner}) ===")
        print(importances.to_string(index=False))


if __name__ == "__main__":
    args = parse_cli_args()

    if args.file is None:
        f_path, target, unsup, family, inter_clean = prompt_interactive_menu()
        run_pipeline(
            file_path=f_path,
            user_target=target,
            force_unsupervised=unsup,
            model_family_override=family,
            interactive_clean=inter_clean,
            reports_dir=Path(args.reports_dir),
        )
    else:
        run_pipeline(
            file_path=Path(args.file),
            user_target=args.target,
            force_unsupervised=args.unsupervised,
            model_family_override=args.family,
            interactive_clean=args.interactive_clean,
            reports_dir=Path(args.reports_dir),
        )