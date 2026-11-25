from pathlib import Path
from sklearn.model_selection import train_test_split

from core.loader import get_reader
from core.transformer import DataCleaner
from core.eda import EDAAnalyzer
from core.model_selector import ModelSelector
from core.preprocessor import DataPreprocessor
from models.base import get_model_trainer
from models.unsupervised.kmeans_engine import KMeansEngine
from evaluation.metrics import ModelEvaluator
from evaluation.visualizer import EvaluationVisualizer


def run_pipeline(
    file_path: Path,
    user_target: str | None = None,
    unsupervised_mode: bool = False,
    reports_dir: Path = Path("reports"),
):
    # 1. Extract (core/loader.py)
    raw_df = get_reader(file_path).read()

    # 2. Clean inconsistent text categories (core/transformer.py)
    clean_df = DataCleaner(raw_df).clean_categories(interactive=False)

    # 3. Structural EDA Diagnostic (runs for both supervised and unsupervised modes)
    diagnostic = EDAAnalyzer(
        clean_df,
        target_col=user_target,
        unsupervised=unsupervised_mode,
    ).run_structural_diagnostic(save_plots_dir=reports_dir)

    selector = ModelSelector(clean_df, diagnostic)

    # --- BRANCH A: UNSUPERVISED K-MEANS CLUSTERING ---
    if unsupervised_mode:
        pruned_df, dropped_summary = selector.prune_features()

        # Scale numeric features and One-Hot Encode surviving categories
        preprocessor = DataPreprocessor(scale_numeric=True)
        X_scaled = preprocessor.fit_transform(pruned_df)

        # Run full K-Means audit, assign clusters, and render Elbow/Silhouette/PCA plots
        engine = KMeansEngine(k_range=range(2, 9))
        audit_table = engine.run_audit(X_scaled)
        clusters = engine.fit_predict(X_scaled)
        engine.plot_audit_and_pca(X_scaled, save_dir=reports_dir)

        # Attach cluster labels back to the cleaned DataFrame for inspection
        clustered_df = clean_df.copy()
        clustered_df["cluster"] = clusters

        print("=== UNSUPERVISED K-MEANS PIPELINE ===")
        print(f"Dropped Columns  : {dropped_summary}")
        print(f"Features Clustered ({X_scaled.shape[1]}): {list(X_scaled.columns)}")
        print("\n=== K-MEANS AUDIT TABLE ===")
        print(audit_table.to_string(index=False))
        print(f"\nElbow K: {engine.elbow_k} | Best Silhouette K: {engine.best_silhouette_k}")
        print(f"Assigned {len(clusters)} rows into {engine.selected_k} clusters.")
        return

    # --- BRANCH B: SUPERVISED LEARNING ---
    # 4. Prune bad features & select supervised model family
    decision = selector.run()
    pruned_df = decision["pruned_df"]
    target_col = decision["target"]
    problem_type = decision["problem_type"]
    model_family = decision["model_family"]

    # 5. Train/Test Split (stratified for classification)
    X = pruned_df.drop(columns=[target_col])
    y = pruned_df[target_col]
    stratify_y = y if problem_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_y
    )

    # 6. Preprocess: Impute, Scale (when needed) & Encode
    needs_scaling = model_family != "tree_based"
    preprocessor = DataPreprocessor(scale_numeric=needs_scaling)
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    # 7. Cross-Validated Training & Prediction
    trainer = get_model_trainer(model_family=model_family, problem_type=problem_type)
    trainer.train(X_train_ready, y_train)
    predictions = trainer.predict(X_test_ready)

    # 8. Evaluation & HD Visualization
    evaluator = ModelEvaluator(problem_type=problem_type)
    leaderboard = evaluator.evaluate_all(
        y_true=y_test,
        predictions=predictions,
        cv_summary=trainer.cv_results_summary,
    )
    winner = evaluator.get_winner(leaderboard)

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

    print(f"Target: {target_col} ({problem_type}) | Family: {model_family}")
    print(f"Dropped Columns: {decision['dropped_columns']}")
    print("\n=== MODEL LEADERBOARD ===")
    print(leaderboard.to_string(index=False))
    print(f"\nWinning Model: {winner}")


if __name__ == "__main__":
    run_pipeline(Path("data/customers.csv"), unsupervised_mode=True)