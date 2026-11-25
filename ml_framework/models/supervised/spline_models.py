import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from models.base import BaseModelTrainer


class MARSEstimator(BaseEstimator):
    """
    Multivariate Adaptive Regression Splines (MARS) using forward stepwise
    reflected hinge functions max(0, x - t) and max(0, t - x) and backward GCV pruning.
    """

    def __init__(self, max_terms: int = 12, penalty: float = 2.0, is_classifier: bool = False):
        self.max_terms = max_terms
        self.penalty = penalty
        self.is_classifier = is_classifier

    def _hinge_pair(self, X: np.ndarray, col: int, knot: float) -> tuple[np.ndarray, np.ndarray]:
        right = np.maximum(0.0, X[:, col] - knot)
        left = np.maximum(0.0, knot - X[:, col])
        return right, left

    def _gcv(self, y: np.ndarray, y_pred: np.ndarray, num_terms: int) -> float:
        n = len(y)
        rss = float(np.sum((y - y_pred) ** 2))
        effective_params = num_terms + self.penalty * (num_terms - 1) / 2.0
        denom = max(1.0 - effective_params / n, 1e-6)
        return rss / (n * (denom ** 2))

    def _build_design_matrix(self, X: np.ndarray, basis_specs: list[tuple[int, float, int]]) -> np.ndarray:
        cols = [np.ones(len(X))]
        for col_idx, knot, direction in basis_specs:
            if direction == 1:
                cols.append(np.maximum(0.0, X[:, col_idx] - knot))
            else:
                cols.append(np.maximum(0.0, knot - X[:, col_idx]))
        return np.column_stack(cols)

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        if self.is_classifier:
            self.classes_ = np.unique(y_arr)
            y_target = (y_arr == self.classes_[-1]).astype(float)
        else:
            y_target = y_arr.astype(float)

        n_samples, n_features = X_arr.shape
        selected_basis: list[tuple[int, float, int]] = []

        # Forward Pass: greedily add reflected hinge pairs that minimize RSS
        while len(selected_basis) < self.max_terms:
            best_rss = np.inf
            best_pair = None

            for col in range(n_features):
                knots = np.quantile(X_arr[:, col], q=np.linspace(0.1, 0.9, 9))
                for knot in np.unique(knots):
                    candidate_specs = selected_basis + [(col, float(knot), 1), (col, float(knot), -1)]
                    B = self._build_design_matrix(X_arr, candidate_specs)
                    coef, _, _, _ = np.linalg.lstsq(B, y_target, rcond=None)
                    rss = float(np.sum((y_target - B @ coef) ** 2))
                    if rss < best_rss:
                        best_rss = rss
                        best_pair = [(col, float(knot), 1), (col, float(knot), -1)]

            if best_pair is None:
                break
            selected_basis.extend(best_pair)

        # Backward Pass: prune individual basis terms that improve GCV
        best_specs = list(selected_basis)
        B_full = self._build_design_matrix(X_arr, best_specs)
        coef_full, _, _, _ = np.linalg.lstsq(B_full, y_target, rcond=None)
        best_gcv = self._gcv(y_target, B_full @ coef_full, len(best_specs) + 1)

        while len(best_specs) > 1:
            improved = False
            for idx in range(len(best_specs)):
                trial_specs = best_specs[:idx] + best_specs[idx + 1:]
                B_trial = self._build_design_matrix(X_arr, trial_specs)
                coef_trial, _, _, _ = np.linalg.lstsq(B_trial, y_target, rcond=None)
                trial_gcv = self._gcv(y_target, B_trial @ coef_trial, len(trial_specs) + 1)
                if trial_gcv < best_gcv:
                    best_gcv = trial_gcv
                    best_specs = trial_specs
                    improved = True
                    break
            if not improved:
                break

        self.basis_specs_ = best_specs
        B_final = self._build_design_matrix(X_arr, self.basis_specs_)
        self.coef_, _, _, _ = np.linalg.lstsq(B_final, y_target, rcond=None)
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        B = self._build_design_matrix(X_arr, self.basis_specs_)
        raw_pred = B @ self.coef_
        if self.is_classifier:
            idx = (raw_pred >= 0.5).astype(int)
            return self.classes_[idx]
        return raw_pred


class SplineModelTrainer(BaseModelTrainer):
    def build_models(self) -> dict[str, BaseEstimator]:
        if self.problem_type == "classification":
            return {
                "MARS_Classifier": MARSEstimator(is_classifier=True),
                "GAM_Logistic": Pipeline([
                    ("spline_basis", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
                    ("penalized_link", LogisticRegression(max_iter=1000, random_state=self.random_state)),
                ]),
            }
        return {
            "MARS_Regressor": MARSEstimator(is_classifier=False),
            "GAM_Spline": Pipeline([
                ("spline_basis", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
                ("penalized_link", Ridge(alpha=1.0, random_state=self.random_state)),
            ]),
        }

    def get_param_grids(self) -> dict[str, dict]:
        if self.problem_type == "classification":
            return {
                "MARS_Classifier": {"max_terms": [6, 12], "penalty": [2.0, 3.0]},
                "GAM_Logistic": {
                    "spline_basis__n_knots": [4, 6],
                    "penalized_link__C": [0.1, 1.0, 10.0],
                },
            }
        return {
            "MARS_Regressor": {"max_terms": [6, 12], "penalty": [2.0, 3.0]},
            "GAM_Spline": {
                "spline_basis__n_knots": [4, 6],
                "penalized_link__alpha": [0.1, 1.0, 10.0],
            },
        }