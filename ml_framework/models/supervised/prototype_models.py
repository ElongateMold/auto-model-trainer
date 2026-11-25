import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from models.base import BaseModelTrainer


class LVQ1Classifier(BaseEstimator, ClassifierMixin):
    """
    Kohonen's Learning Vector Quantization 1 (LVQ1):
    Updates class prototypes via attraction (same class) and repulsion (different class).
    """

    def __init__(
        self,
        prototypes_per_class: int = 2,
        learning_rate: float = 0.05,
        n_epochs: int = 50,
        random_state: int = 42,
    ):
        self.prototypes_per_class = prototypes_per_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        self.classes_, y_idx = np.unique(y_arr, return_inverse=True)

        rng = np.random.default_rng(self.random_state)
        prototypes_list = []
        proto_labels_list = []

        # Initialize prototypes from random training samples per class
        for class_idx in range(len(self.classes_)):
            class_samples = X_arr[y_idx == class_idx]
            replace = len(class_samples) < self.prototypes_per_class
            chosen = rng.choice(len(class_samples), size=self.prototypes_per_class, replace=replace)
            prototypes_list.append(class_samples[chosen].copy())
            proto_labels_list.extend([class_idx] * self.prototypes_per_class)

        self.prototypes_ = np.vstack(prototypes_list)
        self.prototype_labels_ = np.array(proto_labels_list)

        # Iterative attraction / repulsion updates with linear learning-rate decay
        for epoch in range(self.n_epochs):
            alpha = self.learning_rate * (1.0 - epoch / self.n_epochs)
            order = rng.permutation(len(X_arr))

            for i in order:
                x_i = X_arr[i]
                y_i = y_idx[i]

                distances = np.linalg.norm(self.prototypes_ - x_i, axis=1)
                winner_idx = int(np.argmin(distances))

                if self.prototype_labels_[winner_idx] == y_i:
                    # Attraction: move prototype closer to x_i
                    self.prototypes_[winner_idx] += alpha * (x_i - self.prototypes_[winner_idx])
                else:
                    # Repulsion: push prototype away from x_i
                    self.prototypes_[winner_idx] -= alpha * (x_i - self.prototypes_[winner_idx])

        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        preds = []
        for x in X_arr:
            distances = np.linalg.norm(self.prototypes_ - x, axis=1)
            winner_idx = int(np.argmin(distances))
            preds.append(self.classes_[self.prototype_labels_[winner_idx]])
        return np.array(preds)


class DANNClassifier(BaseEstimator, ClassifierMixin):
    """
    Discriminant Adaptive Nearest-Neighbor (Hastie & Tibshirani, 1996):
    Computes a local deformable ellipsoidal metric Sigma(x_0) that stretches distances
    along between-class discriminant directions and shrinks them along within-class spread.
    """

    def __init__(self, n_neighbors: int = 5, n_neighbors_init: int = 50, epsilon: float = 1.0):
        self.n_neighbors = n_neighbors
        self.n_neighbors_init = n_neighbors_init
        self.epsilon = epsilon

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        self.classes_, self.y_train_ = np.unique(y_arr, return_inverse=True)
        return self

    def _compute_sigma(self, x0: np.ndarray) -> np.ndarray:
        n_samples, n_features = self.X_train_.shape
        k_init = min(n_samples, max(self.n_neighbors_init, self.n_neighbors + 1))

        # 1. Initial Euclidean neighborhood around x_0
        euclid_dists = np.linalg.norm(self.X_train_ - x0, axis=1)
        nbr_idx = np.argsort(euclid_dists)[:k_init]
        X_loc = self.X_train_[nbr_idx]
        y_loc = self.y_train_[nbr_idx]
        d_loc = euclid_dists[nbr_idx]

        # 2. Tricube kernel weights around x_0
        d_max = float(np.max(d_loc)) + 1e-12
        weights = (1.0 - (d_loc / d_max) ** 3) ** 3
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            return np.eye(n_features)
        weights /= weight_sum

        # 3. Local weighted Within-Class (W) and Between-Class (B) scatter matrices
        overall_mean = np.sum(weights[:, None] * X_loc, axis=0)
        W = np.zeros((n_features, n_features))
        B = np.zeros((n_features, n_features))

        for c in np.unique(y_loc):
            mask = y_loc == c
            pi_c = float(np.sum(weights[mask]))
            if pi_c <= 1e-12:
                continue
            mean_c = np.sum(weights[mask, None] * X_loc[mask], axis=0) / pi_c
            diff_b = (mean_c - overall_mean)[:, None]
            B += pi_c * (diff_b @ diff_b.T)

            diff_w = X_loc[mask] - mean_c
            W += (diff_w.T * weights[mask]) @ diff_w

        # 4. Sphere W via eigendecomposition and form Sigma(x_0)
        W_reg = W + 1e-4 * np.eye(n_features)
        eigvals, eigvecs = np.linalg.eigh(W_reg)
        eigvals = np.maximum(eigvals, 1e-6)
        W_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        B_star = W_inv_sqrt @ B @ W_inv_sqrt
        sigma = W_inv_sqrt @ (B_star + self.epsilon * np.eye(n_features)) @ W_inv_sqrt
        return sigma

    def predict(self, X):
        X_test = np.asarray(X, dtype=float)
        preds = []

        for x0 in X_test:
            sigma_x0 = self._compute_sigma(x0)
            deltas = self.X_train_ - x0
            # Ellipsoidal quadratic distance: (x - x0)^T Sigma(x0) (x - x0)
            dann_dists = np.sum((deltas @ sigma_x0) * deltas, axis=1)

            k = min(len(self.X_train_), self.n_neighbors)
            nearest_idx = np.argsort(dann_dists)[:k]
            nearest_labels = self.y_train_[nearest_idx]

            counts = np.bincount(nearest_labels)
            preds.append(self.classes_[int(np.argmax(counts))])

        return np.array(preds)


class PrototypeModelTrainer(BaseModelTrainer):
    def build_models(self) -> dict[str, BaseEstimator]:
        if self.problem_type == "classification":
            return {
                "KNN_Euclidean": KNeighborsClassifier(metric="euclidean", weights="distance"),
                "LVQ1": LVQ1Classifier(random_state=self.random_state),
                "DANN": DANNClassifier(),
            }
        # LVQ1 and DANN require discrete class labels; regression uses weighted Euclidean k-NN
        return {
            "KNN_Euclidean_Regressor": KNeighborsRegressor(metric="euclidean", weights="distance"),
        }

    def get_param_grids(self) -> dict[str, dict]:
        if self.problem_type == "classification":
            return {
                "KNN_Euclidean": {"n_neighbors": [3, 5, 9]},
                "LVQ1": {"prototypes_per_class": [1, 2, 3], "learning_rate": [0.02, 0.05]},
                "DANN": {"n_neighbors": [3, 5], "epsilon": [0.5, 1.0, 2.0]},
            }
        return {
            "KNN_Euclidean_Regressor": {"n_neighbors": [3, 5, 9]},
        }