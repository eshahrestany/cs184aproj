from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ============================================================
# Base neural net for biomarkers (PyTorch)
# ============================================================

class BiomarkerNet(nn.Module):
    """
    Simple MLP for biomarker features.

    Input:
        x: (batch, input_dim)

    Output:
        logits: (batch,) raw scores for malignant (before sigmoid).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        prev = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, 1))  # binary logits

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim)
        returns logits: (B,)
        """
        logits = self.net(x)  # (B, 1)
        return logits.squeeze(-1)


def create_biomarker_model(
    input_dim: int,
    hidden_dims: Sequence[int] = (64, 32),
    dropout: float = 0.1,
) -> BiomarkerNet:
    """
    Factory that matches your existing training code signature.
    """
    return BiomarkerNet(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)


# ============================================================
# Neural net "classifier" wrapper (for use in ensembles)
# ============================================================

@dataclass
class NeuralNetClassifier:
    """
    High-level wrapper around BiomarkerNet that exposes fit/predict/predict_proba
    in a scikit-learn-ish way, so it can be used inside an ensemble.

    This does its OWN training loop separate from your train_and_eval.py.
    """

    input_dim: int
    hidden_dims: Sequence[int] = (64, 32)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 64
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)
        self.model = BiomarkerNet(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

    def _iterate_batches(self, X: np.ndarray, y: np.ndarray):
        N = X.shape[0]
        idx = np.arange(N)
        np.random.shuffle(idx)
        for start in range(0, N, self.batch_size):
            end = min(start + self.batch_size, N)
            batch_idx = idx[start:end]
            xb = torch.tensor(X[batch_idx], dtype=torch.float32, device=self.device)
            yb = torch.tensor(y[batch_idx], dtype=torch.float32, device=self.device)
            yield xb, yb

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetClassifier":
        """
        X: (N, D) float32
        y: (N,) labels in {0,1}
        """
        self.model.train()
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        for ep in range(1, self.epochs + 1):
            epoch_loss = 0.0
            total = 0
            correct = 0

            for xb, yb in self._iterate_batches(X, y):
                opt.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

                with torch.no_grad():
                    epoch_loss += float(loss.item()) * xb.size(0)
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    correct += int((preds == yb).sum().item())
                    total += xb.size(0)

            avg_loss = epoch_loss / max(1, total)
            acc = correct / max(1, total)
            print(f"[NeuralNet] epoch {ep:03d}: loss={avg_loss:.4f}, acc={acc:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probabilities for class 1 as shape (N, 2) like sklearn:
        [:,0] = P(class=0), [:,1] = P(class=1)
        """
        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        N = X.shape[0]
        probs_1 = []

        with torch.no_grad():
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                xb = torch.tensor(X[start:end], dtype=torch.float32, device=self.device)
                logits = self.model(xb)
                p = torch.sigmoid(logits)  # (B,)
                probs_1.append(p.cpu().numpy())

        probs_1 = np.concatenate(probs_1, axis=0)
        probs_0 = 1.0 - probs_1
        return np.stack([probs_0, probs_1], axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)  # (N, 2)
        return (proba[:, 1] >= 0.5).astype(int)


# ============================================================
# Random Forest and SVM models (sklearn)
# ============================================================

def create_random_forest_model(
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    class_weight: Optional[str] = "balanced",
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Random forest classifier for biomarkers.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=n_jobs,
    )


def create_svm_model(
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    probability: bool = True,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
) -> SVC:
    """
    SVM classifier for biomarkers.
    """
    return SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        probability=probability,
        class_weight=class_weight,
        random_state=random_state,
    )


# ============================================================
# E-class ensemble: majority vote over NN, RF, SVM
# ============================================================

class EClassifier:
    """
    Ensemble classifier that runs 3 models:
        - NeuralNetClassifier (BiomarkerNet)
        - RandomForestClassifier
        - SVC (with probability=True)

    And takes the majority vote of their class predictions.

    All models are trained inside .fit(X, y).
    """

    def __init__(
        self,
        input_dim: int,
        nn_params: Optional[dict] = None,
        rf_params: Optional[dict] = None,
        svm_params: Optional[dict] = None,
    ):
        if nn_params is None:
            nn_params = {}
        if rf_params is None:
            rf_params = {}
        if svm_params is None:
            svm_params = {}

        # base models
        self.nn = NeuralNetClassifier(input_dim=input_dim, **nn_params)
        self.rf = create_random_forest_model(**rf_params)
        self.svm = create_svm_model(**svm_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EClassifier":
        """
        Fit all three models on the same training data.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        print("[EClassifier] Training NeuralNet...")
        self.nn.fit(X, y)

        print("[EClassifier] Training RandomForest...")
        self.rf.fit(X, y)

        print("[EClassifier] Training SVM...")
        self.svm.fit(X, y)

        return self

    def _predict_each(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)

        # Neural net uses its own wrapper
        y_nn = self.nn.predict(X)  # (N,)

        # sklearn models
        y_rf = self.rf.predict(X)  # (N,)
        y_svm = self.svm.predict(X)  # (N,)

        return y_nn.astype(int), y_rf.astype(int), y_svm.astype(int)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Majority vote over the 3 base models.

        For each sample:
            - if at least 2 say "malignant" (1), output 1
            - otherwise output 0
        """
        y_nn, y_rf, y_svm = self._predict_each(X)
        votes = np.stack([y_nn, y_rf, y_svm], axis=1)  # (N, 3)
        sums = votes.sum(axis=1)  # 0..3
        return (sums >= 2).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Soft version: average class-1 probabilities across models.
        Returns (N, 2) array like sklearn:
            [:,0] = P(class=0)
            [:,1] = P(class=1)
        """
        X = np.asarray(X, dtype=np.float32)

        # Neural net
        p_nn = self.nn.predict_proba(X)[:, 1]  # (N,)

        # RF + SVM
        p_rf = self.rf.predict_proba(X)[:, 1]
        p_svm = self.svm.predict_proba(X)[:, 1]

        p_mean = (p_nn + p_rf + p_svm) / 3.0
        p_mean = np.clip(p_mean, 0.0, 1.0)
        return np.stack([1.0 - p_mean, p_mean], axis=1)


def create_ensemble_eclassifier(
    input_dim: int,
    nn_params: Optional[dict] = None,
    rf_params: Optional[dict] = None,
    svm_params: Optional[dict] = None,
) -> EClassifier:
    return EClassifier(
        input_dim=input_dim,
        nn_params=nn_params,
        rf_params=rf_params,
        svm_params=svm_params,
    )
