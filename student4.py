"""
STUDENT 4 -- Random Forest Classifier
Strategy: Train a Random Forest on pre-normalized features provided by
BaseModel, tune max_depth during refine_model(), then predict the
next-day direction (1 or -1) from the evaluate set.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from model import BaseModel


class Student4Model(BaseModel):

    name = "Student 4 Model -- Random Forest"

    # ── Stage 1: Train ────────────────────────────────────────────────────
    def train(self) -> None:
        """
        Fit a Random Forest on self.train_feat (80% of data).

        self.train_feat  -- pre-normalized features (log_return, range_pct,
                            close_open_pct, log_volume)
        self.train_labels -- next-day direction labels: 1 (UP) or -1 (DOWN)

        We store the fitted model and the best depth found so far so that
        refine_model() can improve on it without re-using training data.
        """
        X = self.train_feat.values
        y = self.train_labels.values

        self.best_depth = 6          # starting depth; refined in Stage 2
        self.model = RandomForestClassifier(
            n_estimators=200,        # 200 trees -- stable without being slow
            max_depth=self.best_depth,
            min_samples_leaf=10,     # prevents overfitting on small leaves
            class_weight="balanced", # handles any 1/-1 imbalance automatically
            random_state=42,
        )
        self.model.fit(X, y)
        train_acc = accuracy_score(y, self.model.predict(X))
        print(f"[Student4] Train accuracy: {train_acc:.3f}")

    # ── Stage 2: Refine ───────────────────────────────────────────────────
    def refine_model(self) -> None:
        """
        Tune max_depth using self.refine_feat (10% of data).

        We grid-search over a small set of depths and keep whichever
        model scores best on the refine set. We never touch self.train_set
        or self.train_feat here -- that would cause data leakage.
        """
        X_refine = self.refine_feat.values
        y_refine = self.refine_labels.values

        X_train = self.train_feat.values
        y_train = self.train_labels.values

        candidate_depths = [3, 4, 5, 6, 8, 10, None]  # None = fully grown tree
        best_acc = -1.0
        best_model = self.model  # fall back to the trained model if nothing improves

        for depth in candidate_depths:
            candidate = RandomForestClassifier(
                n_estimators=200,
                max_depth=depth,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
            )
            candidate.fit(X_train, y_train)       # refit on train each time
            acc = accuracy_score(y_refine, candidate.predict(X_refine))
            print(f"[Student4] Refine  max_depth={str(depth):>4s}  acc={acc:.3f}")

            if acc > best_acc:
                best_acc = acc
                best_model = candidate
                self.best_depth = depth

        self.model = best_model
        print(f"[Student4] Best depth after refinement: {self.best_depth}  "
              f"(refine acc={best_acc:.3f})")

    # ── Stage 3: Predict ──────────────────────────────────────────────────
    def predict(self) -> int:
        """
        Return a single directional signal for the evaluate set (10% of data).

        Strategy: run the refined model on every row in evaluate_feat,
        collect all predictions, and return the majority vote as the
        final signal. This smooths out any single noisy day.

        Returns:
            1  -> majority of evaluate days predicted UP
           -1  -> majority of evaluate days predicted DOWN
        """
        X_eval = self.evaluate_feat.values
        predictions = self.model.predict(X_eval)

        # Majority vote: sum of +1/-1 predictions
        signal = int(np.sign(predictions.sum()))

        # Fallback: if perfectly balanced (sum == 0), default to 1
        if signal == 0:
            signal = 1

        eval_acc = accuracy_score(self.evaluate_labels.values, predictions)
        print(f"[Student4] Evaluate accuracy: {eval_acc:.3f}  |  Signal: {signal:+d}")

        return signal
