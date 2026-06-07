"""
STUDENT 2 -- polynomial-kernel SVM model.
"""

import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from model import BaseModel


class Student2Model(BaseModel):
    name = "Polynomial SVM"

    def _fit_model(self, params: dict) -> SVC:
        """Fit a polynomial-kernel SVM on the training split."""
        model = SVC(
            kernel="poly",
            class_weight="balanced",
            random_state=42,
            **params,
        )
        model.fit(self.train_feat, self.train_labels)
        return model

    def train(self) -> None:
        """Fit the starting polynomial SVM model on training data."""
        self.candidate_params = [
            {"C": 1.0, "gamma": 0.5, "degree": 2, "coef0": 1},
            {"C": 2.0, "gamma": 0.5, "degree": 2, "coef0": 1},
            {"C": 5.0, "gamma": 0.5, "degree": 2, "coef0": 1},
            {"C": 2.0, "gamma": 0.5, "degree": 3, "coef0": 1},
            {"C": 5.0, "gamma": 0.5, "degree": 3, "coef0": 1},
            {"C": 10.0, "gamma": 0.5, "degree": 3, "coef0": 1},
            {"C": 1.0, "gamma": "scale", "degree": 3, "coef0": 1},
            {"C": 5.0, "gamma": "scale", "degree": 3, "coef0": 1},
        ]
        self.best_params = self.candidate_params[0]
        self.model = self._fit_model(self.best_params)
        self.refine_results = []

    def refine_model(self) -> None:
        """Tune polynomial SVM parameters, then refit on train + refine."""
        best_score = -1.0
        best_params = None

        for params in self.candidate_params:
            candidate = self._fit_model(params)
            predictions = candidate.predict(self.refine_feat)
            score = balanced_accuracy_score(self.refine_labels, predictions)
            self.refine_results.append((params, score))

            if score > best_score:
                best_score = score
                best_params = params

        self.best_params = best_params
        self.refine_score = best_score

        combined_feat = pd.concat([self.train_feat, self.refine_feat], axis=0)
        combined_labels = pd.concat([self.train_labels, self.refine_labels], axis=0)
        self.model = SVC(
            kernel="poly",
            class_weight="balanced",
            random_state=42,
            **self.best_params,
        )
        self.model.fit(combined_feat, combined_labels)

        print(f"[{self.name}] best params={self.best_params}, refine balanced accuracy={self.refine_score:.3f}")

    def predict(self) -> int:
        """Predict the latest available evaluation row."""
        latest_row = self.evaluate_feat.iloc[[-1]]
        prediction = int(self.model.predict(latest_row)[0])
        signal = 1 if prediction >= 0 else -1
        direction = "UP" if signal == 1 else "DOWN"
        return signal
