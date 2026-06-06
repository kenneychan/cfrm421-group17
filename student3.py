"""
STUDENT 3 -- implement the three methods below.
This is your file. Do not edit any other student's file.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from model import BaseModel


class Student3Model(BaseModel):
    name = "Student 3 Model"

    def _fit_model(self, params: dict) -> LogisticRegression:
        """Fit a LogisticRegression classifier on the training split."""
        model = LogisticRegression(
            random_state=42,
            class_weight="balanced",
            max_iter=1000,
            **params,
        )
        model.fit(self.train_feat, self.train_labels)
        return model

    def train(self) -> None:
        """
        Stage 1 -- Fit your model using self.train_set (80% of data).
        Store what you need as instance attributes, e.g.:
            self.weights = ...
            self.scaler  = ...
        """
        self.candidate_params = [
            {"C": 0.1},
            {"C": 0.5},
            {"C": 1.0},
            {"C": 2.0},
        ]
        self.model = self._fit_model(self.candidate_params[0])
        self.best_params = self.candidate_params[0]
        self.best_refine_score = None

    def refine_model(self) -> None:
        """
        Stage 2 -- Tune hyperparameters using self.refine_set (10% of data).
        Do not use self.train_set data here.
        """
        best_score = -1.0
        best_model = None
        best_params = None

        for params in self.candidate_params:
            candidate = self._fit_model(params)
            predictions = candidate.predict(self.refine_feat)
            score = accuracy_score(self.refine_labels, predictions)

            if score > best_score:
                best_score = score
                best_model = candidate
                best_params = params

        self.model = best_model
        self.best_params = best_params
        self.best_refine_score = best_score

        # refit on the train + refine windows using the best parameters
        combined_feat = pd.concat([self.train_feat, self.refine_feat], axis=0)
        combined_labels = pd.concat([self.train_labels, self.refine_labels], axis=0)
        self.final_model = LogisticRegression(
            random_state=42,
            class_weight="balanced",
            max_iter=1000,
            **self.best_params,
        )
        self.final_model.fit(combined_feat, combined_labels)

    def predict(self) -> int:
        """
        Stage 3 -- Predict next-day direction using self.evaluate_set (10% of data).
        Return 1 (UP) or -1 (DOWN).
        """
        latest_row = self.evaluate_feat.iloc[[-1]]
        prediction = int(self.final_model.predict(latest_row)[0])
        return 1 if prediction >= 0 else -1
