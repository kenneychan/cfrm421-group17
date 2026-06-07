"""
STUDENT 1 -- implement the three methods below.
This is your file. Do not edit any other student's file.
"""
""" Grace Brozinick """

import pandas as pd
from model import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np


class Student1Model(BaseModel):
    name = "Random Forest Classifier"

    def train(self) -> None:
        """
        Training a Random Forest on the training data
        """
        X_train = self.train_feat
        y_train = self.train_labels
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            bootstrap=True, 
            random_state=42, 
            n_jobs=-1)
        self.model.fit(X_train, y_train)
        """
        Stage 1 -- Fit your model using self.train_set (80% of data).
        Store what you need as instance attributes, e.g.:
            self.weights = ...
            self.scaler  = ...
        """


    def refine_model(self) -> None:
        """
        finding best hyperparameters for my model using the refine set
        """
        X_valid = self.refine_feat
        y_valid = self.refine_labels
        best_score = -1
        best_model = None
        for n_trees in [50, 100, 200]:
            for max_depth in [5, 7, 10]:
                model = RandomForestClassifier(
                    n_estimators=n_trees, 
                    max_depth=max_depth, 
                    bootstrap=True, 
                    random_state=42, 
                    n_jobs=-1)
                model.fit(self.train_feat, self.train_labels)
                score = accuracy_score(y_valid, model.predict(X_valid))
                if score > best_score:
                    best_score = score
                    best_model = model
        self.model = best_model
        self.best_score = best_score
        """
        Stage 2 -- Tune hyperparameters using self.refine_set (10% of data).
        Do not use self.train_set data here.
        """

    def predict(self) -> int:
        """
        predicting the next day direction using the evaluate set and the best model found in refine_model
        """
        most_recent = self.evaluate_feat.iloc[[-1]]
        predict_value = int(self.model.predict(most_recent)[0])
        
        signal = 1 if predict_value >= 0 else -1
        return signal
        """
        Stage 3 -- Predict next-day direction using self.evaluate_set (10% of data).
        Return 1 (UP) or -1 (DOWN).
        
        """
