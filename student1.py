"""
STUDENT 1 -- implement the three methods below.
This is your file. Do not edit any other student's file.
"""

import pandas as pd
from model import BaseModel


class Student1Model(BaseModel):
    name = "Student 1 Model"

    def train(self) -> None:
        """
        Stage 1 -- Fit your model using self.train_set (80% of data).
        Store what you need as instance attributes, e.g.:
            self.weights = ...
            self.scaler  = ...
        """
        # TODO: implement your training logic
        raise NotImplementedError("Student 1: implement train()")

    def refine_model(self) -> None:
        """
        Stage 2 -- Tune hyperparameters using self.refine_set (10% of data).
        Do not use self.train_set data here.
        """
        # TODO: implement your refinement logic
        raise NotImplementedError("Student 1: implement refine_model()")

    def predict(self) -> int:
        """
        Stage 3 -- Predict next-day direction using self.evaluate_set (10% of data).
        Return 1 (UP) or -1 (DOWN).
        """
        # TODO: implement your prediction logic
        raise NotImplementedError("Student 1: implement predict()")
