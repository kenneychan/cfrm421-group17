"""
CONTROL -- random baseline model.
This file is not edited by students.
"""

import random
import pandas as pd
from model import BaseModel


class RandomModel(BaseModel):
    """Baseline control: random coin flip. Every student model should beat this."""
    name = "Random Baseline (control)"

    def train(self) -> None:
        pass  # nothing to fit for a random model

    def refine_model(self) -> None:
        pass  # nothing to tune for a random model

    def predict(self) -> int:
        return random.choice([1, -1])
