"""
STUDENT 2 -- SMA Crossover example.
Replace with your own algorithm.
"""

import pandas as pd
from model import BaseModel


class Student2Model(BaseModel):
    name = "SMA Crossover"

    def train(self) -> None:
        """Compute SMA windows on training data."""
        close = self.train["Close"]
        self.sma_short = float(close.rolling(5).mean().iloc[-1])
        self.sma_long  = float(close.rolling(20).mean().iloc[-1])

    def refine_model(self) -> None:
        """
        Optionally re-evaluate SMA windows on refinement set.
        Could be used to test alternate window lengths.
        """
        close = self.refine["Close"]
        # re-compute on refine data to validate windows hold
        refine_short = float(close.rolling(5).mean().iloc[-1])
        refine_long  = float(close.rolling(20).mean().iloc[-1])
        # if signal flips on refine, fall back to neutral (no trade)
        self.refine_agrees = (refine_short > refine_long) == (self.sma_short > self.sma_long)

    def predict(self) -> int:
        """Return signal based on SMA crossover confirmed by refine stage."""
        if not self.refine_agrees:
            return 1  # neutral fallback -- default long
        return 1 if self.sma_short > self.sma_long else -1
