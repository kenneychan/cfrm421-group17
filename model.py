"""
MODEL -- fetches Yahoo Finance data and defines the BaseModel interface.
Each student subclasses BaseModel and implements three methods:

    train()        -- fit your model on self.train_set (80%)
    refine_model() -- tune hyperparameters on self.refine_set (10%)
    predict()      -- return 1 or -1 using self.evaluate_set (10%)

Data split layout (1-month purge gaps between each set):

  |------ train 80% ------|-- purge --|-- refine 10% --|-- purge --|-- evaluate 10% --|
                          (1 month)                    (1 month)
"""

import yfinance as yf
import pandas as pd
from dateutil.relativedelta import relativedelta


PURGE_MONTHS = 1   # calendar months to drop between splits


class BaseModel:
    """
    Every student inherits from this and implements three methods:
        train()        -- fit on self.train_set
        refine_model() -- tune on self.refine_set
        predict()      -- signal from self.evaluate_set -> 1 or -1

    Constructor args:
        ticker     -- Yahoo Finance ticker symbol   (default: "DELL")
        start_date -- history start date YYYY-MM-DD (default: "2018-12-28")
        end_date   -- history end date   YYYY-MM-DD (default: "2025-12-31")

    Attributes set after __init__:
        self.data      -- full cleaned DataFrame
        self.train_set     -- 80% training set
        self.refine_set    -- 10% refinement / validation set
        self.evaluate_set  -- 10% final holdout set
    """

    name: str = "Unnamed Model"

    def __init__(
        self,
        ticker: str     = "DELL",
        start_date: str = "2018-12-28",
        end_date: str   = "2025-12-31",
    ):
        self.ticker     = ticker
        self.start_date = start_date
        self.end_date   = end_date
        self.data       = self._fetch()
        self.train_set, self.refine_set, self.evaluate_set = self._split()

    def _fetch(self) -> pd.DataFrame:
        """Fetch daily OHLCV data from Yahoo Finance."""
        df = yf.download(
            self.ticker,
            start    = self.start_date,
            end      = self.end_date,
            interval = "1d",
            progress = False,
        )
        # yfinance >= 0.2.x returns MultiIndex columns (field, ticker)
        # flatten to single-level: Open, High, Low, Close, Volume
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        return df

    def _split(self) -> tuple:
        """
        Split self.data into train / refine / evaluate with a 1-month
        calendar purge between each set to prevent leakage.

        Returns:
            (train, refine, evaluate) -- three non-overlapping DataFrames
        """
        idx  = self.data.index
        n    = len(idx)

        i_80 = int(n * 0.80)
        i_90 = int(n * 0.90)

        train_end_date  = idx[i_80 - 1]
        refine_end_date = idx[i_90 - 1]

        purge_after_train  = train_end_date  + relativedelta(months=PURGE_MONTHS)
        purge_after_refine = refine_end_date + relativedelta(months=PURGE_MONTHS)

        train    = self.data.loc[idx[:i_80]]
        refine   = self.data.loc[(idx > purge_after_train) & (idx <= refine_end_date)]
        evaluate = self.data.loc[idx > purge_after_refine]

        return train, refine, evaluate

    # ── Three stages every student must implement ─────────────────────────────

    def train(self) -> None:
        """
        Stage 1 -- Fit your model using self.train_set (80% of data).
        Store anything your model needs as instance attributes (e.g. self.weights).
        Called automatically by the controller before refine_model().
        """
        raise NotImplementedError("Implement train()")

    def refine_model(self) -> None:
        """
        Stage 2 -- Tune hyperparameters using self.refine_set (10% of data).
        self.train_set data must not be used here.
        Called automatically by the controller before predict().
        """
        raise NotImplementedError("Implement refine_model()")

    def predict(self) -> int:
        """
        Stage 3 -- Predict next-day direction using self.evaluate_set (10% of data).
        Called automatically by the controller after train() and refine_model().

        Returns:
            1  -> price goes UP tomorrow
           -1  -> price goes DOWN tomorrow
        """
        raise NotImplementedError("Implement predict()")
