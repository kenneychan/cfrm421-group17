"""
MODEL -- fetches Yahoo Finance data and defines the BaseModel interface.
Each student subclasses BaseModel and implements three methods:

    train()        -- fit your model on self.train_set (80%)
    refine_model() -- tune hyperparameters on self.refine_set (10%)
    predict()      -- return 1 or -1 using self.evaluate_set (10%)

Data split layout (1-month purge gaps between each set):

  |------ train 80% ------|-- purge --|-- refine 10% --|-- purge --|-- evaluate 10% --|
                          (1 month)                    (1 month)

Normalized feature sets are built automatically after the split:

    self.train_feat      -- normalized features for training   (80%)
    self.refine_feat     -- normalized features for refinement (10%)
    self.evaluate_feat   -- normalized features for evaluation (10%)

The scaler is fit ONLY on train_feat to prevent data leakage into
refine and evaluate sets. Raw sets (self.train_set, etc.) remain
available if you prefer to engineer your own features from scratch.
"""

import numpy as np
import yfinance as yf
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler


PURGE_MONTHS = 1   # calendar months to drop between splits


class BaseModel:
    """
    Every student inherits from this and implements three methods:
        train()        -- fit on self.train_set / self.train_feat
        refine_model() -- tune on self.refine_set / self.refine_feat
        predict()      -- signal from self.evaluate_set / self.evaluate_feat -> 1 or -1

    Constructor args:
        ticker     -- Yahoo Finance ticker symbol   (default: "DELL")
        start_date -- history start date YYYY-MM-DD (default: "2018-12-28")
        end_date   -- history end date   YYYY-MM-DD (default: "2025-12-31")

    Raw attributes set after __init__:
        self.data          -- full cleaned DataFrame (raw OHLCV)
        self.train_set     -- 80% training set       (raw OHLCV)
        self.refine_set    -- 10% refinement set     (raw OHLCV)
        self.evaluate_set  -- 10% evaluation set     (raw OHLCV)

    Normalized attributes set after __init__:
        self.train_feat    -- normalized features for training   (80%)
        self.refine_feat   -- normalized features for refinement (10%)
        self.evaluate_feat -- normalized features for evaluation (10%)
        self.scaler        -- fitted StandardScaler instance

    Normalized feature columns:
        log_return      -- log(Close_t / Close_{t-1})          stationary price signal
        range_pct       -- (High - Low) / Close                intraday volatility, scale-free
        close_open_pct  -- (Close - Open) / Open               intraday direction, scale-free
        log_volume      -- log(Volume + 1)                     compresses heavy-tailed volume
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
        self._normalize_splits()

    # ── Data fetching and cleaning ────────────────────────────────────────────

    def _fetch(self) -> pd.DataFrame:
        """Fetch daily OHLCV data from Yahoo Finance and run _clean()."""
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
        df = self._clean(df)
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows that represent definitively bad data.

        These four checks target data errors -- not market events.
        Extreme returns (crashes, earnings gaps) are intentionally kept:
        a real trader lives through them, so the model should too.

        Checks applied (in order):
            1. Duplicate timestamps  -- keep first occurrence, drop the rest
            2. Non-positive prices   -- Open/High/Low/Close must be > 0
            3. Zero volume           -- indicates a halted or phantom trading day
            4. OHLC sanity           -- High must be >= all of O/L/C
                                        Low  must be <= all of O/H/C

        Prints a summary line if any rows were dropped, so problems are visible.
        Returns a cleaned copy -- does NOT modify the input in place.
        """
        n_start = len(df)
        price_cols = ["Open", "High", "Low", "Close"]

        # 1. Duplicate timestamps
        df = df[~df.index.duplicated(keep="first")]

        # 2. Non-positive prices (impossible in real markets)
        df = df[(df[price_cols] > 0).all(axis=1)]

        # 3. Zero volume (halted day or bad data feed)
        df = df[df["Volume"] > 0]

        # 4. OHLC sanity violations (data errors, not market events)
        ohlc_valid = (
            (df["High"] >= df[["Open", "Low",  "Close"]].max(axis=1)) &
            (df["Low"]  <= df[["Open", "High", "Close"]].min(axis=1))
        )
        df = df[ohlc_valid]

        n_dropped = n_start - len(df)
        if n_dropped > 0:
            print(f"[{self.__class__.__name__}] _clean: removed {n_dropped} bad rows "
                  f"({n_start} -> {len(df)})")

        return df

    # ── Train / refine / evaluate split ──────────────────────────────────────

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

    # ── Normalization ─────────────────────────────────────────────────────────

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute normalized features from a raw OHLCV DataFrame.
        All four features are scale-free or log-compressed so that a
        StandardScaler can finish the job without price-level distortion.

        Returns a new DataFrame -- does NOT modify the input.
        """
        out = pd.DataFrame(index=df.index)

        # Log return: log(Close_t) - log(Close_{t-1})
        # Equivalent to log(Close_t / Close_{t-1}) but uses idiomatic pandas .diff().
        # Converts non-stationary prices into a stationary return series.
        # Approximately normally distributed and comparable across time.
        out["log_return"]     = np.log(df["Close"]).diff()

        # Intraday range as a fraction of close price.
        # Captures daily volatility without depending on the absolute price level.
        out["range_pct"]      = (df["High"] - df["Low"]) / df["Close"]

        # Close vs open as a fraction of open price.
        # Captures intraday directionality independent of price level.
        out["close_open_pct"] = (df["Close"] - df["Open"]) / df["Open"]

        # Log volume: log(Volume + 1)
        # Volume can be in the millions and is heavy-tailed; log compression
        # brings it onto a comparable scale before standardization.
        out["log_volume"]     = np.log(df["Volume"] + 1)

        return out.dropna()

    def _normalize_splits(self) -> None:
        """
        Build normalized feature sets and label series for all three splits.

        Critical: StandardScaler is fit ONLY on training data.
        Refine and evaluate are transformed using the training statistics.
        Fitting the scaler on all data would leak future information into
        the training set and invalidate the evaluation.

        Attaches to self:
            self.train_feat      pd.DataFrame  shape (n_train,    n_features)
            self.refine_feat     pd.DataFrame  shape (n_refine,   n_features)
            self.evaluate_feat   pd.DataFrame  shape (n_evaluate, n_features)
            self.scaler          StandardScaler fitted on train_feat

            self.train_labels    pd.Series     shape (n_train - 1,)   int  1 or -1
            self.refine_labels   pd.Series     shape (n_refine - 1,)  int  1 or -1
            self.evaluate_labels pd.Series     shape (n_evaluate - 1,) int 1 or -1

        Features and labels share the same index so they align with .loc[].
        The last row of each split has no label (next-day close is unknown)
        and is dropped from both feat and labels automatically.
        """
        train_raw    = self._add_features(self.train_set)
        refine_raw   = self._add_features(self.refine_set)
        evaluate_raw = self._add_features(self.evaluate_set)

        scaler = StandardScaler()

        # fit_transform on train only
        self.train_feat = pd.DataFrame(
            scaler.fit_transform(train_raw),
            index   = train_raw.index,
            columns = train_raw.columns,
        )

        # transform (never fit) on refine and evaluate
        self.refine_feat = pd.DataFrame(
            scaler.transform(refine_raw),
            index   = refine_raw.index,
            columns = refine_raw.columns,
        )
        self.evaluate_feat = pd.DataFrame(
            scaler.transform(evaluate_raw),
            index   = evaluate_raw.index,
            columns = evaluate_raw.columns,
        )

        # expose scaler in case students need inverse_transform
        self.scaler = scaler

        # build labels and trim feat to matching rows
        self.train_labels,    self.train_feat    = self._make_labels(self.train_set,    self.train_feat)
        self.refine_labels,   self.refine_feat   = self._make_labels(self.refine_set,   self.refine_feat)
        self.evaluate_labels, self.evaluate_feat = self._make_labels(self.evaluate_set, self.evaluate_feat)

    def _make_labels(
        self,
        raw: pd.DataFrame,
        feat: pd.DataFrame,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Compute next-day direction labels for a split and align them with
        the corresponding feature rows.

        Label definition:
            +1  next-day close > today's close  (UP)
            -1  next-day close < today's close  (DOWN)
             0  unchanged -- treated as +1 (flat days are extremely rare
                and usually reflect a data issue rather than a real event)

        The last row of every split has no observable next-day close within
        that split (the next day falls in the purge gap or beyond), so it is
        dropped from both the label series and the feature DataFrame.

        Args:
            raw   raw OHLCV DataFrame for the split (used for Close prices)
            feat  normalized feature DataFrame for the same split

        Returns:
            (labels, feat_trimmed)
                labels       pd.Series  dtype int, index = feat_trimmed.index
                feat_trimmed pd.DataFrame with last row removed
        """
        # pct_change() is the idiomatic pandas daily-return calculation.
        # shift(-1) looks one row forward so today's label = tomorrow's return.
        next_day_return = raw["Close"].pct_change().shift(-1)

        # Convert to direction signal: +1, -1, or 0
        labels = np.sign(next_day_return).dropna().astype(int)
        labels = labels.replace(0, 1)

        # Align feat to the labeled rows (drops the unlabeled final row)
        shared_idx   = feat.index.intersection(labels.index)
        feat_trimmed = feat.loc[shared_idx]
        labels       = labels.loc[shared_idx]

        return labels, feat_trimmed

    # ── Three stages every student must implement ─────────────────────────────

    def train(self) -> None:
        """
        Stage 1 -- Fit your model using self.train_set (80% of data).

        Use self.train_feat for pre-normalized features, or self.train_set
        for raw OHLCV if you prefer to engineer your own features.

        Store anything your model needs as instance attributes (e.g. self.weights).
        Called automatically by the controller before refine_model().
        """
        raise NotImplementedError("Implement train()")

    def refine_model(self) -> None:
        """
        Stage 2 -- Tune hyperparameters using self.refine_set (10% of data).

        Use self.refine_feat for pre-normalized features, or self.refine_set
        for raw OHLCV. self.train_set data must not be used here.
        Called automatically by the controller before predict().
        """
        raise NotImplementedError("Implement refine_model()")

    def predict(self) -> int:
        """
        Stage 3 -- Predict next-day direction using self.evaluate_set (10% of data).

        Use self.evaluate_feat for pre-normalized features, or self.evaluate_set
        for raw OHLCV. Called automatically by the controller after train()
        and refine_model().

        Returns:
            1  -> price goes UP tomorrow
           -1  -> price goes DOWN tomorrow
        """
        raise NotImplementedError("Implement predict()")
