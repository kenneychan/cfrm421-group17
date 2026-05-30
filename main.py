"""
VIEW -- entry point. Fetches data, runs ensemble, prints results.
Run:  python main.py
"""

from controller import run_ensemble
from control import RandomModel
from student1 import Student1Model
from student2 import Student2Model
from student3 import Student3Model
from student4 import Student4Model
from student5 import Student5Model

TICKER     = "DELL"
START_DATE = "2018-12-28"
END_DATE   = "2025-12-31"

# -- 1. REGISTER MODELS -------------------------------------------------------
all_models = [
    RandomModel(  TICKER, START_DATE, END_DATE),   # Control
    Student1Model(TICKER, START_DATE, END_DATE),   # Student 1
    Student2Model(TICKER, START_DATE, END_DATE),   # Student 2
    Student3Model(TICKER, START_DATE, END_DATE),   # Student 3
    Student4Model(TICKER, START_DATE, END_DATE),   # Student 4
    Student5Model(TICKER, START_DATE, END_DATE),   # Student 5
]

# -- 2. PRINT SPLIT SUMMARY ---------------------------------------------------
m0 = all_models[0]
print(f"\nTicker: {TICKER}  |  {START_DATE} -> {END_DATE}  |  {len(m0.data)} total trading days")
print(f"\n  Split summary (80 / 10 / 10  +  1-month purge gaps)")
print(f"  train    {m0.train_set.index[0].date()} -> {m0.train_set.index[-1].date()}   ({len(m0.train_set)} days)")
print(f"  [purge   1 month]")
print(f"  refine   {m0.refine_set.index[0].date()} -> {m0.refine_set.index[-1].date()}   ({len(m0.refine_set)} days)")
print(f"  [purge   1 month]")
print(f"  evaluate {m0.evaluate_set.index[0].date()} -> {m0.evaluate_set.index[-1].date()}   ({len(m0.evaluate_set)} days)")

# -- 3. RUN ENSEMBLE ----------------------------------------------------------
result = run_ensemble(all_models)

# -- 4. DISPLAY RESULTS -------------------------------------------------------
last_close = float(m0.data["Close"].iloc[-1].item() if hasattr(m0.data["Close"].iloc[-1], 'item') else m0.data["Close"].iloc[-1])
print(f"\nLast close: ${last_close:.2f}  ({m0.data.index[-1].date()})\n")
print("=" * 44)
print(f"  ENSEMBLE PREDICTION -- {TICKER} tomorrow")
print("=" * 44)

for name, signal in result["signals"].items():
    arrow = "up  +1  LONG " if signal == 1 else "dn  -1  SHORT" if signal == -1 else "??  --  ERROR"
    print(f"  {arrow}  <-  {name}")

print("-" * 44)
print(f"  Votes UP:   {result['votes_up']} / {len(all_models)}")
print(f"  Votes DOWN: {result['votes_down']} / {len(all_models)}")
print(f"\n  ENSEMBLE  ->  {result['consensus']}  (signal: {result['ensemble_signal']})")
print("=" * 44)
