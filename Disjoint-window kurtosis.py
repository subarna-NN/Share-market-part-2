import numpy as np, pandas as pd, yfinance as yf
from scipy.stats import kurtosis

df = yf.download("^GSPC", start="1985-01-01", end="2025-12-31",
                 progress=False, auto_adjust=True)
close  = df["Close"].squeeze()
logret = np.log(close).diff().dropna()
dates  = logret.index                      # REAL trading dates (fixes the drift)
r      = logret.to_numpy().ravel()

events = {"October 1987":"1987-10-19", "2000--2002":"2000-09-01",
          "September 2008":"2008-09-15", "February 2020":"2020-02-20",
          "2022 drawdown":"2022-01-03"}
def xk(a): return kurtosis(a, fisher=True, bias=False) if len(a) > 8 else float("nan")

for name, d in events.items():
    ev = dates.get_indexer([pd.Timestamp(d)], method="nearest")[0]   # real-date match
    pre  = xk(r[max(0, ev-255):ev-5])                                # year-long baseline
    wins = [(1,51),(51,101),(101,201),(201,401)]                     # disjoint, event day excluded
    vals = [xk(r[ev+a:ev+b]) for a, b in wins]
    print(f"    {name:17s} & ${pre:.2f}$ & " + " & ".join(f"${v:.2f}$" for v in vals) + r" \\")
    if "1987" in name:
        print("    % matched event date:", dates[ev].date(),
              "| 1-50 WITH event day:", round(xk(r[ev:ev+50]), 2))
