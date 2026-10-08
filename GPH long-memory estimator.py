import numpy as np, yfinance as yf
df = yf.download("^GSPC", start="1985-01-01", end="2025-12-31", progress=False, auto_adjust=True)
r  = np.log(df["Close"].squeeze()).diff().dropna().to_numpy().ravel()
x  = np.abs(r - r.mean())                      # volatility proxy: |returns|
n  = len(x)
I  = (np.abs(np.fft.rfft(x - x.mean()))**2) / (2*np.pi*n)   # periodogram
w  = 2*np.pi*np.arange(len(I))/n
m  = int(n**0.5)                               # GPH bandwidth
j  = np.arange(1, m+1)
slope, _ = np.polyfit(np.log(4*np.sin(w[j]/2)**2), np.log(I[j]), 1)
d  = -slope; se = np.pi/np.sqrt(24*m)
print(f"GPH:  d = {d:.3f}   SE {se:.3f}   95% CI [{d-1.96*se:.3f}, {d+1.96*se:.3f}]   (m={m})")
