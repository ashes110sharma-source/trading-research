"""Offline checks with SYNTHETIC data (no internet): parsers, the no-leakage property, and the stage guards.
Usage: python test_offline.py"""
import io, os, sys, json, zipfile, shutil, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import fetch_binance_public as fb, study_flow as sf

def zipped(text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z: z.writestr("x.csv", text)
    return buf.getvalue()

# 1. parsers: klines with and without a header row, ms and us timestamps
rows = "\n".join(f"{1704067200000 + i*60000},100,{101+i},99,{100+i},2,{1704067259999 + i*60000},200,10,1.5,150,0" for i in range(10))
a = fb.klines_to_5m(fb.read_zip_csv(zipped(rows), fb.KCOLS))
hdr = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n"
b = fb.klines_to_5m(fb.read_zip_csv(zipped(hdr + rows), fb.KCOLS))
rows_us = "\n".join(f"{(1704067200000 + i*60000)*1000},100,{101+i},99,{100+i},2,0,200,10,1.5,150,0" for i in range(10))
c = fb.klines_to_5m(fb.read_zip_csv(zipped(rows_us), fb.KCOLS))
assert len(a) == 2 and a.equals(b) and a.equals(c) and a.index[0] == pd.Timestamp("2024-01-01") and a.high.iloc[0] == 105 and a.volume.iloc[0] == 10 and a.taker_buy_volume.iloc[1] == 7.5
m = fb.tidy_metrics(fb.read_zip_csv(zipped("create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n2024-01-01 00:05:00,BTCUSDT,80000,3.4e9,1.5,1.1,2.0,0.9\n")))
assert m.open_interest.iloc[0] == 80000 and m.retail_ls_accounts.iloc[0] == 2.0
f = fb.tidy_funding(fb.read_zip_csv(zipped("calc_time,funding_interval_hours,last_funding_rate\n1704067200000,8,0.0001\n")))
assert f.funding_rate.iloc[0] == 0.0001 and f.index[0] == pd.Timestamp("2024-01-01")
print("PASS parsers: header / no header, ms / us timestamps, metrics and funding files")

# 2. synthetic dataset in the fetcher's output format
tmp = tempfile.mkdtemp(); data = f"{tmp}/data"; res = f"{tmp}/results"; os.makedirs(data)
rng = np.random.default_rng(1); idx = pd.date_range("2024-03-01", "2025-04-01", freq="5min", inclusive="left"); n = len(idx)
px = 60000 * np.exp(np.cumsum(rng.normal(0, 0.0012, n))); op = np.r_[px[0], px[:-1]]
vol = rng.gamma(2, 50, n)
pd.DataFrame({"open": op, "high": np.maximum(op, px) * (1 + abs(rng.normal(0, 0.0004, n))), "low": np.minimum(op, px) * (1 - abs(rng.normal(0, 0.0004, n))),
              "close": px, "volume": vol, "taker_buy_volume": vol * rng.beta(20, 20, n), "trades": 100.0}, index=idx).rename_axis("datetime").to_csv(f"{data}/bars_5m.csv.gz")
oi = 80000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
pd.DataFrame({"open_interest": oi, "open_interest_value": oi * px, "top_ls_accounts": 1.2, "top_ls_positions": 1.1,
              "retail_ls_accounts": 1.5 + np.cumsum(rng.normal(0, 0.01, n)), "taker_ls_vol": 1.0}, index=idx).rename_axis("datetime").to_csv(f"{data}/metrics_5m.csv.gz")
fi = pd.date_range("2024-03-01", "2025-04-01", freq="8h", inclusive="left")
pd.DataFrame({"funding_rate": rng.normal(0.0001, 0.0002, len(fi))}, index=fi).rename_axis("datetime").to_csv(f"{data}/funding.csv")

# 3. no leakage: features and signals built on truncated data must equal the full-data ones
bars, fund, met, _ = sf.load(data); full = sf.features(bars, fund, met); Sf = sf.setups(full); ok = True
for cut in rng.integers(30000, n - 10, 6):
    cut = int(cut) // 3 * 3; t = bars.index[cut]
    part = sf.features(bars.iloc[:cut], fund[fund.index < t], met[met.index < t])
    A = full.loc[part.index]; ok &= bool(np.allclose(A.values.astype(float), part.values.astype(float), equal_nan=True))
    Sp = sf.setups(part)
    for k in Sf:
        x = Sf[k][Sf[k].index <= part.index[-1]]; ok &= len(x) == len(Sp[k]) and bool((x.values == Sp[k].values).all())
assert ok; print(f"PASS no leakage: {full.shape[1]} feature columns and {len(Sf)} setups identical on 6 truncations")

# 4. stage guards
here = os.path.dirname(os.path.abspath(__file__))
r = subprocess.run([sys.executable, f"{here}/study_flow.py", "--data", data, "--out", res, "--stage", "dev"], capture_output=True, text=True); assert r.returncode == 0, r.stderr[-800:]
G = pd.read_csv(f"{res}/dev_grid.csv"); sel = json.load(open(f"{res}/selected.json"))
assert len(G) == 24 and os.path.exists(f"{res}/RESULTS_dev.md")
tr_max = pd.read_csv(f"{data}/bars_5m.csv.gz", parse_dates=["datetime"]).datetime.max()
print(f"PASS dev stage ran on synthetic data: {len(G)} cells, {int(G.trades.sum())} simulated trades, qualified = {sel['qualified']}")
print("     (synthetic prices are random, so 'qualified' should be False):", "OK" if not sel["qualified"] else "UNEXPECTED")
r2 = subprocess.run([sys.executable, f"{here}/study_flow.py", "--data", data, "--out", res, "--stage", "test"], capture_output=True, text=True)
assert (r2.returncode != 0) == (not sel["qualified"]); print("PASS test stage refuses to run unless the dev stage qualified a setup")
print("\nDev table on synthetic data (random prices, for format only):"); print(G.sort_values("net_t", ascending=False)[["setup", "hold_h", "trades", "per_day", "gross_avg_pct", "net_avg_pct", "net_t"]].head(8).to_string(index=False))
shutil.rmtree(tmp)
