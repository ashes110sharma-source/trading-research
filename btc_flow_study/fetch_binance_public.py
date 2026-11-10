"""
Download BTCUSDT perpetual-futures data from Binance's public archive (data.binance.vision) and write three tidy files:
  data/bars_5m.csv.gz   datetime, open, high, low, close, volume, taker_buy_volume, trades      (from 1-min klines)
  data/funding.csv      datetime, funding_rate
  data/metrics_5m.csv.gz datetime, open_interest, open_interest_value, top_ls_accounts, top_ls_positions, retail_ls_accounts, taker_ls_vol
Only the static archive is used. The live futures API is not called.
Usage: python fetch_binance_public.py --start 2020-01 --out data
"""
import argparse, io, os, sys, time, zipfile, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um"
SYM = "BTCUSDT"
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def get(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-fetch"}), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            if k == tries - 1: return ("ERR", f"HTTP {e.code}")
        except Exception as e:
            if k == tries - 1: return ("ERR", str(e)[:80])
        time.sleep(2 * (k + 1))


def read_zip_csv(blob, names=None):
    """Binance archive CSVs sometimes have a header row and sometimes do not."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    first = raw[:200].decode("utf-8", "ignore").split("\n")[0]
    has_header = any(ch.isalpha() for ch in first)
    if has_header: return pd.read_csv(io.BytesIO(raw))
    return pd.read_csv(io.BytesIO(raw), header=None, names=names)


def to_dt(s):
    """Epoch in s / ms / us, or an ISO string."""
    if pd.api.types.is_numeric_dtype(s):
        v = float(s.iloc[0]); unit = "s" if v < 1e11 else "ms" if v < 1e14 else "us"
        return pd.to_datetime(s, unit=unit)
    return pd.to_datetime(s)


def klines_to_5m(df):
    df = df.rename(columns={"count": "trades"})
    df["datetime"] = to_dt(df["open_time"])
    for c in ("open", "high", "low", "close", "volume", "taker_buy_volume", "trades"): df[c] = pd.to_numeric(df[c])
    g = df.groupby(df["datetime"].dt.floor("5min"))
    return g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
                 taker_buy_volume=("taker_buy_volume", "sum"), trades=("trades", "sum"))


def tidy_metrics(df):
    df["datetime"] = to_dt(df["create_time"]).dt.floor("5min")
    out = pd.DataFrame({"datetime": df["datetime"], "open_interest": pd.to_numeric(df["sum_open_interest"]),
                        "open_interest_value": pd.to_numeric(df["sum_open_interest_value"]),
                        "top_ls_accounts": pd.to_numeric(df["count_toptrader_long_short_ratio"]),
                        "top_ls_positions": pd.to_numeric(df["sum_toptrader_long_short_ratio"]),
                        "retail_ls_accounts": pd.to_numeric(df["count_long_short_ratio"]),
                        "taker_ls_vol": pd.to_numeric(df["sum_taker_long_short_vol_ratio"])})
    return out.drop_duplicates("datetime").set_index("datetime")


def tidy_funding(df):
    return pd.DataFrame({"datetime": to_dt(df["calc_time"]), "funding_rate": pd.to_numeric(df["last_funding_rate"])}).set_index("datetime")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--start", default="2020-01"); ap.add_argument("--out", default="data"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); rep = []
    now = pd.Timestamp.now(tz="UTC").tz_localize(None); months = pd.period_range(a.start, now.to_period("M"), freq="M")
    full_months = [m for m in months if m < now.to_period("M")]
    probe = get(f"{BASE}/monthly/klines/{SYM}/1m/{SYM}-1m-{full_months[-1]}.zip")
    if probe is None or isinstance(probe, tuple):
        msg = f"ARCHIVE NOT REACHABLE from this machine ({probe}). Nothing downloaded."; print(msg); open(f"{a.out}/fetch_report.txt", "w").write(msg + "\n"); sys.exit(2)

    def pull(urls, parse, names=None, workers=8):
        got, miss, err = [], 0, []
        def one(u):
            b = get(u)
            if b is None: return None
            if isinstance(b, tuple): return ("ERR", u, b[1])
            return parse(read_zip_csv(b, names))
        with ThreadPoolExecutor(workers) as ex:
            for r in ex.map(one, urls):
                if r is None: miss += 1
                elif isinstance(r, tuple): err.append(r[1:])
                else: got.append(r)
        return got, miss, err

    cur_days = pd.date_range(now.to_period("M").start_time, now.normalize() - pd.Timedelta(days=1), freq="D")
    k_urls = [f"{BASE}/monthly/klines/{SYM}/1m/{SYM}-1m-{m}.zip" for m in full_months] + \
             [f"{BASE}/daily/klines/{SYM}/1m/{SYM}-1m-{d.date()}.zip" for d in cur_days]
    got, miss, err = pull(k_urls, klines_to_5m, KCOLS)
    bars = pd.concat(got).sort_index(); bars = bars[~bars.index.duplicated()]
    bars.to_csv(f"{a.out}/bars_5m.csv.gz"); rep.append(f"klines: {len(got)} files ok, {miss} missing, {len(err)} errors -> {len(bars):,} five-minute bars, {bars.index[0]} to {bars.index[-1]}")

    f_urls = [f"{BASE}/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{m}.zip" for m in months]
    got, miss, err = pull(f_urls, tidy_funding)
    if got:
        fu = pd.concat(got).sort_index(); fu.to_csv(f"{a.out}/funding.csv"); rep.append(f"funding: {len(got)} files ok, {miss} missing, {len(err)} errors -> {len(fu):,} rows, {fu.index[0]} to {fu.index[-1]}")
    else: rep.append(f"funding: NOTHING FOUND ({miss} missing, {len(err)} errors)")

    days = pd.date_range(pd.Period(a.start, "M").start_time, now.normalize() - pd.Timedelta(days=1), freq="D")
    m_urls = [f"{BASE}/daily/metrics/{SYM}/{SYM}-metrics-{d.date()}.zip" for d in days]
    got, miss, err = pull(m_urls, tidy_metrics, workers=12)
    if got:
        me = pd.concat(got).sort_index(); me = me[~me.index.duplicated()]; me.to_csv(f"{a.out}/metrics_5m.csv.gz")
        rep.append(f"metrics (open interest, long/short): {len(got)} daily files ok, {miss} missing, {len(err)} errors -> {len(me):,} rows, {me.index[0]} to {me.index[-1]}")
    else: rep.append(f"metrics: NOTHING FOUND ({miss} missing, {len(err)} errors)")
    open(f"{a.out}/fetch_report.txt", "w").write("\n".join(rep) + "\n"); print("\n".join(rep))


if __name__ == "__main__":
    main()
