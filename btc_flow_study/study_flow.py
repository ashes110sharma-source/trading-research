"""
BTC FLOW STUDY  -  funding, open interest, positioning and taker flow on BTCUSDT perpetuals.
Rules, windows and the pass mark are fixed in PREREGISTRATION.md. Do not change them after seeing results.

Usage:  python study_flow.py --data data --out results --stage dev
        python study_flow.py --data data --out results --stage test      (only runs if the dev stage qualified a setup)
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

TEST_START = pd.Timestamp("2025-01-01")
COST_PCT, STOP_SLIP_PCT = 0.10, 0.02          # round-trip cost; extra slippage charged on stop exits
STOP_ADR, TARGET_R, MAX_PER_DAY = 0.25, 2.0, 2
MIN_TRADES, PASS_T, PASS_QUARTERS = 150, 2.5, 0.60
HOLDS = (240, 480)


def load(data_dir):
    bars = pd.read_csv(f"{data_dir}/bars_5m.csv.gz", parse_dates=["datetime"]).set_index("datetime").sort_index()
    grid = pd.date_range(bars.index[0], bars.index[-1], freq="5min"); gaps = len(grid) - len(bars)
    bars = bars.reindex(grid); c = bars["close"].ffill()
    for col in ("open", "high", "low", "close"): bars[col] = bars[col].fillna(c)
    for col in ("volume", "taker_buy_volume", "trades"): bars[col] = bars[col].fillna(0.0)
    fund = pd.read_csv(f"{data_dir}/funding.csv", parse_dates=["datetime"]).set_index("datetime").sort_index() if os.path.exists(f"{data_dir}/funding.csv") else None
    met = pd.read_csv(f"{data_dir}/metrics_5m.csv.gz", parse_dates=["datetime"]).set_index("datetime").sort_index() if os.path.exists(f"{data_dir}/metrics_5m.csv.gz") else None
    return bars, fund, met, gaps


def features(bars, fund, met):
    """One row per 15-min candle, stamped with the candle's START. Everything is known at that candle's CLOSE."""
    b = bars.groupby(bars.index.floor("15min")).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                                                    volume=("volume", "sum"), tbv=("taker_buy_volume", "sum"))
    day = b.index.normalize(); d = b.groupby(day).agg(high=("high", "max"), low=("low", "min"))
    b["adr"] = (d.high - d.low).shift(1).rolling(20, min_periods=20).median().reindex(day).values
    b["move30"] = (b.close - b.close.shift(2)) / b.adr
    b["hi4h"] = b.high.rolling(16).max().shift(1); b["lo4h"] = b.low.rolling(16).min().shift(1)
    ratio = b.tbv.rolling(2).sum() / b.volume.rolling(2).sum().replace(0, np.nan)
    b["taker_pct"] = ratio.rolling(672, min_periods=336).rank(pct=True)                     # vs the last 7 days
    close_time = b.index + pd.Timedelta(minutes=15)
    if met is not None:
        # a metrics snapshot stamped T is treated as usable only from T + 5 min
        m = met.copy(); m.index = m.index + pd.Timedelta(minutes=5)
        m = m.reindex(m.index.union(close_time)).sort_index().ffill(limit=6).reindex(close_time); m.index = b.index
        b["oi"] = m.open_interest; b["d_oi30"] = b.oi / b.oi.shift(2) - 1; b["d_oi60"] = b.oi / b.oi.shift(4) - 1
        b["retail_pct"] = m.retail_ls_accounts.rolling(2880, min_periods=1440).rank(pct=True)   # vs the last 30 days
    if fund is not None:
        f = fund.copy(); f["fund_pct"] = f.funding_rate.rolling(90, min_periods=45).rank(pct=True)   # vs the last 90 funding events
        # the 15-min candle that closes at or just after the funding time carries the event
        pos = np.searchsorted(close_time.values, f.index.values, side="left"); ok = pos < len(b)
        b["fund_pct"] = np.nan; b.iloc[pos[ok], b.columns.get_loc("fund_pct")] = f.fund_pct.values[ok]
    return b


def first_per_day(s):
    return s[~pd.DataFrame({"d": s.index.normalize(), "dir": s.values}).duplicated().values]


def setups(b):
    """name -> Series of +1 (long) / -1 (short) at signal candles."""
    out = {}; z = pd.Series(0.0, index=b.index)
    def mk(longs, shorts, first=False):
        s = z.copy(); s[longs.fillna(False)] = 1.0; s[shorts.fillna(False)] = -1.0; s[(longs & shorts).fillna(False)] = 0.0
        s = s[s != 0]; return first_per_day(s) if first else s
    if "d_oi30" in b:
        for a, q in ((0.25, 0.005), (0.35, 0.010)):
            out[f"H1 liquidation flush, fade (move {a} ADR, OI -{q*100:g}%)"] = mk((b.move30 <= -a) & (b.d_oi30 <= -q), (b.move30 >= a) & (b.d_oi30 <= -q))
        for q in (0.005, 0.010):
            up, dn = b.close > b.hi4h, b.close < b.lo4h
            out[f"H3 breakout with rising OI, follow (OI +{q*100:g}%)"] = mk(up & (b.d_oi60 >= q), dn & (b.d_oi60 >= q), True)
            out[f"H3b breakout with falling OI, fade (OI -{q*100:g}%)"] = mk(dn & (b.d_oi60 <= -q), up & (b.d_oi60 <= -q), True)
        for p in (0.90, 0.97):
            out[f"H5 retail crowd extreme, fade (top/bottom {round((1-p)*100)}%)"] = mk(b.retail_pct <= 1 - p, b.retail_pct >= p, True)
    if "fund_pct" in b:
        for p in (0.90, 0.95):
            out[f"H2 funding extreme, fade (top/bottom {round((1-p)*100)}%)"] = mk(b.fund_pct <= 1 - p, b.fund_pct >= p)
    for p in (0.95, 0.98):
        out[f"H4 taker flow absorbed, fade (top/bottom {round((1-p)*100)}%)"] = mk((b.taker_pct <= 1 - p) & (b.move30 >= 0), (b.taker_pct >= p) & (b.move30 <= 0))
    return out


def simulate(bars, b, sig, hold_min, t0, t1):
    sig = sig[(sig.index >= t0) & (sig.index < t1)]
    base = bars.index[0]; n = len(bars); o, h, l, c = bars.open.values, bars.high.values, bars.low.values, bars.close.values
    idx = ((sig.index - base) / pd.Timedelta(minutes=5)).astype(int).values + 3            # first 5-min bar after the signal candle
    adr = b.adr.reindex(sig.index).values; sd = sig.values; hold = hold_min // 5
    out = []; p = 0; m = len(sig); cur_day = -1; cnt = 0
    while p < m:
        i = idx[p]
        if i + hold >= n: break
        if np.isnan(adr[p]): p += 1; continue
        dn = i // 288
        if dn != cur_day: cur_day, cnt = dn, 0
        if cnt >= MAX_PER_DAY: p = int(np.searchsorted(idx, (dn + 1) * 288)); continue
        sg = sd[p]; e = o[i]; risk = STOP_ADR * adr[p]; stop = e - sg * risk; tgt = e + sg * TARGET_R * risk; j = i + hold
        adv = (l[i:j] <= stop) if sg > 0 else (h[i:j] >= stop); fav = (h[i:j] >= tgt) if sg > 0 else (l[i:j] <= tgt)
        ks = int(np.argmax(adv)) if adv.any() else 10**9; kt = int(np.argmax(fav)) if fav.any() else 10**9
        if ks <= kt and ks < 10**9: k, x, why, extra = ks, stop, "stop", STOP_SLIP_PCT
        elif kt < 10**9: k, x, why, extra = kt, tgt, "target", 0.0
        else: k, x, why, extra = hold - 1, c[j - 1], "time", 0.0
        gross = sg * (x - e) / e * 100 - extra
        out.append((sig.index[p], bars.index[i], int(sg), e, bars.index[i + k], x, why, gross, gross - COST_PCT))
        cnt += 1; p = int(np.searchsorted(idx, i + k + 1))
    return pd.DataFrame(out, columns=["signal_time", "entry_time", "dir", "entry", "exit_time", "exit", "exit_reason", "gross_pct", "net_pct"])


def summary(tr):
    if len(tr) < 30: return {"trades": len(tr)}
    def t(col):
        p = pd.Series(tr[col].values, index=pd.DatetimeIndex(tr.entry_time)); day = p.index.normalize()
        gs, gn = p.groupby(day).sum(), p.groupby(day).size(); N, D, mu = len(p), len(gs), p.mean()
        return p, mu, mu / (np.sqrt(((gs - gn * mu) ** 2).sum() * D / max(D - 1, 1)) / N)
    p, mu, tn = t("net_pct"); _, mg, tg = t("gross_pct"); q = p.groupby(p.index.to_period("Q")).sum(); eq = p.cumsum()
    span = max((p.index[-1] - p.index[0]).days, 1)
    return {"trades": len(p), "per_day": round(len(p) / span, 2), "win_pct": round((p > 0).mean() * 100, 1), "gross_avg_pct": round(mg, 4), "gross_t": round(tg, 1),
            "net_avg_pct": round(mu, 4), "net_t": round(tn, 1), "net_sum_pct": round(p.sum(), 1), "max_dd_pct": round((eq.cummax() - eq).max(), 1),
            "quarters_pos": int((q > 0).sum()), "quarters": len(q)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results"); ap.add_argument("--stage", default="dev", choices=["dev", "test"])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    bars, fund, met, gaps = load(a.data)
    if a.stage == "dev":                                   # the test window is cut off before anything is computed
        bars = bars[bars.index < TEST_START]
        fund = None if fund is None else fund[fund.index < TEST_START]; met = None if met is None else met[met.index < TEST_START]
    b = features(bars, fund, met); S = setups(b); lines = []
    start = bars.index[0] + pd.Timedelta(days=45)
    lines.append(f"Bars {bars.index[0]} to {bars.index[-1]} ({gaps} missing 5-min bars filled). Funding: {'yes' if fund is not None else 'NO'}. Open interest / long-short: "
                 + (f"yes, from {met.index[0].date()}" if met is not None else "NO"))
    if a.stage == "dev":
        rows = []
        for name, sig in S.items():
            for hold in HOLDS:
                rows.append({"setup": name, "hold_h": hold // 60, **summary(simulate(bars, b, sig, hold, start, TEST_START))})
        G = pd.DataFrame(rows); G.to_csv(f"{a.out}/dev_grid.csv", index=False)
        ok = G[G.trades >= MIN_TRADES].sort_values("net_t", ascending=False) if "net_t" in G else G.iloc[:0]
        lines.append(f"DEV window {start.date()} to {TEST_START.date()}: {len(G)} cells, {int((G.get('net_avg_pct', pd.Series(dtype=float)) > 0).sum())} positive after costs.")
        sel = {"qualified": False}
        if len(ok):
            best = ok.iloc[0]; passed = bool(best.net_t >= PASS_T and best.quarters_pos / best.quarters >= PASS_QUARTERS)
            sel = {"qualified": passed, "setup": best.setup, "hold_h": int(best.hold_h), "net_t": float(best.net_t), "net_avg_pct": float(best.net_avg_pct), "trades": int(best.trades)}
            lines.append(f"Best cell: {best.setup}, hold {best.hold_h}h: {best.trades} trades, net {best.net_avg_pct}% per trade, net t-stat {best.net_t}, "
                         f"{best.quarters_pos}/{best.quarters} quarters positive -> " + ("QUALIFIES for the one-time test" if passed else "does NOT qualify. No test will be run."))
        else: lines.append(f"No cell reached {MIN_TRADES} trades.")
        json.dump(sel, open(f"{a.out}/selected.json", "w"), indent=1)
        table = G.sort_values("net_t", ascending=False).to_string(index=False) if "net_t" in G else G.to_string(index=False)
        open(f"{a.out}/RESULTS_dev.md", "w").write("# BTC flow study - DEV stage\n\n" + "\n\n".join(lines) + "\n\n```\n" + table + "\n```\n")
    else:
        sel = json.load(open(f"{a.out}/selected.json"))
        if not sel.get("qualified"): print("Dev stage did not qualify a setup. Test not run."); sys.exit(1)
        tr = simulate(bars, b, S[sel["setup"]], sel["hold_h"] * 60, TEST_START, bars.index[-1]); tr.to_csv(f"{a.out}/test_trades.csv", index=False)
        s = summary(tr); lines.append(f"TEST {TEST_START.date()} onward, {sel['setup']}, hold {sel['hold_h']}h: {s}")
        lines.append("VERDICT: " + ("PASSED (net t-stat >= 2)" if s.get("net_t", -9) >= 2 else "positive but not proven" if s.get("net_avg_pct", -1) > 0 else "FAILED"))
        open(f"{a.out}/RESULTS_test.md", "w").write("# BTC flow study - one-time TEST\n\n" + "\n\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
