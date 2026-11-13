"""Grid search over entry signal x entry timing x hold days x stop x take-profit x MA5-exit.
Selection on in-sample (first 60% of dates), reported on out-of-sample (last 40%).
Cost 0.585% per round trip. Same-day stop & target hit -> assume stop first (conservative).
Gap through stop/target -> fill at that day's open."""
import sys, glob, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from backtest_tech import signals

COST = 0.00585
DATA = sys.argv[1] if len(sys.argv) > 1 else "data/price_adj"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/optimize_rules.md"
H = 20
HOLDS, STOPS, TPS = [1, 3, 5, 10, 20], [None, -0.03, -0.05, -0.08], [None, 0.05, 0.10, 0.20]

px = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DATA}/*.parquet"))])
px = px[px.stock_id.str.fullmatch(r"[1-9]\d{3}")].sort_values(["stock_id", "date"]).reset_index(drop=True)
px["date"] = pd.to_datetime(px.date)
dates = np.sort(px.date.unique()); cut = dates[int(len(dates) * 0.6)]

O, Hh, L, C = (px[c].to_numpy(float) for c in ("open", "high", "low", "close"))
N = len(px)
sid = px.stock_id.to_numpy()
ends = np.r_[np.flatnonzero(sid[1:] != sid[:-1]), N - 1]
end_of = np.repeat(ends, np.diff(np.r_[-1, ends]))  # last row index of each row's stock
MA5 = px.groupby("stock_id").close.transform(lambda s: s.rolling(5).mean()).to_numpy()

ev = {}  # signal -> array of row indices t
for s_id, g in px.groupby("stock_id", sort=False):
    if len(g) < 210: continue
    sig, _ = signals(g.reset_index(drop=True))
    val20 = g.value.rolling(20).mean().to_numpy(); c = g.close.to_numpy()
    ok = (val20 >= 2e7) & (c >= 10)
    r5 = g.close.pct_change(5).to_numpy()
    hi250 = g.high.rolling(250).max().to_numpy()
    sig["短線反轉: 5日跌幅>10%"] = pd.Series(r5 < -0.10)
    sig["接近一年新高(收盤>=250日高98%)"] = pd.Series(c >= 0.98 * hi250)
    for k, s in sig.items():
        idx = g.index.to_numpy()[np.asarray(s.fillna(False), bool) & ok]
        ev.setdefault(k, []).append(idx)
ev = {k: np.concatenate(v) for k, v in ev.items()}
# subsample the random baseline for speed
rng = np.random.default_rng(0)
b = [k for k in ev if k.startswith("基準")][0]
ev[b] = rng.choice(ev[b], 60000, replace=False)


def run(t, timing):
    """t: signal-row indices. returns dict config -> net returns array, plus dates."""
    if timing == "open":
        e = t + 1; entry = O[np.minimum(e, N - 1)]
        okb = (e + H <= end_of[t]) & (entry < C[t] * 1.095)
    else:  # buy at signal-day close; skip if closed limit-up (can't buy)
        e = t; entry = C[t]
        prevc = C[np.maximum(t - 1, 0)]
        okb = (e + H + 1 <= end_of[t]) & (C[t] < prevc * 1.095)
    t, e, entry = t[okb], e[okb], entry[okb]
    # window: days after entry day for close-entry, including entry day for open-entry
    start = e if timing == "open" else e + 1
    W = start[:, None] + np.arange(H)[None, :]
    o, h, l, c, m = O[W] / entry[:, None], Hh[W] / entry[:, None], L[W] / entry[:, None], C[W] / entry[:, None], MA5[W] / entry[:, None]
    if timing == "open":  # entry day: open is the entry itself, no gap fill on day 0
        o = o.copy(); o[:, 0] = 1.0
    res = {}
    big = H + 1
    for hold, st, tp, ma in itertools.product(HOLDS, STOPS, TPS, (False, True)):
        hd = min(hold, H)
        px_exit = c[:, hd - 1].copy(); day = np.full(len(t), hd - 1)
        cand = []
        if st is not None:
            hit = l[:, :hd] <= 1 + st
            d = np.where(hit.any(1), hit.argmax(1), big)
            fill = np.minimum(1 + st, o[np.arange(len(t)), np.minimum(d, hd - 1)])
            cand.append((d, fill, 0))
        if tp is not None:
            hit = h[:, :hd] >= 1 + tp
            d = np.where(hit.any(1), hit.argmax(1), big)
            fill = np.maximum(1 + tp, o[np.arange(len(t)), np.minimum(d, hd - 1)])
            cand.append((d, fill, 1))
        if ma:
            hit = c[:, :hd] < m[:, :hd]
            d = np.where(hit.any(1), hit.argmax(1), big)
            fill = c[np.arange(len(t)), np.minimum(d, hd - 1)]
            cand.append((d, fill, 2))
        best_d = np.full(len(t), big); best_f = px_exit
        for d, f, pri in sorted(cand, key=lambda x: x[2]):  # priority: stop, tp, ma on same day
            take = d < best_d
            best_d = np.where(take, d, best_d); best_f = np.where(take, f, best_f)
        res[(hold, st, tp, ma)] = best_f - 1 - COST
    return res, px.date.to_numpy()[t]


def tstat(r, d):
    s = pd.Series(r).groupby(d).mean()
    return s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 2 else np.nan


rows = []
for name, t in ev.items():
    for timing in ("open", "close"):
        res, d = run(np.sort(t), timing)
        oos = d >= cut
        for (hold, st, tp, ma), r in res.items():
            if (~oos).sum() < 200 or oos.sum() < 200: continue
            rows.append(dict(signal=name, timing=timing, hold=hold, stop=st, tp=tp, ma5=ma,
                             is_n=(~oos).sum(), is_avg=r[~oos].mean(), is_win=(r[~oos] > 0).mean(),
                             oos_n=oos.sum(), oos_avg=r[oos].mean(), oos_win=(r[oos] > 0).mean(),
                             oos_t=tstat(r[oos], d[oos]), per_day_oos=r[oos].mean() / hold))
R = pd.DataFrame(rows)
R.to_csv(OUT.replace(".md", ".csv"), index=False)

fmt = lambda x: "無" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.0%}"
def line(i, x):
    return (f"| {i} | {x.signal} | {'當日收盤' if x.timing=='close' else '隔日開盤'} | {x.hold} | {fmt(x.stop)} | {fmt(x.tp)} | {'是' if x.ma5 else '否'} | "
            f"{x.is_avg:+.2%} | {x.is_win:.1%} | {x.oos_n:,} | {x.oos_avg:+.2%} | {x.oos_win:.1%} | {x.oos_t:.1f} |")
hdr = ["| # | 進場訊號 | 買點 | 持有上限(日) | 停損 | 停利 | 跌破5日線出場 | 樣本內平均 | 樣本內勝率 | 樣本外筆數 | 樣本外平均 | 樣本外勝率 | 樣本外t |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
base = R[R.signal.str.startswith("基準")]
L_ = ["# 規則最佳化（解除原本限制）", "",
      f"- 組合數：{len(R):,}（{R.signal.nunique()} 種訊號 × 2 種買點 × {len(HOLDS)} 種持有天數 × {len(STOPS)} 種停損 × {len(TPS)} 種停利 × 跌破5日線出場 有/無）",
      f"- 用樣本內（~{pd.Timestamp(cut):%Y-%m-%d}）挑選，樣本外（{pd.Timestamp(cut):%Y-%m-%d}~）驗證；已扣成本 0.585%；不再排除 20 日漲幅 >50% 個股",
      "- 組合這麼多，樣本內一定會有運氣好的；只看樣本外是否仍為正、t>2",
      "", "## A. 樣本內平均報酬最高的 15 組 → 樣本外表現", ""] + hdr
for i, x in enumerate(R.sort_values("is_avg", ascending=False).head(15).itertuples(), 1): L_.append(line(i, x))
L_ += ["", "## B. 樣本內勝率最高的 10 組（且樣本內平均>0）→ 樣本外表現", ""] + hdr
for i, x in enumerate(R[R.is_avg > 0].sort_values("is_win", ascending=False).head(10).itertuples(), 1): L_.append(line(i, x))
L_ += ["", "## C. 每種訊號的最佳組合（以樣本內挑）", ""] + hdr
best = R.loc[R.groupby("signal").is_avg.idxmax()].sort_values("oos_avg", ascending=False)
for i, x in enumerate(best.itertuples(), 1): L_.append(line(i, x))
L_ += ["", "## D. 買點效應（同訊號、持有5日、無停損停利）", "", "| 訊號 | 隔日開盤買 樣本外平均 | 當日收盤買 樣本外平均 |", "|---|---|---|"]
q = R[(R.hold == 5) & R.stop.isna() & R.tp.isna() & ~R.ma5].pivot_table(index="signal", columns="timing", values="oos_avg")
for s_, r_ in q.iterrows(): L_.append(f"| {s_} | {r_.get('open', np.nan):+.2%} | {r_.get('close', np.nan):+.2%} |")
L_ += ["", f"- 樣本外平均 > 0 且 t > 2 的組合數：{((R.oos_avg > 0) & (R.oos_t > 2)).sum()} / {len(R)}",
       f"- 樣本內平均 > 0 的組合中，樣本外仍 > 0 的比例：{(R[R.is_avg > 0].oos_avg > 0).mean():.1%}"]
open(OUT, "w", encoding="utf-8").write("\n".join(L_) + "\n"); print("\n".join(L_))
