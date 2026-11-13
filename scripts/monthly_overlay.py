"""Monthly hold (max 20 trading days) on the '52-week-high' signal, with daily market/stock checks.
Part 1: event study  - entry filters (market regime) x daily exit rules, IS/OOS split.
Part 2: portfolio    - 5 slots, equal weight, buy at close, real cash constraints, vs 0050.
Cost 0.585% per round trip. Prices: FinMind adjusted."""
import sys, glob, itertools
import numpy as np, pandas as pd

COST = 0.00585; H = 20
DATA = sys.argv[1] if len(sys.argv) > 1 else "data/price_adj"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/monthly_overlay.md"

raw = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{DATA}/*.parquet"))])
raw["date"] = pd.to_datetime(raw.date)
mk = raw[raw.stock_id == "TAIEX"].set_index("date").close.sort_index()
etf = raw[raw.stock_id == "0050"].set_index("date").close.sort_index()
px = raw[raw.stock_id.str.fullmatch(r"[1-9]\d{3}")].sort_values(["stock_id", "date"]).reset_index(drop=True)
g = px.groupby("stock_id")
px["ma10"] = g.close.transform(lambda s: s.rolling(10).mean())
px["ma20"] = g.close.transform(lambda s: s.rolling(20).mean())
px["hi250"] = g.high.transform(lambda s: s.rolling(250).max())
px["val20"] = g.value.transform(lambda s: s.rolling(20).mean())
px["ret120"] = g.close.transform(lambda s: s.pct_change(120))
px["prevc"] = g.close.shift(1)

# market regime series aligned to stock rows
m20, m60 = mk.rolling(20).mean(), mk.rolling(60).mean()
mkt = pd.DataFrame({"m": mk, "above20": mk > m20, "above60": mk > m60, "m20up": m20 > m20.shift(5)})
px = px.join(mkt, on="date")

N = len(px); sid = px.stock_id.to_numpy()
ends = np.r_[np.flatnonzero(sid[1:] != sid[:-1]), N - 1]
end_of = np.repeat(ends, np.diff(np.r_[-1, ends]))
C, L = px.close.to_numpy(float), px.low.to_numpy(float)
MA10, MA20 = px.ma10.to_numpy(), px.ma20.to_numpy()
A20, A60 = px.above20.fillna(False).to_numpy(), px.above60.fillna(False).to_numpy()

sig = ((px.close >= 0.98 * px.hi250) & (px.val20 >= 2e7) & (px.close >= 10)
       & (px.close < px.prevc * 1.095)).to_numpy()            # not closed limit-up (buyable at close)
t_all = np.flatnonzero(sig & (np.arange(N) + H + 1 <= end_of))
dates = np.sort(px.date.unique()); cut = dates[int(len(dates) * 0.6)]

ENTRY = {"不過濾": np.ones(N, bool), "大盤在20日線上": A20, "大盤在60日線上": A60,
         "大盤20日線上揚": px.m20up.fillna(False).to_numpy()}

W = t_all[:, None] + 1 + np.arange(H)[None, :]          # days after entry close
entry = C[t_all]
c, l = C[W] / entry[:, None], L[W] / entry[:, None]
ma10, ma20 = MA10[W] / entry[:, None], MA20[W] / entry[:, None]
mA20, mA60 = A20[W], A60[W]
peak = np.maximum.accumulate(c, axis=1)
big = H + 1
def first(hit):
    return np.where(hit.any(1), hit.argmax(1), big)

EXIT = {
    "固定持有20日": [],
    "個股跌破20日線": [("c", c < ma20)],
    "個股跌破10日線": [("c", c < ma10)],
    "從高點回落8%": [("c", c < peak * 0.92)],
    "從高點回落12%": [("c", c < peak * 0.88)],
    "硬停損-8%(盤中)": [("stop", l <= 0.92)],
    "大盤跌破20日線": [("c", ~mA20)],
    "大盤跌破60日線": [("c", ~mA60)],
    "個股破20日線 或 大盤破20日線": [("c", (c < ma20) | ~mA20)],
    "從高點回落12% 或 大盤破60日線": [("c", (c < peak * 0.88) | ~mA60)],
}
rows = []; keep = {}
ix = np.arange(len(t_all))
for ename, rules in EXIT.items():
    d_best = np.full(len(t_all), H - 1); f_best = c[:, H - 1].copy()
    for kind, hit in rules:
        d = first(hit)
        f = np.minimum(0.92, c[ix, np.minimum(d, H - 1)] * 0 + 0.92) if kind == "stop" else c[ix, np.minimum(d, H - 1)]
        take = d < d_best
        d_best = np.where(take, d, d_best); f_best = np.where(take, f, f_best)
    r = f_best - 1 - COST; hold = d_best + 1
    keep[ename] = (r, hold)
    dd = px.date.to_numpy()[t_all]
    for fname, F in ENTRY.items():
        m = F[t_all]
        isn, oos = m & (dd < cut), m & (dd >= cut)
        def tt(mask):
            s = pd.Series(r[mask]).groupby(dd[mask]).mean()
            return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
        rows.append(dict(entry=fname, exit=ename, is_n=isn.sum(), is_avg=r[isn].mean(), is_win=(r[isn] > 0).mean(),
                         oos_n=oos.sum(), oos_avg=r[oos].mean(), oos_win=(r[oos] > 0).mean(), oos_t=tt(oos),
                         hold=hold[m].mean(), worst=np.percentile(r[m], 5)))
R = pd.DataFrame(rows).sort_values("is_avg", ascending=False)

# ---------- Part 2: portfolio simulation ----------
def simulate(fname, ename, slots=5, start_cash=1_000_000):
    r_, hold_ = keep[ename]; F = ENTRY[fname]
    ev = pd.DataFrame({"t": t_all, "date": px.date.to_numpy()[t_all], "r": r_, "hold": hold_,
                       "rank": px.ret120.to_numpy()[t_all], "sid": sid[t_all]})
    ev = ev[F[t_all]].sort_values(["date", "rank"], ascending=[True, False])
    di = {d: i for i, d in enumerate(dates)}
    ev["d0"] = ev.date.map(di); ev["d1"] = ev.d0 + ev.hold
    cash = start_cash; open_pos = []  # (exit_day_idx, amount, r, sid)
    eq = np.zeros(len(dates)); byday = dict(tuple(ev.groupby("d0"))); trades = []
    for i in range(len(dates)):
        still = []
        for p in open_pos:
            if p[0] <= i: cash += p[1] * (1 + p[2]); trades.append(p[2])
            else: still.append(p)
        open_pos = still
        if i in byday and len(open_pos) < slots:
            held = {p[3] for p in open_pos}
            for x in byday[i].itertuples():
                if len(open_pos) >= slots: break
                if x.sid in held: continue
                amt = (cash + sum(p[1] for p in open_pos)) / slots
                amt = min(amt, cash)
                if amt <= 0: break
                cash -= amt; open_pos.append((x.d1, amt, x.r, x.sid, x.t, x.d0)); held.add(x.sid)
        eq[i] = cash + sum(p[1] * C[p[4] + (i - p[5])] / C[p[4]] for p in open_pos)   # mark-to-market
    eq = pd.Series(eq, index=pd.to_datetime(dates))
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    mdd = (eq / eq.cummax() - 1).min()
    yr = eq.groupby(eq.index.year).last() / eq.groupby(eq.index.year).first() - 1
    return dict(total=eq.iloc[-1] / eq.iloc[0] - 1, cagr=(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, mdd=mdd,
                n=len(trades), win=np.mean(np.array(trades) > 0), yearly=yr)

e0 = etf.loc[dates[0]:dates[-1]]
etf_y = e0.groupby(e0.index.year).last() / e0.groupby(e0.index.year).first() - 1
etf_mdd = (e0 / e0.cummax() - 1).min()
picks = [("不過濾", "固定持有20日")] + list(R.head(4)[["entry", "exit"]].itertuples(index=False, name=None))
picks = list(dict.fromkeys(picks))
P = [(f, e, simulate(f, e)) for f, e in picks]

L_ = ["# 月進月出 + 每日檢查大盤／個股（訊號：接近一年新高，收盤買進，最多持有20日）", "",
      f"- 期間 {pd.Timestamp(dates[0]):%Y-%m-%d} ~ {pd.Timestamp(dates[-1]):%Y-%m-%d}；樣本外自 {pd.Timestamp(cut):%Y-%m-%d}；已扣成本 0.585%",
      "- 每日檢查的出場條件都在『收盤』判斷、收盤出場（硬停損為盤中觸價）",
      "", "## 1. 進場過濾 × 每日出場規則（依樣本內平均排序）", "",
      "| # | 進場時大盤條件 | 每日出場規則 | 平均持有日 | 樣本內平均 | 樣本內勝率 | 樣本外筆數 | 樣本外平均 | 樣本外勝率 | 樣本外t | 最差5%單筆 |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for i, x in enumerate(R.itertuples(), 1):
    L_.append(f"| {i} | {x.entry} | {x.exit} | {x.hold:.1f} | {x.is_avg:+.2%} | {x.is_win:.1%} | {x.oos_n:,} | {x.oos_avg:+.2%} | {x.oos_win:.1%} | {x.oos_t:.1f} | {x.worst:+.1%} |")
L_ += ["", "## 2. 實際資金模擬（100萬、同時最多5檔、每檔約1/5資金、同日多訊號取120日漲幅最強者）", "",
       "| 組合 | 總報酬 | 年化 | 最大回落 | 交易次數 | 勝率 | " + " | ".join(str(y) for y in etf_y.index) + " |",
       "|---|---|---|---|---|---|" + "---|" * len(etf_y)]
for f, e, s in P:
    L_.append(f"| {f} + {e} | {s['total']:+.1%} | {s['cagr']:+.1%} | {s['mdd']:.1%} | {s['n']} | {s['win']:.1%} | "
              + " | ".join(f"{s['yearly'].get(y, np.nan):+.1%}" for y in etf_y.index) + " |")
yrs = (e0.index[-1] - e0.index[0]).days / 365.25
L_.append(f"| 對照：0050 買進持有 | {e0.iloc[-1]/e0.iloc[0]-1:+.1%} | {(e0.iloc[-1]/e0.iloc[0])**(1/yrs)-1:+.1%} | {etf_mdd:.1%} | 1 | — | "
          + " | ".join(f"{v:+.1%}" for v in etf_y.values) + " |")
L_ += ["", "- 資金模擬每日以收盤價計算持股市值（含帳面損益），最大回落為帳面淨值從高點的最大跌幅",
       "- 尚未排除處置股；尚未考慮成交量不足時買不到的滑價"]
open(OUT, "w", encoding="utf-8").write("\n".join(L_) + "\n"); print("\n".join(L_))
