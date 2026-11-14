"""Risk-controlled portfolio search on 12 years (2014-11 ~ now) with walk-forward selection.
Goal (user): max drawdown <= 15%, then highest return. Positions in equal slots (5萬 units in practice).
Core signal: close >= 98% of 250-day high, 20d avg value >= NT$20M, close >= 10, not closed limit-up, not disposed.
Same-day ranking: 120-day momentum (price-only so it works for all 12 years). Buy at close, hold up to 20 days.
Cost 0.585% per round trip. Adjusted prices (FinMind)."""
import sys, glob, itertools, json
import numpy as np, pandas as pd

DIRS = sys.argv[1].split(",")          # price dirs, comma separated
DISP = sys.argv[2].split(",")          # disposition parquet files
OUT = sys.argv[3]
COST = 0.00585; HOLD = 20; MAXDD = 0.15

raw = pd.concat([pd.read_parquet(f) for d in DIRS for f in glob.glob(f"{d}/*.parquet")])
raw["date"] = pd.to_datetime(raw.date)
raw = raw.drop_duplicates(["date", "stock_id"], keep="last")
mk = raw[raw.stock_id == "TAIEX"].set_index("date").close.sort_index()
etf = raw[raw.stock_id == "0050"].set_index("date").close.sort_index()
px = raw[raw.stock_id.str.fullmatch(r"[1-9]\d{3}")].sort_values(["stock_id", "date"]).reset_index(drop=True)
g = px.groupby("stock_id")
px["hi250"] = g.high.transform(lambda s: s.rolling(250).max())
px["val20"] = g.value.transform(lambda s: s.rolling(20).mean())
px["prevc"] = g.close.shift(1)
px["r120"] = g.close.pct_change(120)

# disposition flags
flag = np.zeros(len(px), bool); idx = px.groupby("stock_id").indices; dd = px.date.to_numpy()
for f in DISP:
    d = pd.read_parquet(f)
    d = d[d.stock_id.astype(str).str.fullmatch(r"[1-9]\d{3}")]
    for r in d.itertuples():
        s, e = pd.to_datetime(r.date, errors="coerce"), pd.to_datetime(r.period_end, errors="coerce")
        if r.stock_id in idx and pd.notna(s) and pd.notna(e):
            ii = idx[r.stock_id]; flag[ii[(dd[ii] >= np.datetime64(s)) & (dd[ii] <= np.datetime64(e))]] = True
px["disposed"] = flag

tdates = mk.index[mk.index >= px.date.min()]
start = tdates[260]                                     # need 250-day history
tdates = tdates[tdates >= start]
T = len(tdates); di = {d: i for i, d in enumerate(tdates)}
m60, m20 = mk.rolling(60).mean(), mk.rolling(20).mean()
reg = pd.DataFrame({"a": mk > m60, "b": (mk > m60) & (m60 > m60.shift(20)), "c": (mk > m60) & (m60 > m60.shift(20)) & (mk > m20),
                    "below60": mk < m60}).reindex(tdates).fillna(False)

REG = {k: reg[k].to_numpy() for k in "abc"}; B60 = reg.below60.to_numpy()
sig = (px.close >= 0.98 * px.hi250) & (px.val20 >= 2e7) & (px.close >= 10) & (px.close < px.prevc * 1.095) & ~px.disposed & px.date.isin(tdates)
ev = px.loc[sig, ["date", "stock_id", "r120"]].copy()
ev["d"] = ev.date.map(di); ev = ev.sort_values(["d", "r120"], ascending=[True, False])
byday = {k: list(zip(v.stock_id, v.index)) for k, v in ev.groupby("d")}

# per-stock close/low aligned to tdates (NaN when not trading)
C = px.pivot(index="date", columns="stock_id", values="close").reindex(tdates)
L = px.pivot(index="date", columns="stock_id", values="low").reindex(tdates)
cols = {s: j for j, s in enumerate(C.columns)}; Cv, Lv = C.to_numpy(), L.to_numpy()

def sim(regime, regexit, slots, invest, stop, breaker):
    cash, pos, eq, peak = 1.0, [], np.zeros(T), 1.0
    trades = []
    for i in range(T):
        # mark & exits
        keep = []
        for p in pos:
            j, e0, amt, px0 = p
            c = Cv[i, j]; l = Lv[i, j]
            if c != c:  # not trading today (suspended/delisted): carry last value
                keep.append(p); continue
            out = None
            if stop is not None and i > e0 and l == l and l <= px0 * (1 + stop): out = px0 * (1 + stop)
            elif i - e0 >= HOLD: out = c
            elif regexit and B60[i]: out = c
            if out is not None:
                r = out / px0 - 1 - COST; cash += amt * (1 + r); trades.append(r)
            else: keep.append(p)
        pos = keep
        val = cash + sum(p[2] * ((Cv[i, p[0]] if Cv[i, p[0]] == Cv[i, p[0]] else p[3]) / p[3]) for p in pos)
        peak = max(peak, val); ddn = val / peak - 1
        # entries
        if REG[regime][i] and i in byday and (breaker is None or ddn > -breaker):
            held = {p[0] for p in pos}
            for s, _ in byday[i]:
                if len(pos) >= slots: break
                j = cols[s]
                if j in held: continue
                amt = min(val * invest / slots, cash)
                if amt <= val * 0.01: break
                cash -= amt; pos.append((j, i, amt, Cv[i, j])); held.add(j)
        eq[i] = cash + sum(p[2] * ((Cv[i, p[0]] if Cv[i, p[0]] == Cv[i, p[0]] else p[3]) / p[3]) for p in pos)
    return pd.Series(eq, index=tdates), np.array(trades)

def stats(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return dict(cagr=(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1, mdd=(eq / eq.cummax() - 1).min())

GRID = dict(regime=["a", "b", "c"], regexit=[False, True], slots=[5, 8, 10], invest=[1.0, 0.7, 0.5],
            stop=[None, -0.08, -0.10], breaker=[None, 0.06, 0.10])
keys = list(GRID); res = {}; curves = {}
for combo in itertools.product(*GRID.values()):
    p = dict(zip(keys, combo)); eq, tr = sim(**p)
    k = json.dumps(p); curves[k] = eq
    res[k] = dict(**p, **stats(eq), win=(tr > 0).mean() if len(tr) else np.nan, n=len(tr), avg=tr.mean() if len(tr) else np.nan)
R = pd.DataFrame(res.values(), index=res.keys())

# ---------- walk-forward: each year pick params on all prior years (MDD<=15% then max CAGR) ----------
years = sorted(set(tdates.year))
daily = {k: v.pct_change().fillna(0) for k, v in curves.items()}
wf = []; picks = []
for y in years[2:]:
    tr = {k: v[v.index.year < y] for k, v in curves.items()}
    sc = []
    for k, e in tr.items():
        st = stats(e / e.iloc[0]); sc.append((k, st["cagr"], st["mdd"]))
    ok = [x for x in sc if x[2] >= -MAXDD]
    best = max(ok, key=lambda x: x[1]) if ok else max(sc, key=lambda x: x[2])
    picks.append((y, json.loads(best[0])))
    wf.append(daily[best[0]][daily[best[0]].index.year == y])
wfr = pd.concat(wf); wfe = (1 + wfr).cumprod()
e0 = etf.reindex(wfe.index).ffill(); e0 = e0 / e0.iloc[0]

lab = {"a": "大盤>60日線", "b": "大盤>60日線且60日線上揚", "c": "上一項＋大盤>20日線"}
fmt = lambda r: f"{lab[r.regime]}｜{'跌破60日線全出' if r.regexit else '不強制出'}｜{int(r.slots)}檔｜投入{int(r.invest*100)}%｜停損{'無' if pd.isna(r.stop) or r.stop is None else f'{r.stop:.0%}'}｜熔斷{'無' if pd.isna(r.breaker) or r.breaker is None else f'回落{r.breaker:.0%}停買'}"
yr = lambda e: (e.groupby(e.index.year).last() / e.groupby(e.index.year).first() - 1)
Lx = ["# 風控最佳化（12 年：%s ~ %s）" % (tdates[0].date(), tdates[-1].date()), "",
      "- 目標：最大回落 ≤ 15%，在此條件下報酬最高；成本 0.585%；收盤買、最多抱 20 日；同日多檔依 120 日漲幅排序",
      f"- 組合數 {len(R)}；資料含 2015、2018、2020、2022 等空頭；處置股已排除；美股濾網因美股資料只有 2021 之後，12 年測試未納入",
      "", "## 1. 全期間：最大回落 ≤15% 的組合中，年化最高前 10 名（注意：這是全期間挑的，會偏樂觀）", "",
      "| 規則 | 年化 | 最大回落 | 交易勝率 | 每筆平均 | 交易數 |", "|---|---|---|---|---|---|"]
for k, r in R[R.mdd >= -MAXDD].sort_values("cagr", ascending=False).head(10).iterrows():
    Lx.append(f"| {fmt(r)} | {r.cagr:+.1%} | {r.mdd:.1%} | {r.win:.1%} | {r.avg:+.2%} | {r.n} |")
Lx += ["", f"- 全部 {len(R)} 組中，最大回落 ≤15% 的有 {(R.mdd >= -MAXDD).sum()} 組",
       "", "## 2. Walk-forward（每年只用過去資料挑規則，這才是可信的成績）", "",
       f"- 年化 {stats(wfe)['cagr']:+.1%}，最大回落 {stats(wfe)['mdd']:.1%}（{wfe.index[0].date()} ~ {wfe.index[-1].date()}）",
       f"- 同期間 0050：年化 {stats(e0)['cagr']:+.1%}，最大回落 {stats(e0)['mdd']:.1%}", "",
       "| 年度 | 當年採用規則（只用之前資料挑） | 策略報酬 | 0050 |", "|---|---|---|---|"]
wy, ey = yr(wfe), yr(e0)
for y, p in picks:
    Lx.append(f"| {y} | {fmt(pd.Series(p))} | {wy.get(y, np.nan):+.1%} | {ey.get(y, np.nan):+.1%} |")
best_k = R[R.mdd >= -MAXDD].sort_values("cagr", ascending=False).index[0]
Lx += ["", "## 3. 全期最佳組合的各年表現", "", "| 年度 | 策略 | 0050 |", "|---|---|---|"]
bb = yr(curves[best_k]); e1 = etf.reindex(curves[best_k].index).ffill(); be = yr(e1)
for y in bb.index: Lx.append(f"| {y} | {bb[y]:+.1%} | {be.get(y, np.nan):+.1%} |")
open(OUT, "w", encoding="utf-8").write("\n".join(Lx) + "\n")
R.to_csv(OUT.replace(".md", ".csv"))
print("\n".join(Lx))
