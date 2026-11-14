"""Factor study + integrated model on top of the price-only baseline.
Point-in-time rules:
  monthly revenue  -> usable from the 11th of the month it is published (FinMind 'date' = 1st of publish month)
  financials       -> usable after the legal deadline (Q1 5/15, Q2 8/14, Q3 11/14, Q4 3/31 next year)
  institutional / margin / foreign / PER -> published after close, used with a 1-trading-day lag
  shareholder distribution (weekly) -> usable from the next trading day after the data date
Trades: signal on close of day t, buy at close of t (closing auction), cost 0.585% round trip.
Selection of which factors enter the integrated score uses ONLY the in-sample period."""
import sys, os, glob, json
import numpy as np, pandas as pd

COST = 0.00585
PRICE = sys.argv[1] if len(sys.argv) > 1 else "data/price_adj"
FAC = sys.argv[2] if len(sys.argv) > 2 else "data/factors"
OUT = sys.argv[3] if len(sys.argv) > 3 else "results/factor_study.md"
rng = np.random.default_rng(7)

raw = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{PRICE}/*.parquet"))])
raw["date"] = pd.to_datetime(raw.date)
mk = raw[raw.stock_id == "TAIEX"].set_index("date").close.sort_index()
px = raw[raw.stock_id.str.fullmatch(r"[1-9]\d{3}")].sort_values(["stock_id", "date"]).reset_index(drop=True)
g = px.groupby("stock_id")
for n in (5, 20, 60):
    px[f"ma{n}"] = g.close.transform(lambda s, n=n: s.rolling(n).mean())
px["hi250"] = g.high.transform(lambda s: s.rolling(250).max())
px["val20"] = g.value.transform(lambda s: s.rolling(20).mean())
px["vol20"] = g.volume.transform(lambda s: s.rolling(20).mean())
px["prevc"] = g.close.shift(1)
px["r5"] = g.close.pct_change(5); px["r20"] = g.close.pct_change(20); px["r120"] = g.close.pct_change(120)
for h in (5, 20):
    px[f"fwd{h}"] = g.close.shift(-h) / px.close - 1 - COST
m60 = mk.rolling(60).mean()
px = px.join(pd.DataFrame({"mkt_up": mk > m60}), on="date")
tdates = np.sort(px.date.unique())
cut = tdates[int(len(tdates) * 0.6)]
avail = {}

def next_tday(d):
    i = np.searchsorted(tdates, np.asarray(d, "datetime64[ns]"), side="right")
    return pd.to_datetime(tdates[np.minimum(i, len(tdates) - 1)])

def asof(df, cols):
    """merge per-stock features keyed by 'avail' date onto px rows (as-of, no look-ahead)."""
    global px
    df = df.dropna(subset=["avail"]).sort_values("avail")[["stock_id", "avail"] + cols]
    px = px.sort_values("date")
    px = pd.merge_asof(px, df, left_on="date", right_on="avail", by="stock_id", direction="backward").drop(columns="avail")
    px = px.sort_values(["stock_id", "date"]).reset_index(drop=True)

F = {}  # factor name -> boolean column (NaN = unknown)

# ---------- disposition ----------
p = f"{FAC}/disposition.parquet"
if os.path.exists(p):
    d = pd.read_parquet(p); d = d[d.stock_id.str.fullmatch(r"[1-9]\d{3}")]
    d["s"] = pd.to_datetime(d.period_start, errors="coerce"); d["e"] = pd.to_datetime(d.period_end, errors="coerce")
    d["ann"] = pd.to_datetime(d.date)
    flag = np.zeros(len(px), bool)
    idx = px.groupby("stock_id").indices
    dd = px.date.to_numpy()
    for r in d.dropna(subset=["s", "e"]).itertuples():
        if r.stock_id in idx:
            ii = idx[r.stock_id]; sel = (dd[ii] >= np.datetime64(r.ann)) & (dd[ii] <= np.datetime64(r.e))
            flag[ii[sel]] = True
    px["disposed"] = flag
    avail["處置股"] = True

# ---------- monthly revenue ----------
p = f"{FAC}/revenue.parquet"
if os.path.exists(p):
    r = pd.read_parquet(p).drop_duplicates(["stock_id", "revenue_year", "revenue_month"])
    r = r.sort_values(["stock_id", "revenue_year", "revenue_month"])
    r["ym"] = r.revenue_year * 12 + r.revenue_month
    r = r.merge(r[["stock_id", "ym", "revenue"]].assign(ym=lambda x: x.ym + 12).rename(columns={"revenue": "rev_ly"}),
                on=["stock_id", "ym"], how="left")
    r["yoy"] = r.revenue / r.rev_ly - 1
    gr = r.groupby("stock_id")
    r["yoy3"] = gr.revenue.transform(lambda s: s.rolling(3).sum()) / gr.rev_ly.transform(lambda s: s.rolling(3).sum()) - 1
    r["yoy3_prev"] = gr.yoy3.shift(3)
    r["rev_hi12"] = r.revenue >= gr.revenue.transform(lambda s: s.rolling(12, min_periods=12).max())
    r["avail"] = pd.to_datetime(r.date) + pd.Timedelta(days=10)
    asof(r, ["yoy", "yoy3", "yoy3_prev", "rev_hi12"])
    F["月營收年增>20%"] = px.yoy > 0.2
    F["近3月營收年增>15%"] = px.yoy3 > 0.15
    F["營收成長加速(近3月年增>前3月)"] = px.yoy3 > px.yoy3_prev
    F["月營收創12個月新高"] = px.rev_hi12.astype("float") > 0.5
    for k in list(F): F[k] = F[k].where(px.yoy.notna() if "營收" in k else True)

# ---------- financial statements ----------
p = f"{FAC}/fin.parquet"
if os.path.exists(p):
    f = pd.read_parquet(p); f["date"] = pd.to_datetime(f.date)
    dl = {3: (5, 15), 6: (8, 14), 9: (11, 14), 12: (3, 31)}
    f["avail"] = [pd.Timestamp(d.year + (d.month == 12), *dl[d.month]) + pd.Timedelta(days=1) if d.month in dl else pd.NaT for d in f.date]
    f = f.sort_values(["stock_id", "date"])
    f["eps_ly"] = f.groupby("stock_id").EPS.shift(4)
    f["eps_yoy_up"] = f.EPS > f.eps_ly
    f["eps_pos"] = f.EPS > 0
    if "GrossProfit" in f and "Revenue" in f:
        f["gm"] = f.GrossProfit / f.Revenue; f["gm_up"] = f.gm > f.groupby("stock_id").gm.shift(4)
    else: f["gm_up"] = np.nan
    f["fin_avail_date"] = f.avail
    asof(f, ["eps_yoy_up", "eps_pos", "gm_up", "fin_avail_date"])
    F["EPS 較去年同季成長"] = px.eps_yoy_up.astype(float) > 0.5
    F["EPS 為正"] = px.eps_pos.astype(float) > 0.5
    F["毛利率較去年同季提升"] = px.gm_up.astype(float) > 0.5
    F["財報公布後60日內且EPS成長"] = F["EPS 較去年同季成長"] & ((px.date - px.fin_avail_date).dt.days <= 90)
    for k in ["EPS 較去年同季成長", "EPS 為正", "毛利率較去年同季提升", "財報公布後60日內且EPS成長"]:
        F[k] = F[k].where(px.fin_avail_date.notna())

def lagged_daily(name, cols, fn):
    p = f"{FAC}/{name}.parquet"
    if not os.path.exists(p): return None
    d = pd.read_parquet(p); d["date"] = pd.to_datetime(d.date)
    d = d.sort_values(["stock_id", "date"])
    d = fn(d)
    d["avail"] = next_tday(d.date)  # published after close -> usable next trading day
    asof(d, cols); return True

# ---------- institutional ----------
def inst_fn(d):
    cols = [c for c in d.columns if c not in ("date", "stock_id")]
    trust = [c for c in cols if "Trust" in c]; foreign = [c for c in cols if c.startswith("Foreign")]
    d["trust"] = d[trust].sum(1) if trust else np.nan
    d["foreign"] = d[foreign].sum(1) if foreign else np.nan
    gg = d.groupby("stock_id")
    d["trust5"] = gg.trust.transform(lambda s: s.rolling(5).sum())
    d["foreign5"] = gg.foreign.transform(lambda s: s.rolling(5).sum())
    pos = (d.trust > 0).astype(int)
    d["trust_streak"] = pos.groupby((pos != pos.shift()).cumsum()).cumsum() * pos
    return d
if lagged_daily("inst", ["trust5", "foreign5", "trust_streak"], inst_fn):
    F["投信近5日買超"] = (px.trust5 > 0).where(px.trust5.notna())
    F["投信連續買超>=3日"] = (px.trust_streak >= 3).where(px.trust_streak.notna())
    F["外資近5日買超"] = (px.foreign5 > 0).where(px.foreign5.notna())
    F["投信近5日買超>20日均量5%"] = (px.trust5 > 0.05 * px.vol20).where(px.trust5.notna())
    F["土洋同買(投信+外資近5日皆買超)"] = ((px.trust5 > 0) & (px.foreign5 > 0)).where(px.trust5.notna())

# ---------- margin ----------
def margin_fn(d):
    gg = d.groupby("stock_id")
    d["mg_chg5"] = d.MarginPurchaseTodayBalance / gg.MarginPurchaseTodayBalance.shift(5) - 1
    d["mg_use"] = d.MarginPurchaseTodayBalance / d.MarginPurchaseLimit.replace(0, np.nan)
    return d
if lagged_daily("margin", ["mg_chg5", "mg_use"], margin_fn):
    F["融資5日減少"] = (px.mg_chg5 < 0).where(px.mg_chg5.notna())
    F["融資使用率<20%"] = (px.mg_use < 0.2).where(px.mg_use.notna())

# ---------- foreign holding ratio ----------
def foreign_fn(d):
    d["fr_chg20"] = d.ForeignInvestmentSharesRatio - d.groupby("stock_id").ForeignInvestmentSharesRatio.shift(20)
    return d
if lagged_daily("foreign", ["fr_chg20"], foreign_fn):
    F["外資持股比率20日上升"] = (px.fr_chg20 > 0).where(px.fr_chg20.notna())

# ---------- PER ----------
def per_fn(d): return d
if lagged_daily("per", ["PER", "PBR"], per_fn):
    F["本益比<25(且>0)"] = ((px.PER > 0) & (px.PER < 25)).where(px.PER.notna())
    F["股價淨值比<3"] = (px.PBR < 3).where(px.PBR.notna())

# ---------- shareholder distribution (weekly) ----------
p = f"{FAC}/holding.parquet"
if os.path.exists(p):
    h = pd.read_parquet(p); h["date"] = pd.to_datetime(h.date); h = h.sort_values(["stock_id", "date"])
    gh = h.groupby("stock_id")
    h["big_chg"] = h.pct_1000up - gh.pct_1000up.shift(1)
    h["big_chg4"] = h.pct_1000up - gh.pct_1000up.shift(4)
    h["small_chg4"] = h.pct_lt50 - gh.pct_lt50.shift(4)
    h["avail"] = next_tday(h.date)
    asof(h, ["big_chg", "big_chg4", "small_chg4"])
    F["千張大戶持股週增"] = (px.big_chg > 0).where(px.big_chg.notna())
    F["千張大戶持股4週增"] = (px.big_chg4 > 0).where(px.big_chg4.notna())
    F["散戶(<50張)持股4週減"] = (px.small_chg4 < 0).where(px.small_chg4.notna())

# ---------- price-based extras (no lag needed, known at close) ----------
F["站上20日線"] = px.close > px.ma20
F["均線多頭 20>60"] = px.ma20 > px.ma60
F["量能放大(成交量>20日均量1.5倍)"] = px.volume > 1.5 * px.vol20
F["近20日漲幅<30%(未過熱)"] = px.r20 < 0.3
F["120日漲幅前段(>30%)"] = px.r120 > 0.3

# ---------- event sets ----------
liq = (px.val20 >= 2e7) & (px.close >= 10) & (px.close < px.prevc * 1.095)
if "disposed" in px: liq &= ~px.disposed
base_sig = (px.close >= 0.98 * px.hi250) & px.mkt_up.fillna(False) & liq
oos = px.date >= cut

def evalset(mask, h):
    r = px.loc[mask, f"fwd{h}"].dropna(); d = px.loc[r.index, "date"]
    if len(r) < 100: return None
    dm = r.groupby(d).mean()
    return dict(n=len(r), avg=r.mean(), win=(r > 0).mean(), t=dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm))) if len(dm) > 2 else np.nan)

L = ["# 因子研究與統合模型", "",
     f"- 期間 {pd.Timestamp(tdates[0]):%Y-%m-%d} ~ {pd.Timestamp(tdates[-1]):%Y-%m-%d}；樣本外自 {pd.Timestamp(cut):%Y-%m-%d}；收盤買進；已扣成本 0.585%",
     f"- 已納入資料：價格、" + "、".join(lab for fn, lab in [("disposition","處置股"),("revenue","月營收"),("fin","財報"),("inst","三大法人"),("holding","集保大戶"),("margin","融資"),("foreign","外資持股"),("per","本益比")] if os.path.exists(f"{FAC}/{fn}.parquet")),
     "- 已排除處置期間股票" if "disposed" in px else "- 尚未排除處置股",
     "", "## 1. 單一因子：在『基準策略』（大盤在60日線上＋接近一年新高）中，符合 vs 不符合", "",
     "| 因子 | 持有 | 樣本內 符合 平均/勝率 | 樣本內 不符合 平均/勝率 | 樣本外 符合 平均/勝率 (筆數) | 樣本外 不符合 平均/勝率 | 樣本外差距 | 兩段皆有效 |",
     "|---|---|---|---|---|---|---|---|"]
keep_rows = []
for name, s in F.items():
    s = s.reindex(px.index)
    for h in (20, 5):
        a = evalset(base_sig & (s == True) & ~oos, h); b = evalset(base_sig & (s == False) & ~oos, h)
        c = evalset(base_sig & (s == True) & oos, h); e = evalset(base_sig & (s == False) & oos, h)
        if not all([a, b, c, e]): continue
        ok = (a["avg"] > b["avg"]) and (c["avg"] > e["avg"])
        keep_rows.append(dict(name=name, h=h, is_gap=a["avg"] - b["avg"], oos_gap=c["avg"] - e["avg"], ok=ok,
                              is_win_gap=a["win"] - b["win"], oos_win_gap=c["win"] - e["win"]))
        L.append(f"| {name} | {h}日 | {a['avg']:+.2%} / {a['win']:.0%} | {b['avg']:+.2%} / {b['win']:.0%} | "
                 f"{c['avg']:+.2%} / {c['win']:.0%} ({c['n']:,}) | {e['avg']:+.2%} / {e['win']:.0%} | {c['avg']-e['avg']:+.2%} | {'✅' if ok else '❌'} |")
K = pd.DataFrame(keep_rows)

# ---------- integrated score: factors chosen on IN-SAMPLE only (20d hold), win-rate and avg both better ----------
chosen = K[(K.h == 20) & (K.is_gap > 0.003) & (K.is_win_gap > 0)].name.tolist()
score = sum(F[n].reindex(px.index).fillna(False).astype(int) for n in chosen) if chosen else pd.Series(0, index=px.index)
px["score"] = score
L += ["", "## 2. 統合評分（只用樣本內挑因子，樣本外驗證）", "",
      f"- 入選因子（樣本內：符合者平均報酬高 0.3 個百分點以上且勝率較高）：{len(chosen)} 個 → " + "、".join(chosen), "",
      "| 分數 | 持有 | 樣本內筆數 | 樣本內平均 | 樣本內勝率 | 樣本外筆數 | 樣本外平均 | 樣本外勝率 | 樣本外t |", "|---|---|---|---|---|---|---|---|---|"]
q = pd.qcut(score[base_sig], 4, labels=False, duplicates="drop") if base_sig.sum() else None
bins = sorted(score[base_sig].unique())
groups = {}
if len(bins) <= 7:
    groups = {f"{int(b)} 分": score == b for b in bins}
else:
    lo, hi = np.percentile(score[base_sig], [33, 67])
    groups = {f"低 (≤{lo:.0f})": score <= lo, f"中": (score > lo) & (score <= hi), f"高 (>{hi:.0f})": score > hi}
top_cut = np.percentile(score[base_sig], 67) if len(bins) > 1 else -1
if len(bins) > 1 and not (score[base_sig] > top_cut).any(): top_cut = bins[-2]
for gname, gm in groups.items():
    for h in (20, 5):
        a = evalset(base_sig & gm & ~oos, h); c = evalset(base_sig & gm & oos, h)
        if a and c:
            L.append(f"| {gname} | {h}日 | {a['n']:,} | {a['avg']:+.2%} | {a['win']:.1%} | {c['n']:,} | {c['avg']:+.2%} | {c['win']:.1%} | {c['t']:.1f} |")
a = evalset(base_sig & ~oos, 20); c = evalset(base_sig & oos, 20)
L.append(f"| 全部基準訊號 | 20日 | {a['n']:,} | {a['avg']:+.2%} | {a['win']:.1%} | {c['n']:,} | {c['avg']:+.2%} | {c['win']:.1%} | {c['t']:.1f} |")

# ---------- win-rate tools: take-profit on top-score group ----------
N = len(px); sid = px.stock_id.to_numpy()
ends = np.r_[np.flatnonzero(sid[1:] != sid[:-1]), N - 1]; end_of = np.repeat(ends, np.diff(np.r_[-1, ends]))
C, Hh = px.close.to_numpy(float), px.high.to_numpy(float)
top = base_sig & (score > top_cut)
t_all = np.flatnonzero(top.to_numpy() & (np.arange(N) + 21 <= end_of))
W = t_all[:, None] + 1 + np.arange(20)[None, :]
c_, h_ = C[W] / C[t_all][:, None], Hh[W] / C[t_all][:, None]
dts = px.date.to_numpy()[t_all]; o_ = dts >= np.datetime64(cut)
L += ["", f"## 3. 提高勝率的出場設計（高分組：分數 > {top_cut:.0f}，最多持有20日）", "",
      "| 停利 | 樣本外勝率 | 樣本外平均 | 平均持有日 | 樣本內勝率 | 樣本內平均 |", "|---|---|---|---|---|---|"]
for tp in (None, 0.05, 0.08, 0.10, 0.15, 0.20):
    if tp is None: r = c_[:, -1] - 1 - COST; hold = np.full(len(t_all), 20)
    else:
        hit = h_ >= 1 + tp; d = np.where(hit.any(1), hit.argmax(1), 19)
        r = np.where(hit.any(1), tp, c_[:, -1] - 1) - COST; hold = d + 1
    L.append(f"| {'無' if tp is None else f'+{tp:.0%}'} | {(r[o_] > 0).mean():.1%} | {r[o_].mean():+.2%} | {hold[o_].mean():.1f} | {(r[~o_] > 0).mean():.1%} | {r[~o_].mean():+.2%} |")

open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
px[["date", "stock_id", "score"]].to_parquet(OUT.replace(".md", "_scores.parquet")) if False else None
json.dump({"chosen": chosen}, open(OUT.replace(".md", ".json"), "w"), ensure_ascii=False)
print("\n".join(L))

# ---------- 4. portfolio simulation with score ranking + luck test ----------
etf = raw[raw.stock_id == "0050"].set_index("date").close.sort_index()
didx = {d: i for i, d in enumerate(tdates)}
okH = (np.arange(N) + 21 <= end_of)
ev_all = np.flatnonzero(base_sig.to_numpy() & okH)
ev_d = np.array([didx[d] for d in px.date.to_numpy()[ev_all]])
ev_sid = sid[ev_all]
robust = K[(K.h == 20) & K.ok].name.tolist()
score_r = sum(F[n].reindex(px.index).fillna(False).astype(int) for n in robust) if robust else pd.Series(0, index=px.index)

def sim(rank, slots, seed):
    rr = np.random.default_rng(seed)
    key = rank + rr.random(len(rank)) * 1e-3          # ties broken at random
    order = np.lexsort((-key, ev_d))
    byday = {}
    for j in order: byday.setdefault(ev_d[j], []).append(j)
    cash, pos, eq, trades = 1.0, [], np.zeros(len(tdates)), []
    for i in range(len(tdates)):
        keep = []
        for p in pos:
            if p[0] <= i:
                r = C[p[2] + 20] / C[p[2]] - 1 - COST; cash += p[1] * (1 + r); trades.append(r)
            else: keep.append(p)
        pos = keep
        if i in byday:
            held = {p[3] for p in pos}
            for j in byday[i]:
                if len(pos) >= slots: break
                if ev_sid[j] in held: continue
                tot = cash + sum(p[1] * C[p[2] + (i - p[4])] / C[p[2]] for p in pos)
                amt = min(tot / slots, cash)
                if amt <= 1e-9: break
                cash -= amt; pos.append((i + 20, amt, ev_all[j], ev_sid[j], i)); held.add(ev_sid[j])
        eq[i] = cash + sum(p[1] * C[p[2] + (i - p[4])] / C[p[2]] for p in pos)
    eq = pd.Series(eq, index=pd.to_datetime(tdates))
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    o = eq[eq.index >= cut]
    return dict(cagr=eq.iloc[-1] ** (1 / yrs) - 1, mdd=(eq / eq.cummax() - 1).min(),
                oos_cagr=(o.iloc[-1] / o.iloc[0]) ** (365.25 / (o.index[-1] - o.index[0]).days) - 1,
                win=np.mean(np.array(trades) > 0), n=len(trades))

variants = {"不用評分（同日隨機挑）": np.zeros(len(ev_all)),
            "評分A：只用樣本內挑的因子": score.to_numpy()[ev_all].astype(float),
            "評分B：前後兩段都有效的因子（含事後資訊，樣本外會偏樂觀）": score_r.to_numpy()[ev_all].astype(float)}
L += ["", "## 4. 實際資金模擬（大盤在60日線上＋接近一年新高，收盤買、持有20日；同日多檔時依評分排序，同分隨機；各跑 40 次）", "",
      f"- 評分B 使用的因子：{'、'.join(robust) if robust else '無'}", "",
      "| 選股方式 | 持股數 | 全期年化 中位數 | 10%～90% | 樣本外年化 中位數 | 最大回落 中位數 | 交易勝率 中位數 |", "|---|---|---|---|---|---|---|"]
for vname, rk in variants.items():
    for slots in (5, 10):
        S = [sim(rk, slots, s) for s in range(40)]
        cg = np.array([x["cagr"] for x in S]); oc = np.array([x["oos_cagr"] for x in S])
        L.append(f"| {vname} | {slots} | {np.median(cg):+.1%} | {np.percentile(cg,10):+.1%} ～ {np.percentile(cg,90):+.1%} | "
                 f"{np.median(oc):+.1%} | {np.median([x['mdd'] for x in S]):.1%} | {np.median([x['win'] for x in S]):.1%} |")
e0 = etf.loc[tdates[0]:tdates[-1]]; yrs = (e0.index[-1] - e0.index[0]).days / 365.25
eo = e0[e0.index >= cut]
L.append(f"| 對照：0050 買進持有 | — | {(e0.iloc[-1]/e0.iloc[0])**(1/yrs)-1:+.1%} | — | "
         f"{(eo.iloc[-1]/eo.iloc[0])**(365.25/(eo.index[-1]-eo.index[0]).days)-1:+.1%} | {(e0/e0.cummax()-1).min():.1%} | — |")
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L[-12:]))
