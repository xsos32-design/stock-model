"""Does the previous US session move Taiwan, and does it change the strategy's win rate?
US date d closes ~04:00-05:00 Taipei time of d+1, so for TW trading day t the 'previous US night'
is the latest US session dated strictly before t (known before the TW open of t)."""
import sys, glob, json
import numpy as np, pandas as pd

PRICE = sys.argv[1]; US = sys.argv[2]; OUT = sys.argv[3]
COST = 0.00585
raw = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{PRICE}/*.parquet"))]); raw["date"] = pd.to_datetime(raw.date)
tw = raw[raw.stock_id == "TAIEX"].set_index("date").sort_index()
us = {}
for k, f in {"S&P500": "^GSPC", "Nasdaq": "^IXIC", "費半SOX": "^SOX", "台積電ADR": "TSM"}.items():
    d = pd.DataFrame(json.load(open(f"{US}/{f}.json"))["data"]); d["date"] = pd.to_datetime(d.date)
    s = d.set_index("date").Close.sort_index(); us[k] = s.pct_change()
U = pd.DataFrame(us)
# align: for each TW date t, take US row with date < t
idx = U.index.searchsorted(tw.index, side="left") - 1
prevus = U.iloc[np.clip(idx, 0, None)].set_index(tw.index)
prevus[idx < 0] = np.nan

tw["gap"] = tw.open / tw.close.shift(1) - 1           # overnight gap
tw["intra"] = tw.close / tw.open - 1                  # open -> close
tw["cc"] = tw.close.pct_change()
L = ["# 前一晚美股對台股與策略勝率的影響", "",
     f"- 期間 {tw.index[0]:%Y-%m-%d} ~ {tw.index[-1]:%Y-%m-%d}；美股資料：FinMind USStockPrice；『前一晚』= 台股當日開盤前最後一個美股交易日", "",
     "## 1. 前一晚美股漲跌 vs 台股加權指數當天表現（相關係數）", "",
     "| 美股指標 | 對 台股開盤跳空 | 對 台股開盤→收盤 | 對 台股收盤→收盤 |", "|---|---|---|---|"]
for k in U.columns:
    L.append(f"| {k} | {prevus[k].corr(tw.gap):.2f} | {prevus[k].corr(tw.intra):.2f} | {prevus[k].corr(tw.cc):.2f} |")

sox = prevus["費半SOX"]
bins = [-np.inf, -0.03, -0.01, 0, 0.01, 0.03, np.inf]
lab = ["跌>3%", "跌1~3%", "跌0~1%", "漲0~1%", "漲1~3%", "漲>3%"]
b = pd.cut(sox, bins, labels=lab)
L += ["", "## 2. 前一晚費半漲跌分組 → 台股加權當天", "", "| 前一晚費半 | 天數 | 平均開盤跳空 | 平均開盤→收盤 | 平均收盤→收盤 | 收盤上漲機率 |", "|---|---|---|---|---|---|"]
for l in lab:
    m = b == l
    L.append(f"| {l} | {m.sum()} | {tw.gap[m].mean():+.2%} | {tw.intra[m].mean():+.2%} | {tw.cc[m].mean():+.2%} | {(tw.cc[m] > 0).mean():.0%} |")

# ---- strategy events (same definition as monthly model) ----
px = raw[raw.stock_id.str.fullmatch(r"[1-9]\d{3}")].sort_values(["stock_id", "date"]).reset_index(drop=True)
g = px.groupby("stock_id")
px["hi250"] = g.high.transform(lambda s: s.rolling(250).max()); px["val20"] = g.value.transform(lambda s: s.rolling(20).mean())
px["prevc"] = g.close.shift(1)
for h in (1, 5, 20): px[f"f{h}"] = g.close.shift(-h) / px.close - 1 - COST
px["o1"] = g.open.shift(-1) / px.close - 1                 # overnight after entry
m60 = tw.close.rolling(60).mean(); px = px.join((tw.close > m60).rename("mkt_up"), on="date")
px = px.join(prevus.add_prefix("us_"), on="date")
# US night AFTER the entry day t (unknown at entry, shown for understanding only)
nxt = U.iloc[np.clip(U.index.searchsorted(tw.index, side="left"), 0, len(U) - 1)].set_index(tw.index)
px = px.join(nxt.add_prefix("usn_"), on="date")
sig = (px.close >= 0.98 * px.hi250) & px.mkt_up.fillna(False) & (px.val20 >= 2e7) & (px.close >= 10) & (px.close < px.prevc * 1.095)
E = px[sig].dropna(subset=["f20"])
cut = np.sort(px.date.unique())[int(px.date.nunique() * 0.6)]
L += ["", "## 3. 策略（大盤在60日線上＋接近一年新高，收盤買）依『進場當天之前那一晚』費半表現分組", "",
      "進場當天收盤前就知道的資訊，可以直接拿來當進場條件。", "",
      "| 前一晚費半 | 筆數 | 持有20日 平均 | 持有20日 勝率 | 樣本外 平均 | 樣本外 勝率 | 持有5日 平均 | 持有5日 勝率 |", "|---|---|---|---|---|---|---|---|"]
eb = pd.cut(E.us_費半SOX, bins, labels=lab)
for l in lab:
    m = eb == l; o = m & (E.date >= cut)
    if m.sum() < 50: continue
    L.append(f"| {l} | {m.sum():,} | {E.f20[m].mean():+.2%} | {(E.f20[m] > 0).mean():.1%} | {E.f20[o].mean():+.2%} | {(E.f20[o] > 0).mean():.1%} | {E.f5[m].mean():+.2%} | {(E.f5[m] > 0).mean():.1%} |")
L.append(f"| 全部 | {len(E):,} | {E.f20.mean():+.2%} | {(E.f20 > 0).mean():.1%} | {E.f20[E.date >= cut].mean():+.2%} | {(E.f20[E.date >= cut] > 0).mean():.1%} | {E.f5.mean():+.2%} | {(E.f5 > 0).mean():.1%} |")

# combined filter: US prev night + TW same-day reaction
E = E.assign(twday=E.date.map(tw.cc))
L += ["", "## 4. 進場前的組合判斷：前一晚美股 × 台股當天大盤", "",
      "| 條件 | 筆數 | 20日平均 | 20日勝率 | 樣本外平均 | 樣本外勝率 |", "|---|---|---|---|---|---|"]
conds = {"前一晚費半跌、台股當天仍收紅": (E.us_費半SOX < 0) & (E.twday > 0),
         "前一晚費半跌、台股當天收黑": (E.us_費半SOX < 0) & (E.twday <= 0),
         "前一晚費半漲、台股當天收紅": (E.us_費半SOX >= 0) & (E.twday > 0),
         "前一晚費半漲、台股當天收黑": (E.us_費半SOX >= 0) & (E.twday <= 0),
         "前一晚 S&P500 跌超過1%": E["us_S&P500"] < -0.01,
         "前一晚台積電ADR 跌超過2%": E.us_台積電ADR < -0.02}
for k, m in conds.items():
    o = m & (E.date >= cut)
    L.append(f"| {k} | {m.sum():,} | {E.f20[m].mean():+.2%} | {(E.f20[m] > 0).mean():.1%} | {E.f20[o].mean():+.2%} | {(E.f20[o] > 0).mean():.1%} |")

# overnight after entry: how much the next US night matters (for understanding the risk of holding)
L += ["", "## 5. 進場『之後』那一晚美股（進場時還不知道，只用來理解持有風險）", "",
      "| 進場後那晚費半 | 筆數 | 隔天開盤跳空平均 | 持有20日平均 | 持有20日勝率 |", "|---|---|---|---|---|"]
nb = pd.cut(E.usn_費半SOX, bins, labels=lab)
for l in lab:
    m = nb == l
    if m.sum() < 50: continue
    L.append(f"| {l} | {m.sum():,} | {E.o1[m].mean():+.2%} | {E.f20[m].mean():+.2%} | {(E.f20[m] > 0).mean():.1%} |")
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))
