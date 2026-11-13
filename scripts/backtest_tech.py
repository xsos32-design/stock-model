"""Event-study backtest of well-known technical entry signals on Taiwan common stocks.

Rules applied to every strategy (identical, no per-strategy tuning):
- Signal on close of day t, entry at OPEN of day t+1 (no look-ahead).
- Skip entries where t+1 opens >= +9.5% vs prior close (limit-up, usually unbuyable).
- Universe filter on day t: 20-day avg traded value >= NT$20M, close >= 10,
  20-day return <= +50% (user rule: avoid stocks up >50%).
- Cost per round trip: buy fee 0.1425% + sell fee 0.1425% + tax 0.3% = 0.585%.
- Exit A: fixed hold, sell at close of day t+5 (1 trading week).
- Exit B: Wei's rules - stop at -2% from entry (intraday low), else sell at close the
  first day close < MA5, else sell at close of day t+5.
- Exit C (Connors only): original exit, close > MA5, max 10 days.
Split: first 60% of dates = in-sample (IS), last 40% = out-of-sample (OOS).
Data: FinMind TaiwanStockPriceAdj (adjusted for dividends/splits/capital reduction).
"""
import sys, glob, json
import numpy as np
import pandas as pd

COST = 0.001425 * 2 + 0.003
DATA = sys.argv[1] if len(sys.argv) > 1 else "data/price_adj"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/tech_backtest.md"


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def kd(df, n=9):
    lo = df.low.rolling(n).min(); hi = df.high.rolling(n).max()
    rsv = ((df.close - lo) / (hi - lo).replace(0, np.nan) * 100).fillna(50)
    k = np.empty(len(df)); d = np.empty(len(df)); pk = pd_ = 50.0
    for i, r in enumerate(rsv.values):
        pk = pk * 2 / 3 + r / 3; pd_ = pd_ * 2 / 3 + pk / 3; k[i] = pk; d[i] = pd_
    return pd.Series(k, df.index), pd.Series(d, df.index)


def signals(df):
    c, v = df.close, df.volume
    ma5, ma10, ma20, ma200 = (c.rolling(n).mean() for n in (5, 10, 20, 200))
    sd20 = c.rolling(20).std(ddof=0)
    up, lo = ma20 + 2 * sd20, ma20 - 2 * sd20
    bw = (up - lo) / ma20
    k, d = kd(df)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    hist = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()
    prev = lambda s: s.shift(1)
    align = (c > ma5) & (ma5 > ma10) & (ma10 > ma20)
    s = {
        "Connors RSI(2)<5 且站上200日線": (rsi(c, 2) < 5) & (c > ma200),
        "唐奇安/海龜 20日新高突破": c > df.high.rolling(20).max().shift(1),
        "布林 收盤突破上軌(20,2)": (c > up) & (prev(c) <= prev(up)),
        "布林 收窄後突破上軌": (c > up) & (bw.rolling(5).min() <= bw.rolling(120).min() * 1.05),
        "布林 跌破下軌(反轉)": (c < lo) & (prev(c) >= prev(lo)),
        "KD 低檔黃金交叉(K<20)": (k > d) & (prev(k) <= prev(d)) & (prev(k) < 20),
        "KD 黃金交叉(任意位置)": (k > d) & (prev(k) <= prev(d)),
        "MACD 柱狀體翻正": (hist > 0) & (prev(hist) <= 0),
        "均線多頭排列 5>10>20 首日": align & ~align.shift(1, fill_value=False),
        "站上5日線(現行規則)": (c > ma5) & (prev(c) <= prev(ma5)),
        "帶量長紅 量>2倍20日均量且漲>3%": (v > 2 * v.rolling(20).mean().shift(1)) & (c / prev(c) - 1 > 0.03),
        "基準: 任一合格日隨機進場": pd.Series(True, index=df.index),
    }
    return s, ma5


def trades_for_stock(df):
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 210:
        return []
    sig, ma5 = signals(df)
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    val20 = df.value.rolling(20).mean().values
    ret20 = (df.close / df.close.shift(20) - 1).values
    ok = (val20 >= 2e7) & (c >= 10) & (np.nan_to_num(ret20, nan=9) <= 0.5)
    ma5v = ma5.values
    n = len(df); out = []
    dates = df.date.values
    for name, s in sig.items():
        idx = np.where(s.fillna(False).values & ok)[0]
        for t in idx:
            e = t + 1
            if e + 5 >= n or c[t] <= 0 or o[e] <= 0:
                continue
            if o[e] >= c[t] * 1.095:
                continue
            entry = o[e]
            # Exit A: close of t+5
            ra = c[t + 5] / entry - 1
            # Exit B: Wei rules
            stop = entry * 0.98; rb = None
            for j in range(e, t + 6):
                if l[j] <= stop:
                    px = o[j] if (j > e and o[j] < stop) else stop
                    rb = px / entry - 1; break
                if c[j] < ma5v[j]:
                    rb = c[j] / entry - 1; break
            if rb is None:
                rb = c[t + 5] / entry - 1
            row = {"strategy": name, "date": dates[t], "A": ra - COST, "B": rb - COST}
            if name.startswith("Connors"):
                rc = None
                for j in range(e, min(e + 10, n)):
                    if c[j] > ma5v[j]:
                        rc = c[j] / entry - 1; break
                if rc is None:
                    rc = c[min(e + 9, n - 1)] / entry - 1
                row["C"] = rc - COST
            out.append(row)
    return out


def stats(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return dict(n=0, win=np.nan, avg=np.nan, med=np.nan, pf=np.nan, t=np.nan)
    pos, neg = x[x > 0].sum(), -x[x < 0].sum()
    return dict(n=len(x), win=(x > 0).mean(), avg=x.mean(), med=np.median(x),
                pf=pos / neg if neg > 0 else np.inf, t=x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan)


def main():
    files = sorted(glob.glob(f"{DATA}/*.parquet"))
    px = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    px["date"] = pd.to_datetime(px["date"])
    stocks = px[px.stock_id.str.fullmatch(r"[1-9]\d{3}")]
    allt = []
    for sid, g in stocks.groupby("stock_id"):
        allt += trades_for_stock(g)
    T = pd.DataFrame(allt)
    dates = np.sort(stocks.date.unique())
    cut = dates[int(len(dates) * 0.6)]
    T["oos"] = T.date >= cut

    rows = []
    for name, g in T.groupby("strategy"):
        for ex in ("A", "B", "C"):
            if ex not in g or g[ex].isna().all():
                continue
            a, i, o = stats(g[ex]), stats(g.loc[~g.oos, ex]), stats(g.loc[g.oos, ex])
            # t-stat on per-date average (signals on the same day are correlated, not independent)
            o["t"] = stats(g.loc[g.oos].groupby("date")[ex].mean())["t"]
            rows.append(dict(strategy=name, exit=ex, **{f"all_{k}": v for k, v in a.items()},
                             is_avg=i["avg"], is_win=i["win"], oos_n=o["n"], oos_avg=o["avg"], oos_win=o["win"], oos_t=o["t"]))
    R = pd.DataFrame(rows)

    # 0050 buy & hold reference (adjusted, over the same window)
    ref = px[px.stock_id == "0050"].sort_values("date")
    bh = ""
    if len(ref):
        yrs = (ref.date.iloc[-1] - ref.date.iloc[0]).days / 365.25
        tot = ref.close.iloc[-1] / ref.close.iloc[0] - 1
        wk = ref.close.pct_change(5).dropna()
        bh = (f"0050 買進持有 {ref.date.iloc[0]:%Y-%m-%d}~{ref.date.iloc[-1]:%Y-%m-%d}：總報酬 {tot:+.1%}，"
              f"年化 {(1+tot)**(1/yrs)-1:+.1%}；任意 5 日持有平均 {wk.mean():+.2%}、上漲機率 {(wk>0).mean():.1%}（未扣成本）")

    basemap = R[R.strategy.str.startswith("基準")].set_index("exit").oos_avg
    base = R[(R.strategy.str.startswith("基準")) & (R.exit == "A")].iloc[0]
    lab = {"A": "持有5日", "B": "Wei出場規則", "C": "Connors原始出場"}
    R = R.sort_values(["oos_avg"], ascending=False)
    L = ["# 技術指標回測排名（扣除交易成本 0.585%）", "",
         f"- 資料：FinMind 還原股價，{stocks.stock_id.nunique()} 檔上市櫃普通股（含期間下市），"
         f"{pd.Timestamp(dates[0]):%Y-%m-%d} ~ {pd.Timestamp(dates[-1]):%Y-%m-%d}",
         f"- 樣本內 / 樣本外切點：{pd.Timestamp(cut):%Y-%m-%d}（排名依樣本外平均報酬）",
         "- 訊號為當日收盤判定、隔日開盤進場；隔日開盤漲停不計入；流動性：20日均成交值≥2,000萬、股價≥10、20日漲幅≤50%",
         f"- {bh}" if bh else "",
         f"- 基準（任一合格日隨機進場、持有5日）：樣本外平均 {base.oos_avg:+.2%}、勝率 {base.oos_win:.1%}",
         "- t 值以『同日訊號取平均』後計算（同一天多檔訊號高度相關，不能當獨立樣本）；t > 2 才算統計上可能不是運氣",
         "- 『贏基準』= 樣本外平均報酬高於同一出場方式的隨機進場",
         "", "| 排名 | 策略 | 出場 | 樣本外筆數 | 樣本外勝率 | 樣本外平均 | 樣本外t值 | 樣本內平均 | 全期勝率 | 全期平均 | 獲利因子 | 贏基準 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, x in enumerate(R.itertuples(), 1):
        beat = "—" if x.strategy.startswith("基準") else ("✅" if x.oos_avg > basemap.get(x.exit if x.exit in basemap else "A") else "❌")
        L.append(f"| {r} | {x.strategy} | {lab[x.exit]} | {x.oos_n:,} | {x.oos_win:.1%} | {x.oos_avg:+.2%} | {x.oos_t:.1f} | "
                 f"{x.is_avg:+.2%} | {x.all_win:.1%} | {x.all_avg:+.2%} | {x.all_pf:.2f} | {beat} |")
    # yearly for top rows
    T["year"] = pd.to_datetime(T.date).dt.year
    yr = T.groupby(["strategy", "year"]).A.mean().unstack()
    L += ["", "## 各年度平均報酬（持有5日，扣成本）", "", "| 策略 | " + " | ".join(map(str, yr.columns)) + " |",
          "|---|" + "---|" * len(yr.columns)]
    for s_, r_ in yr.iterrows():
        L.append(f"| {s_} | " + " | ".join(f"{v:+.2%}" if pd.notna(v) else "" for v in r_) + " |")
    import os; os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(l for l in L if l is not None) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
