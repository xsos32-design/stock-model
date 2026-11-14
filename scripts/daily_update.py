"""Daily update for the website (runs in GitHub Actions after the TW close and again before the open).
Sources: TWSE / TPEx OpenAPI (official, free) for prices, TAIEX, revenue, income statements, disposition;
FinMind USStockPrice for the US night. Writes docs/data/*.json.
Usage: python scripts/daily_update.py [--init]"""
import os, sys, json, re, time, datetime as dt, urllib.request, urllib.parse
import numpy as np, pandas as pd

LIVE, DOCS = "data/live", "docs/data"
os.makedirs(LIVE, exist_ok=True); os.makedirs(f"{DOCS}/history", exist_ok=True)
TOKEN = os.environ.get("FINMIND_TOKEN", "")
COST = 0.001425 * 2 + 0.003
UA = {"User-Agent": "Mozilla/5.0 stock-model"}

def get(url, headers=UA, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as r:
                return json.loads(r.read().decode("utf-8-sig"))
        except Exception as e:
            print("fetch fail", url, e, flush=True); time.sleep(10 * (k + 1))
    return None

def roc(s):
    s = re.sub(r"\D", "", str(s))
    if len(s) == 8: return pd.Timestamp(s)
    if len(s) == 7: return pd.Timestamp(int(s[:3]) + 1911, int(s[3:5]), int(s[5:7]))
    return pd.NaT

def f(x):
    try: return float(str(x).replace(",", "").replace("+", ""))
    except: return np.nan

# ---------------- init from the 5-year research data ----------------
if "--init" in sys.argv:
    import glob
    px = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob("data/price_adj/*.parquet"))])
    px["date"] = pd.to_datetime(px.date)
    keep = np.sort(px.date.unique())[-320:]
    px[px.stock_id.str.fullmatch(r"[1-9]\d{3}") & px.date.isin(keep)].to_parquet(f"{LIVE}/prices.parquet", index=False)
    px[px.stock_id == "TAIEX"][["date", "close"]].to_csv(f"{LIVE}/taiex.csv", index=False)
    r = pd.read_parquet("data/factors/revenue.parquet")
    r["avail"] = pd.to_datetime(r.date) + pd.Timedelta(days=10)
    r[["stock_id", "revenue_year", "revenue_month", "revenue", "avail"]].to_parquet(f"{LIVE}/revenue.parquet", index=False)
    fi = pd.read_parquet("data/factors/fin.parquet")
    fi["gm"] = fi.GrossProfit / fi.Revenue
    fi[["date", "stock_id", "gm"]].to_parquet(f"{LIVE}/fin.parquet", index=False)
    print("init done")

px = pd.read_parquet(f"{LIVE}/prices.parquet"); px["date"] = pd.to_datetime(px.date)
taiex = pd.read_csv(f"{LIVE}/taiex.csv", parse_dates=["date"])
rev = pd.read_parquet(f"{LIVE}/revenue.parquet")
fin = pd.read_parquet(f"{LIVE}/fin.parquet"); fin["date"] = pd.to_datetime(fin.date)
names = pd.read_csv("data/universe.csv", dtype=str).set_index("stock_id")
names = names[~names.index.duplicated()]
mkt_of = names["type"].to_dict(); name_of = names["stock_name"].to_dict(); ind_of = names["industry_category"].to_dict()

# ---------------- daily prices (TWSE + TPEx) ----------------
rows = []
tw = get("https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL") or []
for x in tw:
    c = x.get("Code", "")
    if re.fullmatch(r"[1-9]\d{3}", c):
        rows.append(dict(date=roc(x["Date"]), stock_id=c, open=f(x["OpeningPrice"]), high=f(x["HighestPrice"]), low=f(x["LowestPrice"]),
                         close=f(x["ClosingPrice"]), volume=f(x["TradeVolume"]), value=f(x["TradeValue"])))
        name_of.setdefault(c, x.get("Name")); mkt_of.setdefault(c, "twse")
tp = get("https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes") or []
for x in tp:
    c = x.get("SecuritiesCompanyCode", "")
    if re.fullmatch(r"[1-9]\d{3}", c):
        rows.append(dict(date=roc(x["Date"]), stock_id=c, open=f(x["Open"]), high=f(x["High"]), low=f(x["Low"]),
                         close=f(x["Close"]), volume=f(x["TradingShares"]), value=f(x["TransactionAmount"])))
        name_of.setdefault(c, x.get("CompanyName")); mkt_of.setdefault(c, "tpex")
new = pd.DataFrame(rows).dropna(subset=["date", "close"])
new = new[new.close > 0]
if len(new):
    px = pd.concat([px[~px.date.isin(new.date.unique())], new]).drop_duplicates(["date", "stock_id"], keep="last")
    keep = np.sort(px.date.unique())[-320:]
    px = px[px.date.isin(keep)]
    px.to_parquet(f"{LIVE}/prices.parquet", index=False)

mi = get("https://openapi.twse.com.tw/v1/exchangeReport/MI_INDEX") or []
for x in mi:
    if x.get("指數") == "發行量加權股價指數":
        d = roc(x["日期"]); c = f(x["收盤指數"])
        taiex = pd.concat([taiex[taiex.date != d], pd.DataFrame([{"date": d, "close": c}])]).sort_values("date")
        taiex.to_csv(f"{LIVE}/taiex.csv", index=False)

# ---------------- monthly revenue ----------------
have = set(zip(rev.stock_id, rev.revenue_year, rev.revenue_month)); add = []
for url in ("https://openapi.twse.com.tw/v1/opendata/t187ap05_P", "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap05_O"):
    for x in get(url) or []:
        c = x.get("公司代號", ""); ym = re.sub(r"\D", "", str(x.get("資料年月", "")))
        if not re.fullmatch(r"[1-9]\d{3}", c) or len(ym) < 4: continue
        y, m = int(ym[:-2]) + 1911, int(ym[-2:])
        rv = f(x.get("營業收入-當月營收")) * 1000
        if (c, y, m) not in have:
            have.add((c, y, m)); add.append(dict(stock_id=c, revenue_year=y, revenue_month=m, revenue=rv, avail=roc(x.get("出表日期"))))
if add: rev = pd.concat([rev, pd.DataFrame(add)], ignore_index=True)
rev["avail"] = pd.to_datetime(rev.avail)
rev.to_parquet(f"{LIVE}/revenue.parquet", index=False)

# ---------------- quarterly income statement (gross margin) ----------------
haveq = set(zip(fin.stock_id, fin.date)); addq = []
for url in ("https://openapi.twse.com.tw/v1/opendata/t187ap06_L_ci", "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap06_O_ci"):
    for x in get(url) or []:
        c = x.get("公司代號", "")
        try: y, q = int(x["年度"]) + 1911, int(x["季別"])
        except: continue
        if not re.fullmatch(r"[1-9]\d{3}", c): continue
        d = pd.Timestamp(y, q * 3, 1) + pd.offsets.MonthEnd(0)
        gp = f(x.get("營業毛利（毛損）淨額", x.get("營業毛利（毛損）"))); rv = f(x.get("營業收入"))
        if rv and rv == rv and (c, d) not in haveq:
            haveq.add((c, d)); addq.append(dict(date=d, stock_id=c, gm=gp / rv))
if addq: fin = pd.concat([fin, pd.DataFrame(addq)], ignore_index=True)
fin.to_parquet(f"{LIVE}/fin.parquet", index=False)

# ---------------- disposition (current) ----------------
disposed = {}
for x in get("https://openapi.twse.com.tw/v1/announcement/punish") or []:
    disposed[x.get("Code")] = f"{x.get('DispositionPeriod','')} {x.get('DispositionMeasures','')}"
for x in get("https://www.tpex.org.tw/openapi/v1/tpex_disposal_information") or []:
    disposed[x.get("SecuritiesCompanyCode")] = x.get("DispositionPeriod", "")
notice = {x.get("Code"): x.get("TradingInfoForAttention", "") for x in (get("https://openapi.twse.com.tw/v1/announcement/notice") or [])}

# ---------------- US night (FinMind, free tier) ----------------
us = {}
for k, sid in {"費半SOX": "^SOX", "S&P500": "^GSPC", "Nasdaq": "^IXIC", "台積電ADR": "TSM"}.items():
    q = urllib.parse.urlencode({"dataset": "USStockPrice", "data_id": sid, "start_date": (dt.date.today() - dt.timedelta(days=15)).isoformat()})
    j = get("https://api.finmindtrade.com/api/v4/data?" + q, {**UA, "Authorization": f"Bearer {TOKEN}"}) or {}
    d = pd.DataFrame(j.get("data") or [])
    if len(d) >= 2:
        d = d.sort_values("date"); us[k] = dict(date=d.date.iloc[-1], close=float(d.Close.iloc[-1]), chg=float(d.Close.iloc[-1] / d.Close.iloc[-2] - 1))
    time.sleep(2)
gate_reasons = []
if us.get("費半SOX", {}).get("chg", 0) <= -0.03: gate_reasons.append("前一晚費半跌 3% 以上")
if us.get("台積電ADR", {}).get("chg", 0) <= -0.02: gate_reasons.append("前一晚台積電 ADR 跌 2% 以上")

# ---------------- signals ----------------
px = px.sort_values(["stock_id", "date"]).reset_index(drop=True)
g = px.groupby("stock_id")
px["hi250"] = g.high.transform(lambda s: s.rolling(250, min_periods=200).max())
px["val20"] = g.value.transform(lambda s: s.rolling(20).mean())
px["ma20"] = g.close.transform(lambda s: s.rolling(20).mean())
px["prevc"] = g.close.shift(1)
px["r120"] = g.close.pct_change(120); px["r20"] = g.close.pct_change(20)
T = px.date.max(); cur = px[px.date == T].set_index("stock_id")
tdates = np.sort(px.date.unique())
tx = taiex.set_index("date").close; ma60 = tx.rolling(60).mean()
mkt_up = bool(tx.iloc[-1] > ma60.iloc[-1])

rev = rev.sort_values(["stock_id", "revenue_year", "revenue_month"])
rev["ym"] = rev.revenue_year * 12 + rev.revenue_month
ly = rev[["stock_id", "ym", "revenue"]].assign(ym=lambda x: x.ym + 12).rename(columns={"revenue": "rev_ly"})
rv = rev.merge(ly, on=["stock_id", "ym"], how="left")
rv = rv[pd.to_datetime(rv.avail) <= T + pd.Timedelta(days=1)]
gr = rv.groupby("stock_id")
rv["yoy"] = rv.revenue / rv.rev_ly - 1
rv["yoy3"] = gr.revenue.transform(lambda s: s.rolling(3).sum()) / gr.rev_ly.transform(lambda s: s.rolling(3).sum()) - 1
rv["yoy3_prev"] = gr.yoy3.shift(3)
lastrev = rv.groupby("stock_id").tail(1).set_index("stock_id")
fin = fin.sort_values(["stock_id", "date"]); fin["gm_ly"] = fin.groupby("stock_id").gm.shift(4)
dl = {3: (5, 15), 6: (8, 14), 9: (11, 14), 12: (3, 31)}
fin["avail"] = [pd.Timestamp(d.year + (d.month == 12), *dl[d.month]) if d.month in dl else pd.NaT for d in fin.date]
lastfin = fin[fin.avail <= T].groupby("stock_id").tail(1).set_index("stock_id")

out = []
for sid_, r in cur.iterrows():
    lr = lastrev.loc[sid_] if sid_ in lastrev.index else None
    lf = lastfin.loc[sid_] if sid_ in lastfin.index else None
    facts = {
        "月營收年增>20%": bool(lr is not None and lr.yoy > 0.2),
        "近3月營收年增>15%": bool(lr is not None and lr.yoy3 > 0.15),
        "營收成長加速": bool(lr is not None and lr.yoy3 > lr.yoy3_prev),
        "毛利率較去年同季提升": bool(lf is not None and lf.gm > lf.gm_ly),
        "120日漲幅>30%": bool(r.r120 > 0.3),
    }
    checks = {
        "流動性(20日均成交值≥2千萬)": bool(r.val20 >= 2e7), "股價≥10元": bool(r.close >= 10),
        "非處置股": sid_ not in disposed, "收盤未漲停(買得到)": bool(r.close < r.prevc * 1.095),
        "接近一年新高(≥98%)": bool(r.hi250 == r.hi250 and r.close >= 0.98 * r.hi250),
    }
    out.append(dict(id=sid_, name=name_of.get(sid_, ""), mkt=mkt_of.get(sid_, ""), ind=ind_of.get(sid_, ""),
                    close=round(float(r.close), 2), pct_hi=round(float(r.close / r.hi250), 4) if r.hi250 == r.hi250 else None,
                    r20=None if r.r20 != r.r20 else round(float(r.r20), 4), r120=None if r.r120 != r.r120 else round(float(r.r120), 4),
                    val20=None if r.val20 != r.val20 else round(float(r.val20) / 1e8, 2),
                    yoy=None if lr is None or lr.yoy != lr.yoy else round(float(lr.yoy), 4),
                    yoy3=None if lr is None or lr.yoy3 != lr.yoy3 else round(float(lr.yoy3), 4),
                    rev_month=None if lr is None else f"{int(lr.revenue_year)}-{int(lr.revenue_month):02d}",
                    score=int(sum(facts.values())), factors=facts, checks=checks,
                    signal=all(checks.values()), disposed=disposed.get(sid_), notice=notice.get(sid_)))
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))): return None
    if isinstance(o, (np.floating,)): return None if (o != o or np.isinf(o)) else float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
out = clean(out)
S = pd.DataFrame(out)
cand = S[S.signal].sort_values(["score", "r120"], ascending=[False, False])
i = np.searchsorted(tdates, T)
exit_est = (pd.Timestamp(T) + pd.offsets.BDay(20)).strftime("%Y-%m-%d")

today = dict(date=T.strftime("%Y-%m-%d"), generated=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
             taiex=dict(close=round(float(tx.iloc[-1]), 2), ma60=round(float(ma60.iloc[-1]), 2), chg=round(float(tx.iloc[-1] / tx.iloc[-2] - 1), 4),
                        series=[[d.strftime("%Y-%m-%d"), round(float(c), 2), None if m != m else round(float(m), 2)] for d, c, m in zip(tx.index[-120:], tx.values[-120:], ma60.values[-120:])]),
             mkt_up=mkt_up, us=us, gate=gate_reasons, can_enter=mkt_up and not gate_reasons,
             exit_est=exit_est, n_cand=len(cand), candidates=clean(cand.to_dict("records")),
             disposed=[{"id": k, "name": name_of.get(k, ""), "info": v} for k, v in disposed.items() if re.fullmatch(r"[1-9]\d{3}", str(k))])
json.dump(today, open(f"{DOCS}/today.json", "w"), ensure_ascii=False, default=str)
json.dump(clean(S.to_dict("records")), open(f"{DOCS}/stocks.json", "w"), ensure_ascii=False, default=str)
json.dump(dict(date=today["date"], can_enter=today["can_enter"], mkt_up=mkt_up, gate=gate_reasons,
               picks=[dict(id=x["id"], name=x["name"], close=x["close"], score=x["score"]) for x in cand.head(20).to_dict("records")]),
          open(f"{DOCS}/history/{today['date']}.json", "w"), ensure_ascii=False)

# ---------------- live track record: candidates archived >= 20 trading days ago ----------------
import glob
recs = []
closes = px.pivot(index="date", columns="stock_id", values="close")
for p in sorted(glob.glob(f"{DOCS}/history/*.json")):
    h = json.load(open(p)); d0 = pd.Timestamp(h["date"])
    if d0 not in closes.index: continue
    k = closes.index.get_loc(d0)
    for x in h["picks"][:10]:
        if x["id"] not in closes: continue
        k2 = min(k + 20, len(closes) - 1)
        ret = closes[x["id"]].iloc[k2] / x["close"] - 1 - COST
        recs.append(dict(date=h["date"], id=x["id"], name=x["name"], score=x["score"], can_enter=h["can_enter"],
                         days=int(k2 - k), done=bool(k2 - k >= 20), ret=round(float(ret), 4)))
R = pd.DataFrame(recs)
summ = {}
if len(R):
    dn = R[R.done & R.can_enter]
    summ = dict(n_done=int(len(dn)), win=None if not len(dn) else round(float((dn.ret > 0).mean()), 4),
                avg=None if not len(dn) else round(float(dn.ret.mean()), 4), n_open=int((~R.done).sum()))
json.dump(dict(summary=summ, records=R.to_dict("records") if len(R) else []), open(f"{DOCS}/track.json", "w"), ensure_ascii=False)
print("updated", today["date"], "candidates", len(cand), "can_enter", today["can_enter"], "us", us)
