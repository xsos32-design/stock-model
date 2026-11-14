"""Backfill 5y factor data per stock from FinMind (Backer level), compacted to small parquet files.
Usage: python scripts/fetch_factors.py <name>   name in: revenue inst holding margin foreign per fin disposition"""
import os, sys, json, time, re, datetime as dt, urllib.request, urllib.parse
import pandas as pd

TOKEN = os.environ["FINMIND_TOKEN"]
API = "https://api.finmindtrade.com/api/v4/data"
HDR = {"Authorization": f"Bearer {TOKEN}", "User-Agent": "stock-model"}
START = os.environ.get("START") or (dt.date.today() - dt.timedelta(days=365 * 5 + 40)).isoformat()
END = os.environ.get("END", "")
OUTDIR = os.environ.get("OUTDIR", "data/factors")
IDS = os.environ.get("IDS", "data/universe.csv")
GAP = 2.4
NAME = sys.argv[1]
DS = {"revenue": "TaiwanStockMonthRevenue", "inst": "TaiwanStockInstitutionalInvestorsBuySell",
      "holding": "TaiwanStockHoldingSharesPer", "margin": "TaiwanStockMarginPurchaseShortSale",
      "foreign": "TaiwanStockShareholding", "per": "TaiwanStockPER", "fin": "TaiwanStockFinancialStatements",
      "disposition": "TaiwanStockDispositionSecuritiesPeriod"}

def call(dataset, **q):
    url = API + "?" + urllib.parse.urlencode({"dataset": dataset, **q})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=120) as r:
                return json.loads(r.read()).get("data") or []
        except urllib.error.HTTPError as e:
            if e.code == 402: print("quota, sleep 10m", flush=True); time.sleep(600); continue
            print("HTTP", e.code, dataset, q.get("data_id"), e.read()[:150], flush=True)
            if e.code == 400: return []
        except Exception as e:
            print("err", dataset, q.get("data_id"), e, flush=True)
        time.sleep(15 * (attempt + 1))
    return []

def num(s):
    m = re.findall(r"[\d,]+", str(s)); return int(m[0].replace(",", "")) if m else None

def compact(df):
    if NAME == "revenue":
        return df[["date", "stock_id", "revenue", "revenue_year", "revenue_month"]]
    if NAME == "inst":
        df["net"] = df.buy - df.sell
        p = df.pivot_table(index=["date", "stock_id"], columns="name", values="net", aggfunc="sum").reset_index()
        return p
    if NAME == "holding":
        df["lo"] = df.HoldingSharesLevel.map(num)
        big = df[df.lo.fillna(0) >= 1_000_001].groupby(["date", "stock_id"]).percent.sum().rename("pct_1000up")
        b400 = df[df.lo.fillna(0) >= 400_001].groupby(["date", "stock_id"]).percent.sum().rename("pct_400up")
        small = df[df.lo.notna() & (df.lo < 50_001)].groupby(["date", "stock_id"]).percent.sum().rename("pct_lt50")
        ppl = df[df.lo.notna()].groupby(["date", "stock_id"]).people.sum().rename("people")
        return pd.concat([big, b400, small, ppl], axis=1).reset_index()
    if NAME == "margin":
        return df[["date", "stock_id", "MarginPurchaseTodayBalance", "ShortSaleTodayBalance", "MarginPurchaseLimit"]]
    if NAME == "foreign":
        return df[["date", "stock_id", "ForeignInvestmentSharesRatio", "NumberOfSharesIssued"]]
    if NAME == "per":
        return df[["date", "stock_id", "PER", "PBR", "dividend_yield"]]
    if NAME == "fin":
        keep = {"EPS", "Revenue", "GrossProfit", "OperatingIncome", "IncomeAfterTaxes"}
        d = df[df["type"].isin(keep)]
        return d.pivot_table(index=["date", "stock_id"], columns="type", values="value", aggfunc="last").reset_index()
    return df

os.makedirs(OUTDIR, exist_ok=True)
if NAME == "disposition":
    rows = []
    s = dt.date.fromisoformat(START)
    STOP = dt.date.fromisoformat(END) if END else dt.date.today()
    while s < STOP:
        e = min(s + dt.timedelta(days=90), STOP)
        r = call(DS[NAME], start_date=s.isoformat(), end_date=e.isoformat()); rows += r
        print(s, len(r), flush=True); s = e + dt.timedelta(days=1); time.sleep(GAP)
    df = pd.DataFrame(rows)
else:
    ids = pd.read_csv(IDS, dtype=str).stock_id.tolist()
    frames, empty, levels = [], [], set()
    for i, sid in enumerate(ids):
        r = call(DS[NAME], data_id=sid, start_date=START, **({"end_date": END} if END else {}))
        if r:
            d = pd.DataFrame(r)
            if NAME == "holding": levels |= set(d.HoldingSharesLevel.unique())
            try: frames.append(compact(d))
            except Exception as ex: print("compact err", sid, ex, flush=True)
        else: empty.append(sid)
        if i % 200 == 0: print(f"{i}/{len(ids)} {sid} rows={len(r)}", flush=True)
        time.sleep(GAP)
    df = pd.concat(frames, ignore_index=True)
    json.dump({"rows": len(df), "stocks": int(df.stock_id.nunique()), "empty": len(empty),
               "levels": sorted(levels)}, open(f"{OUTDIR}/{NAME}.json", "w"), ensure_ascii=False, indent=1)
df.to_parquet(f"{OUTDIR}/{NAME}.parquet", index=False, compression="zstd")
print(NAME, len(df), "rows", df.stock_id.nunique() if "stock_id" in df else "")
