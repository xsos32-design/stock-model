"""Backfill 5y adjusted daily prices (FinMind TaiwanStockPriceAdj) for all listed/OTC common stocks,
including stocks delisted in the window (avoid survivorship bias). Writes data/price_adj/<year>.parquet."""
import os, re, json, time, datetime as dt, urllib.request, urllib.parse
import pandas as pd

TOKEN = os.environ["FINMIND_TOKEN"]
API = "https://api.finmindtrade.com/api/v4/data"
HDR = {"Authorization": f"Bearer {TOKEN}", "User-Agent": "stock-model"}
START = (dt.date.today() - dt.timedelta(days=365 * 5 + 2)).isoformat()
GAP = 2.4  # seconds between calls -> ~1500/hr, under the 1600/hr Backer limit

def call(dataset, **q):
    q = {"dataset": dataset, **q}
    url = API + "?" + urllib.parse.urlencode(q)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=90) as r:
                j = json.loads(r.read())
            return j.get("data") or []
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:200]
            if e.code == 402:           # hourly quota hit
                print("quota hit, sleeping 10 min", flush=True); time.sleep(600); continue
            print(f"HTTP {e.code} {dataset} {q.get('data_id')} {body}", flush=True)
            if e.code == 400: return []
        except Exception as e:
            print(f"err {dataset} {q.get('data_id')} {e}", flush=True)
        time.sleep(15 * (attempt + 1))
    return []

info = pd.DataFrame(call("TaiwanStockInfo"))
info = info[info["type"].isin(["twse", "tpex"]) & info["stock_id"].str.fullmatch(r"[1-9]\d{3}")]
info = info.drop_duplicates("stock_id")[["stock_id", "stock_name", "type", "industry_category"]]
time.sleep(GAP)
dl = pd.DataFrame(call("TaiwanStockDelisting", start_date=START))
dl = dl[dl["stock_id"].str.fullmatch(r"[1-9]\d{3}")] if len(dl) else dl
time.sleep(GAP)
ids = sorted(set(info["stock_id"]) | set(dl["stock_id"] if len(dl) else []))
print(f"universe: {len(info)} listed + {len(dl)} delisted -> {len(ids)} ids", flush=True)

os.makedirs("data/price_adj", exist_ok=True)
meta = info.copy()
if len(dl):
    d2 = dl[~dl["stock_id"].isin(info["stock_id"])][["stock_id", "stock_name"]].assign(type="delisted", industry_category="")
    meta = pd.concat([meta, d2])
meta.to_csv("data/universe.csv", index=False)

frames, empty = [], []
for i, sid in enumerate(ids + ["TAIEX", "TPEx", "0050"]):
    rows = call("TaiwanStockPriceAdj", data_id=sid, start_date=START)
    if not rows and sid in ("TAIEX", "TPEx"):
        rows = call("TaiwanStockPrice", data_id=sid, start_date=START)
    if rows: frames.append(pd.DataFrame(rows))
    else: empty.append(sid)
    if i % 100 == 0: print(f"{i}/{len(ids)+3} {sid} rows={len(rows)}", flush=True)
    time.sleep(GAP)

df = pd.concat(frames, ignore_index=True)
df = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume", "Trading_money": "value"})
df = df[["date", "stock_id", "open", "high", "low", "close", "volume", "value"]]
df["date"] = pd.to_datetime(df["date"])
for c in ["open", "high", "low", "close"]: df[c] = df[c].astype("float32")
for y, g in df.groupby(df["date"].dt.year):
    g.to_parquet(f"data/price_adj/{y}.parquet", index=False, compression="zstd")
summary = {"start": START, "ids_requested": len(ids) + 3, "ids_with_data": df["stock_id"].nunique(),
           "rows": len(df), "first": str(df["date"].min().date()), "last": str(df["date"].max().date()),
           "empty_ids": empty}
json.dump(summary, open("data/price_adj/summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "empty_ids"}), "empty:", len(empty))
