"""Backfill adjusted daily prices (FinMind TaiwanStockPriceAdj, dividends included) for every Taiwan-listed ETF
since 2014-11, including delisted ETFs (avoid survivorship bias). Writes data/etf/<year>.parquet + universe.csv."""
import os, json, time, urllib.request, urllib.parse
import pandas as pd

TOKEN = os.environ["FINMIND_TOKEN"]
API = "https://api.finmindtrade.com/api/v4/data"
HDR = {"Authorization": f"Bearer {TOKEN}", "User-Agent": "stock-model"}
START = os.environ.get("START", "2014-11-01")
OUT = "data/etf"
GAP = 2.4
PAT = r"00\d{2,4}[A-Z]?"

def call(dataset, **q):
    url = API + "?" + urllib.parse.urlencode({"dataset": dataset, **q})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=90) as r:
                return json.loads(r.read()).get("data") or []
        except urllib.error.HTTPError as e:
            if e.code == 402: print("quota, sleep 10m", flush=True); time.sleep(600); continue
            print("HTTP", e.code, dataset, q.get("data_id"), flush=True)
            if e.code == 400: return []
        except Exception as e:
            print("err", dataset, q.get("data_id"), e, flush=True)
        time.sleep(15 * (attempt + 1))
    return []

info = pd.DataFrame(call("TaiwanStockInfo")); time.sleep(GAP)
info = info[info.stock_id.str.fullmatch(PAT)].drop_duplicates("stock_id")[["stock_id", "stock_name", "type", "industry_category"]]
dl = pd.DataFrame(call("TaiwanStockDelisting", start_date=START)); time.sleep(GAP)
dl = dl[dl.stock_id.str.fullmatch(PAT)] if len(dl) else dl
if len(dl):
    info = pd.concat([info, dl[~dl.stock_id.isin(info.stock_id)][["stock_id", "stock_name"]].assign(type="delisted", industry_category="")])
os.makedirs(OUT, exist_ok=True); info.to_csv(f"{OUT}/universe.csv", index=False)
ids = sorted(info.stock_id)
print("ETF ids:", len(ids), flush=True)

frames, empty = [], []
for i, sid in enumerate(ids):
    rows = call("TaiwanStockPriceAdj", data_id=sid, start_date=START)
    (frames.append(pd.DataFrame(rows)) if rows else empty.append(sid))
    if i % 50 == 0: print(f"{i}/{len(ids)} {sid} rows={len(rows)}", flush=True)
    time.sleep(GAP)
df = pd.concat(frames, ignore_index=True).rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume", "Trading_money": "value"})
df = df[["date", "stock_id", "open", "high", "low", "close", "volume", "value"]]
df["date"] = pd.to_datetime(df.date)
for c in ["open", "high", "low", "close"]: df[c] = df[c].astype("float32")
for y, g in df.groupby(df.date.dt.year): g.to_parquet(f"{OUT}/{y}.parquet", index=False, compression="zstd")
s = dict(start=START, ids=len(ids), with_data=int(df.stock_id.nunique()), rows=len(df), first=str(df.date.min().date()),
         last=str(df.date.max().date()), empty=empty)
json.dump(s, open(f"{OUT}/summary.json", "w"), ensure_ascii=False, indent=1); print(s)
