"""Fetch market-level datasets (2014-11 ~ today) for regime / risk research. Unknown dataset names are logged, not fatal."""
import os, json, time, datetime as dt, urllib.request, urllib.parse
import pandas as pd
TOKEN = os.environ["FINMIND_TOKEN"]; API = "https://api.finmindtrade.com/api/v4/data"
HDR = {"Authorization": f"Bearer {TOKEN}", "User-Agent": "stock-model"}
START = dt.date(2014, 11, 1); TODAY = dt.date.today()
SETS = [("TaiwanFuturesInstitutionalInvestors", "TX"), ("TaiwanOptionInstitutionalInvestors", "TXO"),
        ("TaiwanFuturesDaily", "TX"), ("TaiwanFuturesOpenInterestLargeTraders", "TX"),
        ("TaiwanStockTotalMarginPurchaseShortSale", None), ("TaiwanStockTotalInstitutionalInvestors", None),
        ("TaiwanTotalExceptMarginPurchaseShortSale", None), ("TaiwanStockMarginMaintenance", None),
        ("TaiwanExchangeRate", "USD"), ("CnnFearGreedIndex", None), ("TaiwanBusinessIndicator", None),
        ("GovernmentBondsYield", "United States 10-Year"), ("CrudeOilPrices", "WTI"),
        ("TaiwanOptionDaily", "TXO"), ("TaiwanStockPrice", "TAIEX"), ("TaiwanStockPrice", "TPEx")]
def call(ds, did, s, e):
    q = {"dataset": ds, "start_date": s.isoformat(), "end_date": e.isoformat()}
    if did: q["data_id"] = did
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(API + "?" + urllib.parse.urlencode(q), headers=HDR), timeout=120) as r:
                j = json.loads(r.read()); return j.get("data") or [], j.get("msg")
        except urllib.error.HTTPError as ex:
            msg = ex.read()[:150].decode("utf-8", "replace")
            if ex.code == 402: time.sleep(600); continue
            return [], f"HTTP {ex.code} {msg}"
        except Exception as ex:
            time.sleep(10)
    return [], "error"
os.makedirs("data/market", exist_ok=True); log = {}
for ds, did in SETS:
    rows, msgs, s = [], set(), START
    step = 120 if ds in ("TaiwanFuturesDaily", "TaiwanOptionDaily", "TaiwanOptionInstitutionalInvestors") else 365
    while s <= TODAY:
        e = min(s + dt.timedelta(days=step), TODAY); r, m = call(ds, did, s, e); rows += r; msgs.add(str(m)); s = e + dt.timedelta(days=1); time.sleep(2.4)
    name = ds + (f"_{did}" if did else "")
    if rows:
        df = pd.DataFrame(rows)
        if ds == "TaiwanOptionDaily":   # keep only what is needed for put/call ratios
            keep = [c for c in ["date", "call_put", "volume", "open_interest"] if c in df]
            df = df[keep].groupby([c for c in ["date", "call_put"] if c in keep]).sum(numeric_only=True).reset_index()
        if ds == "TaiwanFuturesDaily" and "contract_date" in df:
            df = df.sort_values(["date", "contract_date"]).groupby("date").head(1)
        df.to_parquet(f"data/market/{name.replace(' ', '_')}.parquet", index=False)
    log[name] = {"rows": len(rows), "msg": sorted(msgs)[:3], "first": rows[0].get("date") if rows else None, "last": rows[-1].get("date") if rows else None}
    print(name, log[name], flush=True)
json.dump(log, open("data/market/summary.json", "w"), ensure_ascii=False, indent=1)
