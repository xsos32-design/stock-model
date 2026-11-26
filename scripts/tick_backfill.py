"""Tick (逐筆) backfill for the '499-lot order' day-trade study, limited to hot stocks to save FinMind quota.
Universe: top-10 by 20-day traded value in every industry + top-50 overall (common stocks only).
Per stock-day we keep 1-minute bars plus large-trade counts (ticks >= 400 lots, and exactly 499 lots), split by
buy/sell side, instead of raw ticks (raw ticks would be far too big for the repo).
Resumable: a stock file is skipped once all its days are present. Env: DAYS (default 120), CHUNK (stocks per run)."""
import os, json, time, glob, urllib.request, urllib.parse
import numpy as np, pandas as pd

TOKEN = os.environ["FINMIND_TOKEN"]
API = "https://api.finmindtrade.com/api/v4/data"
HDR = {"Authorization": f"Bearer {TOKEN}", "User-Agent": "stock-model"}
DAYS = int(os.environ.get("DAYS", "120")); CHUNK = int(os.environ.get("CHUNK", "12"))
OUT = "data/tick"; os.makedirs(OUT, exist_ok=True)
GAP = 2.3

def call(dataset, **q):
    url = API + "?" + urllib.parse.urlencode({"dataset": dataset, **q})
    for a in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HDR), timeout=120) as r:
                j = json.loads(r.read())
            return j.get("data") or [], j.get("msg", "")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:200]
            if e.code == 402: print("quota, sleep 10m", flush=True); time.sleep(600); continue
            if e.code == 400: return [], body
            print("HTTP", e.code, body, flush=True)
        except Exception as e:
            print("err", e, flush=True)
        time.sleep(15 * (a + 1))
    return [], "failed"

# ---- universe from data already in the repo ----
uni = pd.read_csv("data/universe.csv", dtype=str)
px = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob("data/price_adj/*.parquet"))[-1:]] +
               [pd.read_parquet(p) for p in sorted(glob.glob("data/live/daily/*.parquet"))[-30:]])
px = px[px.stock_id.str.fullmatch(r"[1-9]\d{3}")]
px["date"] = pd.to_datetime(px.date)
v = px[px.date >= px.date.max() - pd.Timedelta(days=40)].groupby("stock_id").value.mean().rename("v").reset_index()
v = v.merge(uni[["stock_id", "industry_category"]].drop_duplicates("stock_id"), on="stock_id", how="left")
top_ind = v.sort_values("v", ascending=False).groupby("industry_category").head(10)
pool = set(top_ind.stock_id) | set(v.nlargest(50, "v").stock_id)
ids = list(v[v.stock_id.isin(pool)].sort_values("v", ascending=False).stock_id.head(int(os.environ.get("MAXN", "180"))))   # most traded first
json.dump(ids, open(f"{OUT}/universe.json", "w"))
print("universe", len(ids), flush=True)

# ---- trading days ----
td, _ = call("TaiwanStockTradingDate", start_date=(pd.Timestamp.today() - pd.Timedelta(days=DAYS * 2)).strftime("%Y-%m-%d"))
today_tpe = (pd.Timestamp.utcnow() + pd.Timedelta(hours=8)).strftime("%Y-%m-%d")
days = [d["date"] for d in td if d["date"] < today_tpe][-DAYS:]   # never today: ticks appear only after the close
time.sleep(GAP)

# ---- probe ----
rows, msg = call("TaiwanStockPriceTick", data_id="2330", start_date=days[-2])
print("probe 2330", days[-2], "rows", len(rows), "msg", msg, "cols", list(rows[0].keys()) if rows else None, flush=True)
if not rows:
    json.dump(dict(ok=False, msg=msg), open(f"{OUT}/probe.json", "w"), ensure_ascii=False); raise SystemExit("tick dataset not available: " + msg)
json.dump(dict(ok=True, cols=list(rows[0].keys()), sample=rows[:3]), open(f"{OUT}/probe.json", "w"), ensure_ascii=False, default=str)
time.sleep(GAP)

def bars(rows, day):
    t = pd.DataFrame(rows)
    if t.empty: return None
    t["ts"] = pd.to_datetime(day + " " + t.Time.astype(str).str[:8], errors="coerce")
    t = t.dropna(subset=["ts"])
    t["p"] = t.deal_price.astype(float); vol = t.volume.astype(float)
    t["lots"] = vol / 1000 if vol.median() > 50 else vol            # FinMind may report shares or lots
    side = t.TickType.astype(str) if "TickType" in t else pd.Series("0", index=t.index)
    t["buy"] = side.isin(["2", "B", "buy"]); t["sell"] = side.isin(["1", "S", "sell"])
    t["big"] = t.lots >= 400; t["b499"] = (t.lots >= 495) & (t.lots <= 499)
    t["m"] = t.ts.dt.floor("1min")
    g = t.groupby("m")
    b = pd.DataFrame({"o": g.p.first(), "h": g.p.max(), "l": g.p.min(), "c": g.p.last(), "v": g.lots.sum(),
                      "big_buy": t[t.big & t.buy].groupby("m").lots.sum(), "big_sell": t[t.big & t.sell].groupby("m").lots.sum(),
                      "n499_buy": t[t.b499 & t.buy].groupby("m").size(), "n499_sell": t[t.b499 & t.sell].groupby("m").size()}).fillna(0)
    return b.reset_index().rename(columns={"m": "ts"})

done_runs = 0
for sid in ids:
    fp = f"{OUT}/{sid}.parquet"
    have = set(pd.read_parquet(fp).ts.dt.strftime("%Y-%m-%d")) if os.path.exists(fp) else set()
    need = [d for d in days if d not in have]
    if not need: continue
    if done_runs >= CHUNK and len(need) > 3: break
    frames = [pd.read_parquet(fp)] if have else []
    for d in need:
        rows, msg = call("TaiwanStockPriceTick", data_id=sid, start_date=d)
        b = bars(rows, d)
        if b is not None: frames.append(b)
        time.sleep(GAP)
    if frames:
        df = pd.concat(frames).drop_duplicates("ts").sort_values("ts")
        for c in ["o", "h", "l", "c"]: df[c] = df[c].astype("float32")
        df.to_parquet(fp, index=False, compression="zstd")
    if len(need) > 3: done_runs += 1   # small top-ups (1-3 new days) do not use up the per-run budget
    print("done", sid, len(need), "days", flush=True)
left = sum(1 for s in ids if not os.path.exists(f"{OUT}/{s}.parquet"))
json.dump(dict(ids=len(ids), files=len(glob.glob(f"{OUT}/[0-9]*.parquet")), left=left, days=len(days), first=days[0], last=days[-1]),
          open(f"{OUT}/summary.json", "w"))
print("left", left, flush=True)
