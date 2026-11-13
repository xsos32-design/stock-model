"""Probe which FinMind / TWSE / TPEx datasets this account can actually read.
Writes metadata only (status, rows, date range, columns) to results/probe.md. Never prints the token."""
import os, json, time, datetime as dt, urllib.request, urllib.parse

TOKEN = os.environ.get("FINMIND_TOKEN", "")
API = "https://api.finmindtrade.com/api/v4/data"
HDR = {"Authorization": f"Bearer {TOKEN}", "User-Agent": "stock-model-probe"}
today = dt.date.today()
recent = (today - dt.timedelta(days=20)).isoformat()
five_y = (today - dt.timedelta(days=365 * 5 + 2)).isoformat()

def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "stock-model-probe"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return -1, str(e)

def fm(dataset, data_id=None, start=recent, end=None):
    q = {"dataset": dataset, "start_date": start}
    if data_id: q["data_id"] = data_id
    if end: q["end_date"] = end
    code, body = get(API + "?" + urllib.parse.urlencode(q), HDR)
    try:
        j = json.loads(body)
    except Exception:
        return {"http": code, "msg": body[:120], "rows": 0}
    d = j.get("data") or []
    dates = sorted({x.get("date") for x in d if x.get("date")})
    return {"http": code, "msg": str(j.get("msg", ""))[:120], "rows": len(d),
            "first": dates[0] if dates else "", "last": dates[-1] if dates else "",
            "cols": ",".join(list(d[0].keys())[:12]) if d else ""}

# (dataset, data_id, start) ; start "5y" = test full 5-year backfill
TESTS = [
    ("TaiwanStockInfo", None, recent),
    ("TaiwanStockTradingDate", None, recent),
    ("TaiwanStockPrice", "2330", "5y"),
    ("TaiwanStockPrice", "TAIEX", "5y"),
    ("TaiwanStockPrice", None, today.isoformat()),          # whole market one day
    ("TaiwanStockPriceAdj", "2330", "5y"),
    ("TaiwanStockPER", "2330", "5y"),
    ("TaiwanStockDayTrading", "2330", recent),
    ("TaiwanStockTotalReturnIndex", "TAIEX", recent),
    ("TaiwanStockMarginPurchaseShortSale", "2330", "5y"),
    ("TaiwanStockTotalMarginPurchaseShortSale", None, recent),
    ("TaiwanStockInstitutionalInvestorsBuySell", "2330", "5y"),
    ("TaiwanStockInstitutionalInvestorsBuySell", None, today.isoformat()),
    ("TaiwanStockTotalInstitutionalInvestors", None, recent),
    ("TaiwanStockShareholding", "2330", "5y"),
    ("TaiwanStockHoldingSharesPer", "2330", recent),
    ("TaiwanStockSecuritiesLending", "2330", recent),
    ("TaiwanStockMonthRevenue", "2330", "5y"),
    ("TaiwanStockFinancialStatements", "2330", "5y"),
    ("TaiwanStockBalanceSheet", "2330", "5y"),
    ("TaiwanStockCashFlowsStatement", "2330", "5y"),
    ("TaiwanStockDividend", "2330", "5y"),
    ("TaiwanStockDividendResult", "2330", "5y"),
    ("TaiwanStockCapitalReductionReferencePrice", None, five_y),
    ("TaiwanStockSplitPrice", None, five_y),
    ("TaiwanStockDelisting", None, five_y),
    ("TaiwanStockDispositionSecuritiesPeriod", None, recent),
    ("TaiwanStockSuspended", None, recent),
    ("TaiwanStockMarketValue", "2330", recent),
    ("TaiwanStockNews", "2330", today.isoformat()),
]

lines = [f"# 資料探測結果 {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC", ""]
code, body = get("https://api.web.finmindtrade.com/v2/user_info", HDR)
try:
    u = json.loads(body)
    lines.append(f"- 帳號等級 level：{u.get('level')}　本小時已用：{u.get('user_count')} / {u.get('api_request_limit')}")
except Exception:
    lines.append(f"- user_info 讀取失敗 http={code}")
lines.append(f"- token 是否有載入：{'是' if TOKEN else '否'}")
lines += ["", "## FinMind", "", "| dataset | data_id | 起始 | http | rows | 最早 | 最晚 | msg | 欄位 |", "|---|---|---|---|---|---|---|---|---|"]
for ds, did, st in TESTS:
    start = five_y if st == "5y" else st
    r = fm(ds, did, start)
    lines.append(f"| {ds} | {did or '全市場'} | {start} | {r['http']} | {r['rows']} | {r.get('first','')} | {r.get('last','')} | {r['msg']} | {r.get('cols','')} |")
    time.sleep(0.6)

TW = ["exchangeReport/STOCK_DAY_ALL", "exchangeReport/BWIBBU_ALL", "exchangeReport/MI_MARGN",
      "opendata/t187ap05_P", "exchangeReport/TWT85U", "announcement/punish", "announcement/notice",
      "fund/T86", "exchangeReport/MI_INDEX"]
TP = ["tpex_mainboard_daily_close_quotes", "tpex_mainboard_peratio_analysis", "tpex_mainboard_margin_balance",
      "tpex_disposal_information", "tpex_trading_warning_information", "tpex_3insti_daily_trading"]
lines += ["", "## 證交所 / 櫃買 OpenAPI", "", "| endpoint | http | rows | 欄位 |", "|---|---|---|---|"]
for base, eps in (("https://openapi.twse.com.tw/v1/", TW), ("https://www.tpex.org.tw/openapi/v1/", TP)):
    for ep in eps:
        code, body = get(base + ep)
        try:
            j = json.loads(body); rows = len(j); cols = ",".join(list(j[0].keys())[:10]) if j else ""
        except Exception:
            rows, cols = 0, body[:60].replace("|", "/").replace("\n", " ")
        lines.append(f"| {base.split('/')[2]}/{ep} | {code} | {rows} | {cols} |")

os.makedirs("results", exist_ok=True)
open("results/probe.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
