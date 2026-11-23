"""Meteoblue Forecast API(選填;需 METEOBLUE_API_KEY,Free Weather API 試用 1 年)。
免費方案的 clouds 套件只有 day/6h/3h,所以雲層是 3 小時解析度,這裡線性內插成逐時,權重在共識裡應低於 ECMWF。
特色:支援 asl 參數(依實際海拔修正)— 山區站點很有用。meteoblue 的雲層定義與 Open-Meteo 略不同(低 ≤2 km、中 ≤7 km、高 ≤13 km AGL)。
⚠ 回應欄位名稱我沒有用真實金鑰驗證過;解析採「找得到就用」的容錯寫法,找不到會把實際欄位印到 log。"""
import calendar
import re
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .fetch import get_json, log

URL = "https://my.meteoblue.com/packages/basic-1h_clouds-3h"
NAMES = {"cloud_cover_low": ("lowclouds", "low_clouds"), "cloud_cover_mid": ("midclouds", "mid_clouds"),
         "cloud_cover_high": ("highclouds", "high_clouds"), "visibility": ("visibility",),
         "precipitation": ("precipitation",), "relative_humidity_2m": ("relativehumidity", "relative_humidity"),
         "temperature_2m": ("temperature",), "wind_speed_10m": ("windspeed",)}


def _ts(v):
    if isinstance(v, (int, float)):
        return int(v)
    m = re.match(r"(\d{4})-(\d\d)-(\d\d)[ T](\d\d):(\d\d)", str(v))
    return calendar.timegm(tuple(int(x) for x in m.groups()) + (0, 0, 0)) if m else None


def parse(res):
    """→ 逐時資料點(來源名 meteoblue);回傳 None 表示認不得格式。"""
    blocks = [(res.get(k) or {}) for k in ("data_1h", "data_3h")]
    series = {}
    for blk in blocks:
        ts = [_ts(t) for t in blk.get("time", [])]
        if not ts or None in ts:
            continue
        for var, names in NAMES.items():
            for n in names:
                if n in blk and var not in series:
                    series[var] = (ts, blk[n])
    if not any(v.startswith("cloud_cover") for v in series):
        log("[meteoblue] 認不得欄位,實際 keys:", {k: list((res.get(k) or {}).keys())[:12] for k in ("data_1h", "data_3h")})
        return None
    lo = min(s[0][0] for s in series.values())
    hi = max(s[0][-1] for s in series.values())
    t0 = lo // 3600 * 3600
    axis = list(range(t0, hi + 1, 3600))
    out = {}
    for var, (ts, vals) in series.items():
        arr, j = [], 0
        for t in axis:
            while j + 1 < len(ts) and ts[j + 1] <= t:
                j += 1
            if t < ts[0] or t > ts[-1] or vals[j] is None:
                arr.append(None)
            elif j + 1 < len(ts) and vals[j + 1] is not None and ts[j + 1] > ts[j]:
                arr.append(vals[j] + (vals[j + 1] - vals[j]) * (t - ts[j]) / (ts[j + 1] - ts[j]))
            else:
                arr.append(vals[j])
        if var == "wind_speed_10m":
            arr = [None if v is None else v / 3.6 for v in arr]      # 預設 km/h → m/s
        out[var] = arr
    return {"time": axis, "elev": None, "m": {"meteoblue": out}}


def fetch_points(points, key, workers=6):
    """points: [(lat, lon, asl|None)] → {(lat, lon): 資料點}。逐點請求(API 一次一個地點),失敗的略過。"""
    def one(p):
        q = {"lat": f"{p[0]:.4f}", "lon": f"{p[1]:.4f}", "apikey": key, "format": "json", "tz": "UTC",
             "history_days": 2}
        if p[2] is not None:
            q["asl"] = int(p[2])
        try:
            return (p[0], p[1]), parse(get_json(URL + "?" + urllib.parse.urlencode(q), retries=2))
        except Exception as e:
            log(f"[meteoblue] {p[:2]} failed: {e}")
            return (p[0], p[1]), None
    with ThreadPoolExecutor(workers) as ex:
        return {k: v for k, v in ex.map(one, points) if v}


class Cached:
    """3 小時快取(與 Open-Meteo 同步),避免每小時耗用額度。"""
    def __init__(self, store, key, ttl=3 * 3600, clock=time.time):
        self.store, self.key, self.ttl, self.clock = store, key, ttl, clock

    def get(self, points):
        now, out, todo = self.clock(), {}, []
        for p in points:
            k = f"mb|{p[0]:.4f},{p[1]:.4f}|{p[2]}"
            e = self.store.get(k)
            if e and now - e["t"] <= self.ttl:
                out[(p[0], p[1])] = e["d"]
            else:
                todo.append(p)
        for k, d in fetch_points(todo, self.key).items():
            asl = next((p[2] for p in todo if (p[0], p[1]) == k), None)
            self.store[f"mb|{k[0]:.4f},{k[1]:.4f}|{asl}"] = {"t": now, "d": d}
            out[k] = d
        return out
