"""中央氣象署開放資料:即時測站觀測(雲海 nowcast)。需要環境變數 CWA_API_KEY(免費申請)。
欄位解析採「遞迴找 key」的容錯寫法,因為 CWA 資料集曾多次改版。"""
import json
import math
import urllib.request

URL = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/{}?Authorization={}&format=JSON"
DATASETS = ("O-A0003-001", "O-A0001-001")      # 10 分鐘綜觀氣象 / 自動氣象站
BAD = {-99.0, -999.0, -998.0, -99.5}


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x in BAD else x


def _walk(node, found):
    if isinstance(node, dict):
        if "StationLatitude" in node and "StationLongitude" in node:
            if node.get("CoordinateName") in (None, "WGS84") or "coord" not in found:
                found["coord"] = (_num(node["StationLatitude"]), _num(node["StationLongitude"]))
        for k, v in node.items():
            if k in ("StationId", "StationName", "StationAltitude", "AirTemperature", "RelativeHumidity",
                     "WindSpeed", "stationId", "locationName", "lat", "lon", "ELEV", "TEMP", "HUMD", "WDSD") \
                    and not isinstance(v, (dict, list)) and k not in found:
                found[k] = v
            _walk(v, found)
    elif isinstance(node, list):
        for v in node:
            _walk(v, found)


def _flatten(st):
    f = {}
    _walk(st, f)
    if "weatherElement" in st and isinstance(st["weatherElement"], list):      # 舊格式
        for el in st["weatherElement"]:
            if isinstance(el, dict) and el.get("elementName") in ("ELEV", "TEMP", "HUMD", "WDSD"):
                f.setdefault(el["elementName"], el.get("elementValue"))
    sid = f.get("StationId") or f.get("stationId")
    if not sid:
        return None
    lat, lon = f.get("coord", (None, None))
    lat = lat if lat is not None else _num(f.get("lat"))
    lon = lon if lon is not None else _num(f.get("lon"))
    rh = _num(f.get("RelativeHumidity", f.get("HUMD")))
    if rh is not None and "HUMD" in f and "RelativeHumidity" not in f and rh <= 1.0:
        rh *= 100
    return {"id": str(sid), "name": f.get("StationName") or f.get("locationName"),
            "lat": lat, "lon": lon, "alt": _num(f.get("StationAltitude", f.get("ELEV"))),
            "T": _num(f.get("AirTemperature", f.get("TEMP"))), "RH": rh,
            "ws": _num(f.get("WindSpeed", f.get("WDSD")))}


def fetch_obs(key, datasets=DATASETS, opener=urllib.request.urlopen):
    out = {}
    for ds in datasets:
        try:
            with opener(URL.format(ds, key), timeout=30) as r:
                recs = json.load(r).get("records", {})
        except Exception:
            continue
        for st in recs.get("Station") or recs.get("location") or []:
            s = _flatten(st)
            if s and s["lat"] is not None and s["lon"] is not None and s["id"] not in out:
                out[s["id"]] = s
    return out


def _km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742.0 * math.asin(math.sqrt(h))


def sea_sample(site, stations):
    """山上測站 vs 谷地測站 → {score 0~100, ws, rh_v, rh_p};配不到測站對回傳 None。
    谷地近飽和 + 山上乾 + 穩定層 → 雲海現正成形。"""
    E, here = site["elev"], (site["lat"], site["lon"])
    near = [s for s in stations.values() if s["alt"] is not None and s["RH"] is not None
            and s["T"] is not None and _km(here, (s["lat"], s["lon"])) <= 30]
    peaks = [s for s in near if abs(s["alt"] - E) <= 400 and _km(here, (s["lat"], s["lon"])) <= 12]
    valleys = [s for s in near if E - 1800 <= s["alt"] <= E - 500]
    if not peaks or not valleys:
        return None
    p = min(peaks, key=lambda s: _km(here, (s["lat"], s["lon"])))
    v = min(valleys, key=lambda s: _km(here, (s["lat"], s["lon"])))
    sat = max(0.0, min(1.0, (v["RH"] - 80) / 15))
    clear = 1.0 if p["RH"] < 88 else max(0.0, 1 - (p["RH"] - 88) / 10)
    lapse = (v["T"] - p["T"]) / max((p["alt"] - v["alt"]) / 1000.0, 0.1)
    stab = max(0.0, min(1.0, (8 - lapse) / 6))
    return {"score": round(100 * sat * clear * (0.5 + 0.5 * stab), 1), "ws": p.get("ws"),
            "rh_v": v["RH"], "rh_p": p["RH"], "st": [v["id"], p["id"]]}


def nowcast_sea(site, stations):
    s = sea_sample(site, stations)
    return None if s is None else s["score"]
