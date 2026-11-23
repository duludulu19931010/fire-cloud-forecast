"""資料層:Open-Meteo(預報 / 集合 / 空品 / 歷史預報)+ 磁碟快取(NWP 每 3 小時才重抓,逐時重算)。
資料點格式: {"time":[unix...], "elev":m, "m":{來源名:{變數:[...], "_time":[選填,該來源自己的時間軸]}}}"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

MODELS = ["ecmwf_ifs025", "icon_seamless", "gfs_seamless"]
LEVELS = [1000, 975, 950, 925, 900, 850, 800, 700, 600]
V_PATH = ["cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"]
V_SITE = V_PATH + ["relative_humidity_2m", "temperature_2m", "precipitation", "wind_speed_10m", "visibility"]
V_PLEV = (["surface_pressure", "relative_humidity_2m", "temperature_2m"]
          + [f"{v}_{p}hPa" for p in LEVELS for v in ("relative_humidity", "temperature", "geopotential_height")])
V_AIR = ["aerosol_optical_depth", "dust", "pm2_5"]

HOSTS = {"fc": "https://api.open-meteo.com/v1/forecast",
         "air": "https://air-quality-api.open-meteo.com/v1/air-quality",
         "ens": "https://ensemble-api.open-meteo.com/v1/ensemble",
         "prev": "https://previous-runs-api.open-meteo.com/v1/forecast"}


def log(*a):
    print(*a, file=sys.stderr)


def get_json(url, retries=4, timeout=60):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "fire-cloud-forecast/2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and i < retries - 1:
                time.sleep(min(60, 5 * 2 ** i))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if i < retries - 1:
                time.sleep(5 * (i + 1))
                continue
            raise


def parse_models(r, models):
    h = r["hourly"]
    m = {mod: {} for mod in models}
    for k, arr in h.items():
        if k == "time":
            continue
        if len(models) == 1:
            m[models[0]][k] = arr
            continue
        for mod in models:
            if k.endswith("_" + mod):
                m[mod][k[: -len(mod) - 1]] = arr
                break
    return {"time": h["time"], "elev": r.get("elevation"), "m": m}


def parse_ens(r, variables, model):
    """集合預報:控制成員 → ens00,第 n 個成員 → ensNN。對「變數_memberNN[_模式]」格式容錯。"""
    h, m = r["hourly"], {}
    for k, arr in h.items():
        if k == "time":
            continue
        mm = re.match(r"^(.+?)_member(\d+)(?:_" + re.escape(model) + r")?$", k)
        if mm:
            var, n = mm.group(1), int(mm.group(2))
        else:
            kk = k[: -len(model) - 1] if k.endswith("_" + model) else k
            if kk not in variables:
                continue
            var, n = kk, 0
        m.setdefault(f"ens{n:02d}", {})[var] = arr
    return {"time": h["time"], "elev": r.get("elevation"), "m": m}


def parse_prev(r, variables, models, day):
    """歷史預報:把「變數_previous_dayN[_模式]」還原成一般變數名。"""
    h, m = r["hourly"], {mod: {} for mod in models}
    suf = f"_previous_day{day}"
    for k, arr in h.items():
        if k == "time" or suf not in k:
            continue
        base, _, rest = k.partition(suf)
        mod = rest[1:] if rest.startswith("_") else (models[0] if len(models) == 1 else None)
        if mod in m:
            m[mod][base] = arr
    return {"time": h["time"], "elev": r.get("elevation"), "m": m}


def merge(dst, src):
    """把 src 的各來源併進 dst(時間軸不同時保留各自的 _time)。"""
    if not dst or not src:
        return dst
    for name, d in src["m"].items():
        if d:
            d = dict(d)
            if src["time"] != dst["time"]:
                d["_time"] = src["time"]
            dst["m"][name] = d
    return dst


class OpenMeteo:
    def __init__(self, cache_path=".cache/om.json", max_age=3 * 3600, api_key=None, clock=time.time, chunk=40):
        self.path, self.max_age, self.key, self.clock, self.chunk = cache_path, max_age, api_key, clock, chunk
        self.cache = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    self.cache = json.load(f)
            except (OSError, ValueError):
                self.cache = {}

    @staticmethod
    def _k(group, p):
        return f"{group}|{p[0]:.4f},{p[1]:.4f}"

    def _url(self, api, params):
        base = HOSTS[api]
        if self.key:
            base = base.replace("://", "://customer-", 1)
            params = dict(params, apikey=self.key)
        return base + "?" + urllib.parse.urlencode(params, safe=",")

    def _cached(self, group, points, fetch_part, ttl=None):
        now, ttl = self.clock(), ttl or self.max_age
        pts = sorted(set(points))
        todo = [p for p in pts if self._k(group, p) not in self.cache
                or now - self.cache[self._k(group, p)]["t"] > ttl]
        for i in range(0, len(todo), self.chunk):
            part = todo[i:i + self.chunk]
            try:
                for p, d in zip(part, fetch_part(part)):
                    self.cache[self._k(group, p)] = {"t": now, "d": d}
            except Exception as e:                 # 失敗時退回舊快取,不讓整條管線掛掉
                log(f"[fetch] {group} chunk {i} failed: {e}")
        return {p: self.cache[self._k(group, p)]["d"] for p in pts if self._k(group, p) in self.cache}

    @staticmethod
    def _q(pts, variables, **kw):
        q = {"latitude": ",".join(f"{p[0]:.4f}" for p in pts), "longitude": ",".join(f"{p[1]:.4f}" for p in pts),
             "hourly": ",".join(variables), "timeformat": "unixtime"}
        q.update(kw)
        return q

    @staticmethod
    def _list(res):
        return [res] if isinstance(res, dict) else res

    def get(self, points, variables, group, days=3, past_days=2, models=MODELS):
        def part(pts):
            q = self._q(pts, variables, models=",".join(models), forecast_days=days, past_days=past_days,
                        wind_speed_unit="ms")
            return [parse_models(r, models) for r in self._list(get_json(self._url("fc", q)))]
        return self._cached(group, points, part)

    def get_air(self, points, days=3, past_days=2):
        def part(pts):
            q = self._q(pts, V_AIR, forecast_days=days, past_days=past_days)
            out = []
            for r in self._list(get_json(self._url("air", q))):
                d = parse_models(r, ["cams"])
                out.append(d)
            return out
        return self._cached("air", points, part, ttl=6 * 3600)

    def get_ens(self, points, variables=tuple(V_PATH), model="ecmwf_ifs025", days=2):
        def part(pts):
            q = self._q(pts, variables, models=model, forecast_days=days)
            return [parse_ens(r, set(variables), model) for r in self._list(get_json(self._url("ens", q)))]
        return self._cached("ens", points, part, ttl=6 * 3600)

    def get_prev(self, points, variables, day=1, past_days=30, models=MODELS):
        """Previous Runs API:N 天前發布的預報(回補冷啟動歷史用)。"""
        vs = [f"{v}_previous_day{day}" for v in variables]

        def part(pts):
            q = self._q(pts, vs, models=",".join(models), past_days=past_days, forecast_days=1)
            return [parse_prev(r, variables, models, day) for r in self._list(get_json(self._url("prev", q)))]
        return self._cached(f"prev{day}", points, part, ttl=12 * 3600)

    def flush(self, keep_age=24 * 3600):
        now = self.clock()
        self.cache = {k: v for k, v in self.cache.items() if now - v["t"] < keep_age}
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, separators=(",", ":"))
