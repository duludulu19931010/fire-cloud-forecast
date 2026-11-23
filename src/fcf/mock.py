"""合成資料(測試 / --mock 乾跑用):與 OpenMeteo.get 介面相同,不連網。"""
import math

from .fetch import LEVELS, MODELS

SCENARIO = {"low": 8.0, "mid": 20.0, "high": 55.0, "wall": None, "moist_top": 1500.0}


class MockOpenMeteo:
    def __init__(self, now, scenario=None, days=3):
        self.sc = dict(SCENARIO, **(scenario or {}))
        self.t0 = int(now // 86400 * 86400) - 2 * 86400          # 含 past_days=2
        self.n = (days + 2) * 24 + 1

    def flush(self):
        pass

    def _series(self, f):
        return [f(self.t0 + i * 3600) for i in range(self.n)]

    def get(self, points, variables, group, days=3, past_days=2, models=MODELS):
        out = {}
        for (lat, lon) in set(points):
            m = {}
            for k, mod in enumerate(models):
                d = {}
                for v in variables:
                    d[v] = self._series(lambda t, v=v, k=k: self._val(v, lat, lon, t, k))
                m[mod] = d
            out[(lat, lon)] = {"time": [self.t0 + i * 3600 for i in range(self.n)], "elev": 5.0, "m": m}
        return out

    def _val(self, v, lat, lon, t, k):
        sc = self.sc
        if v.startswith("cloud_cover_"):
            lay = v.split("_")[-1]
            val = sc[lay] + 3 * k
            if lay != "low" and sc.get("west_clear_lon") is not None and lon < sc["west_clear_lon"]:
                val = 0.0
            w = sc["wall"]                      # (層, 經度下限, 經度上限):人造雲牆
            if w and lay == w[0] and w[1] <= lon <= w[2]:
                val = 100.0
            return max(0.0, min(100.0, val))
        if v == "precipitation":
            return 0.0
        if v == "wind_speed_10m":
            return 2.5
        if v == "visibility":
            return 24000.0 if k else None
        if v == "aerosol_optical_depth":
            return 0.25
        if v in ("dust", "pm2_5"):
            return 10.0
        if v == "surface_pressure":
            return 1010.0
        if v == "relative_humidity_2m":
            return 96.0 if sc["moist_top"] > 0 else 50.0
        if v == "temperature_2m":
            return 16.0
        for p in LEVELS:
            z = 44330.8 * (1 - (p / 1013.25) ** 0.190263)
            if v == f"geopotential_height_{p}hPa":
                return z
            if v == f"temperature_{p}hPa":
                base = 16.0 - 6.5 * z / 1000
                return base + (4.0 if z > sc["moist_top"] else 0.0)
            if v == f"relative_humidity_{p}hPa":
                return 97.0 if z <= sc["moist_top"] else 35.0
        return 0.0


    def get_air(self, points, days=3, past_days=2):
        return self.get(points, ["aerosol_optical_depth", "dust", "pm2_5"], "air", models=["cams"])

    def get_ens(self, points, variables=("cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"),
                model="ecmwf_ifs025", days=2, members=12):
        out = {}
        for (lat, lon) in set(points):
            m = {}
            for n in range(members):
                m[f"ens{n:02d}"] = {v: self._series(lambda t, v=v, n=n: min(100.0, max(0.0, self._val(v, lat, lon, t, 0) + (n - members / 2) * 9)))
                                    for v in variables}
            out[(lat, lon)] = {"time": [self.t0 + i * 3600 for i in range(self.n)], "elev": 5.0, "m": m}
        return out

    def get_prev(self, points, variables, day=1, past_days=30, models=MODELS):
        return self.get(points, list(variables), f"prev{day}", models=models)
