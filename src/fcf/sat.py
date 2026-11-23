"""Himawari-9 衛星:① nowcast 雲層(紅外亮溫→雲頂高)② 「火燒雲真的發生了嗎」偵測器(可見光紅/藍比)。
資料:NOAA ISatSS AHI-L1b-FLDK(AWS 公開 S3,10 分鐘,分塊 NetCDF,GOES 式固定網格,衛星 140.7°E)。
選用依賴:numpy + h5netcdf(或 netCDF4)。缺依賴/抓不到 → 回傳 None,整個功能自動停用,不影響主管線。
⚠ 我在沙箱內無法連網:固定網格數學、偵測器、雲層估計都用合成資料測過;
   S3 目錄結構與 tile 檔案格式是依 NOAA/earth2studio 文件的「假設」(標 ASSUMPTION),請先跑 scripts/sat_probe.py 驗證。"""
import io
import json
import math
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from . import geo
from .fetch import log

# ── GOES 式固定網格投影(Himawari-9 ISatSS 用同一套;衛星高度/經度由檔案屬性覆蓋)──
REQ, RPOL, HSAT, LON0 = 6378137.0, 6356752.31414, 35786023.0, 140.7


def to_scan(lat, lon, h=HSAT, lon0=LON0):
    """地理座標 → 掃描角 (x, y) 弧度;在衛星背面回傳 None。"""
    H = h + REQ
    phi = math.atan((RPOL ** 2 / REQ ** 2) * math.tan(math.radians(lat)))
    e2 = 1 - RPOL ** 2 / REQ ** 2
    rc = RPOL / math.sqrt(1 - e2 * math.cos(phi) ** 2)
    sx = H - rc * math.cos(phi) * math.cos(math.radians(lon - lon0))
    sy = -rc * math.cos(phi) * math.sin(math.radians(lon - lon0))
    sz = rc * math.sin(phi)
    if H * (H - sx) < sy ** 2 + (REQ ** 2 / RPOL ** 2) * sz ** 2:
        return None
    return math.asin(-sy / math.sqrt(sx ** 2 + sy ** 2 + sz ** 2)), math.atan(sz / sx)


def to_geo(x, y, h=HSAT, lon0=LON0):
    H = h + REQ
    a = math.sin(x) ** 2 + math.cos(x) ** 2 * (math.cos(y) ** 2 + (REQ ** 2 / RPOL ** 2) * math.sin(y) ** 2)
    b = -2 * H * math.cos(x) * math.cos(y)
    c = H ** 2 - REQ ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    rs = (-b - math.sqrt(disc)) / (2 * a)
    sx, sy, sz = rs * math.cos(x) * math.cos(y), -rs * math.sin(x), rs * math.cos(x) * math.sin(y)
    return (math.degrees(math.atan((REQ ** 2 / RPOL ** 2) * sz / math.sqrt((H - sx) ** 2 + sy ** 2))),
            lon0 - math.degrees(math.atan(sy / (H - sx))))


def planck_bt(rad, fk1, fk2, bc1, bc2):
    return (fk2 / math.log(fk1 / rad + 1) - bc1) / bc2 if rad and rad > 0 else None


# ───────────────────────── 場景(規則經緯網格) ─────────────────────────
class Scene:
    """lat 由北往南、lon 由西往東的規則網格;bt/r01/r03 為 [ny][nx] 巢狀 list(None=無資料)。"""
    def __init__(self, ts, lat_max, lon_min, step, bt, r01=None, r03=None):
        self.ts, self.lat_max, self.lon_min, self.step = ts, lat_max, lon_min, step
        self.bt, self.r01, self.r03 = bt, r01, r03
        self.ny, self.nx = len(bt), len(bt[0])

    def latlon(self, i, j):
        return self.lat_max - i * self.step, self.lon_min + j * self.step

    def cell(self, lat, lon):
        i, j = round((self.lat_max - lat) / self.step), round((lon - self.lon_min) / self.step)
        return (i, j) if 0 <= i < self.ny and 0 <= j < self.nx else None


def _bd(lat0, lon0, lat, lon):
    dy, dx = (lat - lat0) * 111.19, (lon - lon0) * 111.19 * math.cos(math.radians(lat0))
    return math.hypot(dx, dy), (math.degrees(math.atan2(dx, dy)) + 360) % 360


def _cells_within(sc, lat0, lon0, rmin, rmax, az=None, half=180.0):
    r = int(rmax / 111.19 / sc.step) + 1
    c = sc.cell(lat0, lon0)
    if not c:
        return
    for i in range(max(0, c[0] - r), min(sc.ny, c[0] + r + 1)):
        for j in range(max(0, c[1] - r), min(sc.nx, c[1] + r + 1)):
            la, lo = sc.latlon(i, j)
            d, b = _bd(lat0, lon0, la, lo)
            if rmin <= d <= rmax and (az is None or abs((b - az + 180) % 360 - 180) <= half):
                yield i, j


def height_km(bt, sfc_k):
    return (sfc_k - bt) / 6.5                      # 粗略:雲頂高 ≈ 溫差 / 6.5 K/km(卷雲偏低估,逆溫下偏差大)


def layer_fracs(sc, lat, lon, sfc_k, radius=30.0):
    """半徑內低/中/高雲覆蓋比例(紅外亮溫推雲頂高);無資料回傳 None。"""
    n = lo = mid = hi = 0
    for i, j in _cells_within(sc, lat, lon, 0, radius):
        t = sc.bt[i][j]
        if t is None:
            continue
        n += 1
        z = height_km(t, sfc_k)
        if z >= 6:
            hi += 1
        elif z >= 2:
            mid += 1
        elif z >= 0.5:
            lo += 1
    return None if n < 5 else {"low": lo / n, "mid": mid / n, "high": hi / n}


def nowcast_covers(sc, site, az, sfc_k, us=None):
    from .scoring import US
    out = []
    for u in (us or US):
        la, lo = geo.dest(site["lat"], site["lon"], az, u)
        out.append(layer_fracs(sc, la, lo, sfc_k, radius=25.0 if u else 20.0))
    return out if sum(o is not None for o in out) >= 5 else None


# ───────────────────────── 火燒雲偵測器 ─────────────────────────
THRS = (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0)
DETECTOR_PRIOR = {"ratio_thr": 1.4, "tau": 0.4}


def glow_features(sc, site, az, sfc_k, half=30.0, rmin=20.0, rmax=220.0):
    """太陽方向扇形內「被紅光照亮的雲」比例 + 頭頂低雲比例。紅度 = 紅(0.64µm)/藍(0.47µm) 輻射比。"""
    if sc.r01 is None or sc.r03 is None:
        return None
    cells = [(i, j) for i, j in _cells_within(sc, site["lat"], site["lon"], rmin, rmax, az, half)
             if sc.bt[i][j] is not None and sc.r03[i][j] is not None and sc.r01[i][j] is not None]
    if len(cells) < 30:
        return None
    v = sorted(sc.r03[i][j] for i, j in cells)
    floor = 0.12 * v[int(0.95 * (len(v) - 1))]                  # 太陽快落時訊號很弱:門檻相對於扇形內的亮度
    red = {t: 0 for t in THRS}
    cloud = lit = 0
    for i, j in cells:
        if height_km(sc.bt[i][j], sfc_k) < 0.5:
            continue
        cloud += 1
        a, b = sc.r03[i][j], sc.r01[i][j]
        if (a + b) / 2 < floor:
            continue
        lit += 1
        ratio = a / max(b, 1e-6)
        for t in THRS:
            if ratio >= t:
                red[t] += 1
    over = [(i, j) for i, j in _cells_within(sc, site["lat"], site["lon"], 0, 15) if sc.bt[i][j] is not None]
    low = sum(1 for i, j in over if 0.5 <= height_km(sc.bt[i][j], sfc_k) < 2.5)
    n = len(cells)
    return {"n": n, "cloud_frac": round(cloud / n, 3), "lit_frac": round(lit / n, 3),
            "red_frac": {str(t): round(red[t] / n, 4) for t in THRS},
            "low_overhead": round(low / len(over), 3) if over else 0.0}


def glow_index(feat, ratio_thr=1.4, norm=0.15, min_cloud=0.1):
    """0~1 連續指標:紅光雲覆蓋 × 頭頂開口 ×(畫布存在)。"""
    if not feat:
        return None
    key = str(min(THRS, key=lambda t: abs(t - ratio_thr)))
    idx = min(1.0, feat["red_frac"][key] / norm) * (1 - feat["low_overhead"])
    if feat["cloud_frac"] < min_cloud:
        idx *= feat["cloud_frac"] / min_cloud
    return idx


# ───────────────────────── 取檔(AWS S3)與組場景 ─────────────────────────
BUCKET = "noaa-himawari9"
PREFIX = "AHI-L1b-FLDK/{Y}/{m}/{d}/{H}{M}/"       # ASSUMPTION:目錄結構請用 sat_probe 驗證,可在 config.sat.prefix 覆蓋
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def s3_list(prefix, bucket=BUCKET, delimiter=None, opener=urllib.request.urlopen):
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        with opener(f"https://{bucket}.s3.amazonaws.com/?" + urllib.parse.urlencode(q), timeout=30) as r:
            root = ET.fromstring(r.read())
        keys += [e.text for e in root.iter(NS + "Key")] + [e.text for e in root.iter(NS + "Prefix") if delimiter and e.text and e.text != prefix]
        nt = root.find(NS + "NextContinuationToken")
        if nt is None or not nt.text:
            return keys
        token = nt.text


def read_tile(blob):
    """讀一個 tile NetCDF → dict(x, y, rad, [bt 係數])。需要 numpy + h5netcdf/netCDF4;否則 None。"""
    try:
        import numpy as np
    except ImportError:
        return None
    try:
        import h5netcdf
        ds = h5netcdf.File(io.BytesIO(blob), "r")
        get = lambda n: ds.variables[n]
        attrs = lambda v: dict(v.attrs)
    except ImportError:
        try:
            import netCDF4
            ds = netCDF4.Dataset("tile", memory=blob)
            get = lambda n: ds.variables[n]
            attrs = lambda v: {k: v.getncattr(k) for k in v.ncattrs()}
        except ImportError:
            return None

    def arr(n):
        v = get(n)
        a = np.array(v[:], dtype="float64")
        at = attrs(v)
        fill = at.get("_FillValue")
        if fill is not None:
            a[a == float(np.asarray(fill).ravel()[0])] = np.nan
        return a * float(np.asarray(at.get("scale_factor", 1.0)).ravel()[0]) + float(np.asarray(at.get("add_offset", 0.0)).ravel()[0])

    out = {"x": arr("x"), "y": arr("y"), "rad": arr("Rad")}
    for k in ("planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2"):
        if k in ds.variables:
            out[k] = float(np.asarray(get(k)[:]).ravel()[0])
    try:
        pv = next(v for n, v in ds.variables.items() if "perspective_point_height" in attrs(v))
        out["h"], out["lon0"] = float(attrs(pv)["perspective_point_height"]), float(attrs(pv)["longitude_of_projection_origin"])
    except StopIteration:
        pass
    return out


def assemble(tiles, bbox, step=0.04, ts=0, kind="bt"):
    """tiles: read_tile 結果列表 → 規則經緯網格的 2D list(最近鄰)。bbox=(lat_min, lon_min, lat_max, lon_max)。"""
    import numpy as np
    lat_min, lon_min, lat_max, lon_max = bbox
    lats = np.arange(lat_max, lat_min - 1e-9, -step)
    lons = np.arange(lon_min, lon_max + 1e-9, step)
    grid = np.full((len(lats), len(lons)), np.nan)
    h, lon0 = tiles[0].get("h", HSAT), tiles[0].get("lon0", LON0)
    xs = np.full(grid.shape, np.nan)
    ys = np.full(grid.shape, np.nan)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            s = to_scan(float(la), float(lo), h, lon0)
            if s:
                xs[i, j], ys[i, j] = s
    for t in tiles:
        x, y, rad = t["x"], t["y"], t["rad"]
        if kind == "bt" and all(k in t for k in ("planck_fk1", "planck_fk2", "planck_bc1", "planck_bc2")):
            with np.errstate(all="ignore"):
                rad = (t["planck_fk2"] / np.log(t["planck_fk1"] / rad + 1) - t["planck_bc1"]) / t["planck_bc2"]
        dx, dy = (x[-1] - x[0]) / (len(x) - 1), (y[-1] - y[0]) / (len(y) - 1)
        ix, iy = np.rint((xs - x[0]) / dx), np.rint((ys - y[0]) / dy)
        ok = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
        grid[ok] = rad[iy[ok].astype(int), ix[ok].astype(int)]
    return [[None if np.isnan(v) else float(v) for v in row] for row in grid], (float(lats[0]), float(lons[0]), step)


class Himawari:
    """取最近可用的 10 分鐘場景。tile 位置索引只建一次並快取(ASSUMPTION:各波段共用同一組 tile 編號)。"""
    def __init__(self, cache_dir=".cache/sat", prefix=PREFIX, bbox=(18.0, 112.0, 30.0, 128.0), step=0.04,
                 opener=urllib.request.urlopen, reader=read_tile, blob_dir=None):
        """cache_dir 只放小的 tile 位置索引(進 Actions cache);大的 NetCDF 暫存放 blob_dir(/tmp,不進 cache,6 小時後清掉)。"""
        import tempfile
        import time
        self.prefix, self.bbox, self.step, self.opener, self.reader = prefix, bbox, step, opener, reader
        self.dir = blob_dir or os.path.join(tempfile.gettempdir(), "fcf_sat")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(self.dir, exist_ok=True)
        for f in os.listdir(self.dir):
            fp = os.path.join(self.dir, f)
            if os.path.isfile(fp) and time.time() - os.path.getmtime(fp) > 6 * 3600:
                os.remove(fp)
        self.index_path = os.path.join(cache_dir, "tiles.json")
        try:
            with open(self.index_path, encoding="utf-8") as f:
                self.index = json.load(f)
        except (OSError, ValueError):
            self.index = {}

    def _slot_keys(self, ts):
        from datetime import datetime, timezone
        d = datetime.fromtimestamp(ts // 600 * 600, timezone.utc)
        return s3_list(self.prefix.format(Y=d.year, m=f"{d.month:02d}", d=f"{d.day:02d}", H=f"{d.hour:02d}", M=f"{d.minute:02d}"),
                       opener=self.opener)

    def _blob(self, key):
        p = os.path.join(self.dir, re.sub(r"[^\w.-]", "_", key))
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()
        with self.opener(f"https://{BUCKET}.s3.amazonaws.com/" + urllib.parse.quote(key), timeout=120) as r:
            b = r.read()
        with open(p, "wb") as f:
            f.write(b)
        return b

    def _tiles_for(self, keys, band, need):
        """need=(xmin,xmax,ymin,ymax) 掃描角範圍;索引未知的 tile 先讀 x/y 範圍再快取。"""
        sel = []
        for k in keys:
            m = re.search(rf"-B{band:02d}[-_].*?-T(\d+)[_.]", k) or re.search(rf"B{band:02d}.*?T(\d+)", k)
            if not m:
                continue
            tid = m.group(1)
            ext = self.index.get(tid)
            if ext is None:
                t = self.reader(self._blob(k))
                if not t:
                    return None
                ext = [float(t["x"].min()), float(t["x"].max()), float(t["y"].min()), float(t["y"].max())]
                self.index[tid] = ext
            if ext[0] <= need[1] and ext[1] >= need[0] and ext[2] <= need[3] and ext[3] >= need[2]:
                sel.append(k)
        return sel

    def scene(self, ts, want_visible=True, tries=3):
        try:
            import numpy  # noqa: F401
        except ImportError:
            return None
        la0, lo0, la1, lo1 = self.bbox
        sc = [to_scan(la, lo) for la in (la0, la1) for lo in (lo0, lo1)]
        sc = [s for s in sc if s]
        if not sc:
            return None
        need = (min(s[0] for s in sc), max(s[0] for s in sc), min(s[1] for s in sc), max(s[1] for s in sc))
        for k in range(tries):
            t = ts - k * 600
            try:
                keys = self._slot_keys(t)
                if not keys:
                    continue
                out = {}
                for name, band, kind in (("bt", 13, "bt"), ("r01", 1, "rad"), ("r03", 3, "rad")):
                    if name != "bt" and not want_visible:
                        continue
                    sel = self._tiles_for(keys, band, need)
                    tiles = [self.reader(self._blob(x)) for x in (sel or [])]
                    tiles = [x for x in tiles if x]
                    if not tiles:
                        raise RuntimeError(f"band {band} no tiles")
                    out[name], (lat_max, lon_min, step) = assemble(tiles, self.bbox, self.step, t, kind)
                with open(self.index_path, "w", encoding="utf-8") as f:
                    json.dump(self.index, f)
                return Scene(t, lat_max, lon_min, step, out["bt"], out.get("r01"), out.get("r03"))
            except Exception as e:
                log(f"[sat] slot {t} failed: {e}")
        return None


# ───────────────────────── 預覽圖(人工驗證用,純標準庫 PNG) ─────────────────────────
def write_png(path, rows):
    import struct
    import zlib
    h, w = len(rows), len(rows[0])
    raw = b"".join(b"\x00" + bytes(c for px in r for c in px) for r in rows)
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def preview(sc, path):
    """偽真彩色:紅=r03、藍=r01、綠=平均;無可見光時用亮溫灰階。"""
    vis = sc.r03 is not None and sc.r01 is not None
    flat = sorted(v for row in (sc.r03 if vis else sc.bt) for v in row if v is not None)
    if not flat:
        return False
    hi = flat[int(0.98 * (len(flat) - 1))] or 1.0
    rows = []
    for i in range(sc.ny):
        row = []
        for j in range(sc.nx):
            if vis and sc.r03[i][j] is not None and sc.r01[i][j] is not None:
                r, b = (min(1.0, sc.r03[i][j] / hi) ** 0.5, min(1.0, sc.r01[i][j] / hi) ** 0.5)
                row.append((int(255 * r), int(255 * (r + b) / 2), int(255 * b)))
            else:
                v = sc.bt[i][j]
                g = 0 if v is None else int(255 * max(0.0, min(1.0, (300 - v) / 80)))
                row.append((g, g, g))
        rows.append(row)
    write_png(path, rows)
    return True
