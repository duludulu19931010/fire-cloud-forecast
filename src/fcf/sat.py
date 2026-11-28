"""Himawari-9 衛星:① nowcast 雲層(紅外亮溫→雲頂高)② 「火燒雲真的發生了嗎」偵測器(可見光紅/藍反照率比)。
資料:NOAA 在 AWS 公開 S3(noaa-himawari9)的 AHI-L1b-FLDK —— JMA 原生 HSD 格式(bz2 壓縮、每波段切 10 段、10 分鐘一景)。
目錄 AHI-L1b-FLDK/年/月/日/時分/ 與檔名 HS_H09_日期_時分_Bxx_FLDK_Rxx_Sxxxx.DAT.bz2 已由 sat-probe 在 GitHub 上實測確認。
HSD 檔頭/校正/地理定位依 JMA「Himawari Standard Data User's Guide」的格式寫成,純標準庫(bz2 + struct),不需 numpy。
已由 sat-probe 對真實檔案驗證:目錄/檔名、檔頭區塊、波長、CFAC/COFF、校正係數、紅外亮溫範圍(220–292 K)。
⚠ 尚待驗證:像素定位對真實影像(海岸線)的準確度、日落時可見光反照率的數值、偵測器閾值(先驗,需人工回報校準)。
"""
import bz2
import json
import math
import os
import re
import struct
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from array import array

from . import geo
from .fetch import log

BUCKET = "noaa-himawari9"
PREFIX = "AHI-L1b-FLDK/{Y}/{m}/{d}/{H}{M}/"                 # 已實測確認
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
SEGS = 10
GRID = {20: (5500, 20466275, 2750.5), 10: (11000, 40932549, 5500.5), 5: (22000, 81865099, 11000.5)}   # 全盤行數 / CFAC=LFAC / COFF=LOFF(sat-probe 對真實檔頭實測;選段預估用,實際換算一律以檔頭為準)
BLOCKS = ((1, 282), (2, 50), (3, 127), (4, 139), (5, 147), (6, 259), (7, 47))                          # HSD 檔頭前 7 個固定長度區塊


def band_res(b):
    return 5 if b == 3 else 10 if b in (1, 2, 4) else 20


class HSDError(Exception):
    pass


def parse_hsd(blob):
    """HSD 一個波段的一段 → 參數 + 計數值。blob 可為 bz2 或已解壓。"""
    raw = bz2.decompress(blob) if blob[:3] == b"BZh" else blob
    offs, off = {}, 0
    for no, ln in BLOCKS:
        if off + 3 > len(raw):
            raise HSDError(f"檔案太短,讀到區塊 {no} 前就結束(長度 {len(raw)})")
        n, l = struct.unpack_from("<BH", raw, off)
        if n != no or l != ln:
            raise HSDError(f"區塊 {no} 對不上:預期長度 {ln},實際 編號={n} 長度={l},offset={off},前 16 bytes={raw[off:off + 16].hex()}")
        offs[no] = off
        off += ln
    e = ">" if raw[5] == 1 else "<"
    total_hdr, total_data = struct.unpack_from(e + "II", raw, 70)
    _, _, bits, ncol, nline, _ = struct.unpack_from(e + "BHHHHB", raw, offs[2])
    (_, _, sub_lon, cfac, lfac, coff, loff, rs, req, rpol, _, _, _, _) = struct.unpack_from(e + "BHdIIffddddddd", raw, offs[3])
    (_, _, band, wl, _, err, outp, gain, const) = struct.unpack_from(e + "BHHdHHHdd", raw, offs[5])
    p = {"band": band, "wl": wl, "ncol": ncol, "nline": nline, "bits": bits, "sub_lon": sub_lon, "cfac": cfac, "lfac": lfac,
         "coff": coff, "loff": loff, "rs": rs, "req": req, "rpol": rpol, "err": err, "out": outp, "gain": gain, "const": const}
    if band >= 7:
        p.update(zip(("c0", "c1", "c2", "C0", "C1", "C2", "cc", "h", "k"), struct.unpack_from(e + "9d", raw, offs[5] + 35)))
    else:
        p["cprime"] = struct.unpack_from(e + "d", raw, offs[5] + 35)[0]
    p["nseg"], p["segno"], p["first_line"] = struct.unpack_from(e + "BBH", raw, offs[7] + 3)
    if bits != 16 or ncol <= 0 or nline <= 0 or len(raw) < total_hdr + ncol * nline * 2:
        raise HSDError(f"資料區不合理:bits={bits} ncol={ncol} nline={nline} total_hdr={total_hdr} 檔案長度={len(raw)}")
    counts = array("H")
    counts.frombytes(raw[total_hdr: total_hdr + ncol * nline * 2])
    if e == ">" and sys.byteorder == "little":
        counts.byteswap()
    p["counts"] = counts
    return p


def geo_to_pix(lat, lon, p):
    """經緯度 → (column, line)(1 起算);在衛星背面回傳 None。CGMS/HSD 規格公式。"""
    d2r = math.pi / 180
    req, rpol, rs = p["req"], p["rpol"], p["rs"]
    phi = math.atan((rpol ** 2 / req ** 2) * math.tan(lat * d2r))
    e2 = (req ** 2 - rpol ** 2) / req ** 2
    re_ = rpol / math.sqrt(1 - e2 * math.cos(phi) ** 2)
    dl = (lon - p["sub_lon"]) * d2r
    px, py, pz = re_ * math.cos(phi) * math.cos(dl), re_ * math.cos(phi) * math.sin(dl), re_ * math.sin(phi)
    r1, r2, r3 = rs - px, -py, pz
    if px * r1 + py * r2 - pz * r3 <= 0:
        return None
    rn = math.sqrt(r1 * r1 + r2 * r2 + r3 * r3)
    x, y = math.degrees(math.atan(-r2 / r1)), math.degrees(math.asin(-r3 / rn))
    return (int(math.floor(p["coff"] + x * 2 ** -16 * p["cfac"] + 0.5)),      # COFF/LOFF 為 xxxx.5:整個式子四捨五入成整數像素號
            int(math.floor(p["loff"] + y * 2 ** -16 * p["lfac"] + 0.5)))


def pix_to_geo(col, line, p):
    x = math.radians((col - p["coff"]) / (2 ** -16 * p["cfac"]))
    y = math.radians((line - p["loff"]) / (2 ** -16 * p["lfac"]))
    rs, req, rpol = p["rs"], p["req"], p["rpol"]
    q = math.cos(y) ** 2 + (req ** 2 / rpol ** 2) * math.sin(y) ** 2
    sd2 = (rs * math.cos(x) * math.cos(y)) ** 2 - q * (rs ** 2 - req ** 2)
    if sd2 < 0:
        return None
    sn = (rs * math.cos(x) * math.cos(y) - math.sqrt(sd2)) / q
    s1, s2, s3 = rs - sn * math.cos(x) * math.cos(y), sn * math.sin(x) * math.cos(y), -sn * math.sin(y)
    return (math.degrees(math.atan((req ** 2 / rpol ** 2) * s3 / math.hypot(s1, s2))),
            math.degrees(math.atan2(s2, s1)) + p["sub_lon"])


def convert(p, cnt):
    """計數值 → 可見光:反照率;紅外:亮度溫度 K。錯誤/掃描區外像素回傳 None。"""
    if cnt == p["err"] or cnt == p["out"]:
        return None
    rad = p["gain"] * cnt + p["const"]
    if p["band"] <= 6:
        return max(0.0, p["cprime"] * rad)             # 暗像素的校正偏移會算出負值,反照率不可能為負
    h, c, k = p.get("h") or 6.62606957e-34, p.get("cc") or 2.99792458e8, p.get("k") or 1.3806488e-23
    lam, I = p["wl"] * 1e-6, rad * 1e6
    if I <= 0:
        return None
    te = (h * c / (k * lam)) / math.log(2 * h * c * c / (lam ** 5 * I) + 1)
    return p["c0"] + p["c1"] * te + p["c2"] * te * te


def build_scene(bands, bbox, step, ts):
    """bands: {波段: [parse_hsd 結果...]} → Scene(規則經緯網格,最近鄰取樣)。"""
    lat_min, lon_min, lat_max, lon_max = bbox
    ny, nx = int(round((lat_max - lat_min) / step)) + 1, int(round((lon_max - lon_min) / step)) + 1
    out = {}
    for name, band in (("bt", 13), ("r01", 1), ("r03", 3)):
        segs = bands.get(band)
        if not segs:
            continue
        p0 = segs[0]
        by_first = {s["first_line"]: s for s in segs}
        nl, grid = p0["nline"], []
        for i in range(ny):
            row, lat = [], lat_max - i * step
            for j in range(nx):
                v = None
                pix = geo_to_pix(lat, lon_min + j * step, p0)
                if pix and 1 <= pix[0] <= p0["ncol"]:
                    s = by_first.get(((pix[1] - 1) // nl) * nl + 1)
                    if s is not None:
                        v = convert(s, s["counts"][(pix[1] - s["first_line"]) * s["ncol"] + pix[0] - 1])
                row.append(v)
            grid.append(row)
        out[name] = grid
    if "bt" not in out:
        return None
    return Scene(ts, lat_max, lon_min, step, out["bt"], out.get("r01"), out.get("r03"))


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


# ───────────────────────── 取檔(AWS S3) ─────────────────────────
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


def from_config(cfg, cache_dir):
    c = (cfg or {}).get("sat") or {}
    return Himawari(cache_dir, c.get("prefix") or PREFIX, tuple(c["bbox"]) if c.get("bbox") else (20.5, 115.0, 27.5, 125.0),
                    step=c.get("step", 0.04))


class Himawari:
    """取最近可用的 10 分鐘場景。只下載涵蓋 bbox 的那幾段;大檔暫存在 /tmp(6 小時後清除);組好的場景在同一次執行內共用。"""
    def __init__(self, cache_dir=".cache/sat", prefix=PREFIX, bbox=(20.5, 115.0, 27.5, 125.0), step=0.04,
                 opener=urllib.request.urlopen, grid=None, blob_dir=None):
        self.prefix, self.bbox, self.step, self.opener, self.grid = prefix, bbox, step, opener, grid or GRID
        self.dir = blob_dir or os.path.join(tempfile.gettempdir(), "fcf_sat")
        os.makedirs(self.dir, exist_ok=True)
        for f in os.listdir(self.dir):
            fp = os.path.join(self.dir, f)
            if os.path.isfile(fp) and time.time() - os.path.getmtime(fp) > 6 * 3600:
                os.remove(fp)
        self._scenes, self.last_error = {}, None

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
        with self.opener(f"https://{BUCKET}.s3.amazonaws.com/" + urllib.parse.quote(key), timeout=180) as r:
            b = r.read()
        with open(p, "wb") as f:
            f.write(b)
        return b

    def needed_segments(self, band, hdr=None):
        """bbox 落在哪幾段。hdr=None 用規格預設值估;給 parse_hsd 結果則用真實檔頭(更可靠)。"""
        if hdr is None:
            total, cfac, coff = self.grid[band_res(band)]
            nl = total // SEGS
            p = {"sub_lon": 140.7, "cfac": cfac, "lfac": cfac, "coff": coff, "loff": coff, "rs": 42164.0, "req": 6378.137, "rpol": 6356.7523}
        else:
            nl, p = hdr["nline"], hdr
        la0, lo0, la1, lo1 = self.bbox
        lines = [pix[1] for a in range(5) for b in range(5)
                 if (pix := geo_to_pix(la0 + (la1 - la0) * a / 4, lo0 + (lo1 - lo0) * b / 4, p))]
        if not lines:
            return []
        return list(range((min(lines) - 1) // nl + 1, (max(lines) - 1) // nl + 2))

    def scene(self, ts, want_visible=True, tries=3):
        slot0 = int(ts // 600 * 600)
        for k in range(tries):
            t = slot0 - k * 600
            ck = (t, want_visible)
            if ck in self._scenes:
                return self._scenes[ck]
            try:
                keys = self._slot_keys(t)
                if not keys:
                    raise RuntimeError("該時段目錄沒有檔案")
                idx = {}
                for key in keys:
                    m = re.search(r"_B(\d\d)_FLDK_R\d\d_S(\d\d)\d\d\.DAT", key)
                    if m:
                        idx[(int(m.group(1)), int(m.group(2)))] = key
                bands = {}
                for band in ((13, 1, 3) if want_visible else (13,)):
                    segs = self.needed_segments(band)
                    if not segs:
                        raise RuntimeError(f"波段 {band}:bbox 不在衛星視野內")
                    parsed = {}

                    def load(seg, band=band, parsed=parsed):
                        if (band, seg) not in idx:
                            raise RuntimeError(f"波段 {band} 缺第 {seg} 段(目錄只有 {sorted(s for b, s in idx if b == band)})")
                        parsed[seg] = parse_hsd(self._blob(idx[(band, seg)]))
                    for seg in segs:
                        load(seg)
                    real = self.needed_segments(band, parsed[segs[0]])          # 用真實檔頭重算;規格表若有誤差,在這裡自我修正
                    for seg in real:
                        if seg not in parsed:
                            load(seg)
                    bands[band] = [parsed[seg] for seg in real]
                sc = build_scene(bands, self.bbox, self.step, t)
                self._scenes[ck] = sc
                if sc:
                    return sc
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                log(f"[sat] slot {t} failed: {self.last_error}")
        return None


# ───────────────────────── 預覽圖(人工驗證用,純標準庫 PNG) ─────────────────────────
# 台灣本島粗略海岸線(約 ±5–10 km),疊在預覽圖上檢查衛星像素定位有沒有對準:白天無雲處陸地與海面反差明顯,紅線應落在邊界上。
TAIWAN_OUTLINE = [(25.30, 121.54), (25.00, 122.00), (24.60, 121.87), (24.00, 121.62), (23.50, 121.45), (23.10, 121.38),
                  (22.75, 121.15), (22.35, 120.90), (21.90, 120.85), (22.05, 120.70), (22.45, 120.52), (22.62, 120.27),
                  (23.00, 120.10), (23.45, 120.13), (23.75, 120.18), (24.28, 120.50), (24.60, 120.80), (24.85, 120.92),
                  (25.05, 121.18), (25.18, 121.40)]


def write_png(path, rows):
    import struct
    import zlib
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    h, w = len(rows), len(rows[0])
    raw = b"".join(b"\x00" + bytes(c for px in r for c in px) for r in rows)

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def preview(sc, path, outline=None, scale=3):
    """偽真彩色:紅=r03、藍=r01、綠=平均;可見光太暗(黃昏/夜間)的像素改用紅外亮溫灰階(越冷越亮)。outline 為海岸線疊圖。"""
    vis = sc.r03 is not None and sc.r01 is not None
    flat = sorted(v for row in (sc.r03 if vis else sc.bt) for v in row if v is not None)
    if not flat:
        return False
    hi = max(flat[int(0.98 * (len(flat) - 1))], 1e-6)
    mark = set()
    if outline:
        pts = outline + [outline[0]]
        for (a, b), (c, d) in zip(pts, pts[1:]):
            n = max(2, int(max(abs(c - a), abs(d - b)) / 0.02))
            for k in range(n + 1):
                cell = sc.cell(a + (c - a) * k / n, b + (d - b) * k / n)
                if cell:
                    mark.add(cell)
    rows = []
    for i in range(sc.ny):
        row = []
        for j in range(sc.nx):
            if (i, j) in mark:
                px = (255, 40, 40)
            elif vis and sc.r03[i][j] is not None and sc.r01[i][j] is not None and max(sc.r03[i][j], sc.r01[i][j]) >= 0.02:
                r = max(0.0, min(1.0, sc.r03[i][j] / hi)) ** 0.5
                b = max(0.0, min(1.0, sc.r01[i][j] / hi)) ** 0.5
                px = (int(255 * r), int(255 * (r + b) / 2), int(255 * b))
            else:
                v = sc.bt[i][j]
                g = 0 if v is None else int(255 * max(0.0, min(1.0, (300 - v) / 80)))
                px = (g, g, g)
            row.extend([px] * scale)
        rows.extend([row] * scale)
    write_png(path, rows)
    return True
