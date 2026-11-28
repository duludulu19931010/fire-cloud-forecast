import csv
import json
import math
import os
import random
import shutil
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fcf import astro, backfill, events, fetch, geo, learn, meteoblue, mock, obs, run, sat, scoring, store, truth  # noqa: E402
from fcf.fetch import V_PATH, V_SITE  # noqa: E402

with open(os.path.join(os.path.dirname(__file__), "..", "config", "sites.json"), encoding="utf-8") as _f:
    CFG = json.load(_f)
BALI = next(s for s in CFG["sites"] if s["id"] == "bali")
W = {"ecmwf_ifs025": .4, "icon_seamless": .3, "gfs_seamless": .3}
NOW = datetime(2026, 11, 17, 8, 0, tzinfo=timezone.utc).timestamp()


# ───────────────────────── 天文 / 幾何 / 評分 ─────────────────────────
class Astro(unittest.TestCase):
    def test_sunset_taipei_area(self):           # 對照 fc.nekolens.tw 顯示 11/18 八里日落 17:07
        ev = astro.day_events(date(2026, 11, 18), BALI["lat"], BALI["lon"], 8)
        loc = datetime.fromtimestamp(ev["sunset"], timezone.utc) + timedelta(hours=8)
        self.assertLessEqual(abs((loc.hour * 60 + loc.minute) - (17 * 60 + 7)), 3)


class Geo(unittest.TestCase):
    def test_shadow_height(self):
        self.assertAlmostEqual(geo.lit_min_height(3.0, 0), 8.73, delta=0.2)
        self.assertEqual(geo.lit_min_height(3.0, 400), 0.0)

    def test_mid_wall_blocks_but_low_wall_does_not_for_high_cloud(self):
        base = {"low": 0, "mid": 0, "high": 0}
        mid = [dict(base, mid=1.0) for _ in scoring.US]
        low = [dict(base, low=1.0) for _ in scoring.US]
        self.assertLess(geo.path_T(9.0, 25, 2.5, scoring.US, mid), 0.3)
        self.assertGreater(geo.path_T(9.0, 25, 2.5, scoring.US, low), 0.9)


def glow_pts(sc, with_air=True):
    om = mock.MockOpenMeteo(NOW, sc)
    ev = astro.day_events(date(2026, 11, 18), BALI["lat"], BALI["lon"], 8)
    az = astro.sun_pos(ev["sunset"], BALI["lat"], BALI["lon"])[1]
    k0, ks = events.key(BALI["lat"], BALI["lon"]), events.path_keys(BALI, az)
    sp = om.get([k0], V_SITE, "site")[k0]
    if with_air:
        fetch.merge(sp, om.get_air([k0])[k0])
    return ev, az, [sp] + [om.get(ks, V_PATH, "path")[k] for k in ks]


class Scoring(unittest.TestCase):
    good = {"low": 3, "mid": 25, "high": 55, "west_clear_lon": 120.3}

    def score(self, sc, **kw):
        ev, _, pts = glow_pts(sc)
        return scoring.score_glow("sunset", ev["sunset"], BALI, pts, W, **kw)

    def test_ordering(self):
        g = self.score(self.good)["score"]
        self.assertGreater(g, self.score({"low": 0, "mid": 0, "high": 0})["score"] + 15)
        self.assertLess(self.score(dict(self.good, low=99))["score"], 8)
        self.assertLess(self.score(dict(self.good, wall=("mid", 119.0, 121.0)))["score"], g * 0.6)

    def test_nowcast_blend_scores(self):
        clean = [{"low": 0, "mid": 0, "high": .55}] * 9
        wall = [{"low": 0, "mid": .9, "high": .55}] * 9
        base = self.score(self.good)["score"]
        self.assertGreaterEqual(self.score(self.good, nowcast={"covers": clean, "w": 1})["score_nc"], base)
        self.assertLess(self.score(self.good, nowcast={"covers": wall, "w": 1})["score_nc"], base * 0.3)

    def test_missing_path_points_use_consensus_and_extra_source(self):
        ev, _, pts = glow_pts(self.good)
        pts[0]["m"]["meteoblue"] = dict(pts[0]["m"]["ecmwf_ifs025"])       # 新來源只有站點資料
        r = scoring.score_glow("sunset", ev["sunset"], BALI, pts, dict(W, meteoblue=.2))
        self.assertIn("meteoblue", r["models"])

    def test_ensemble_members(self):
        om = mock.MockOpenMeteo(NOW, self.good)
        ev, az, _ = glow_pts(self.good)
        ks = [events.key(BALI["lat"], BALI["lon"])] + events.path_keys(BALI, az)
        data = om.get_ens(ks)
        mp = {n: [{"time": data[k]["time"], "m": {n: data[k]["m"][n]}} for k in ks] for n in data[ks[0]]["m"]}
        sc = scoring.score_members("sunset", ev["sunset"], BALI, mp)
        self.assertEqual(len(sc), 12)
        self.assertGreater(max(sc.values()) - min(sc.values()), 5)


# ───────────────────────── 資料解析 ─────────────────────────
class Parsers(unittest.TestCase):
    def test_meteoblue_3h_to_hourly(self):
        res = {"data_1h": {"time": ["2026-11-18 00:00", "2026-11-18 01:00"], "windspeed": [7.2, 7.2]},
               "data_3h": {"time": ["2026-11-18 00:00", "2026-11-18 03:00"], "lowclouds": [0, 30], "highclouds": [60, 0],
                           "midclouds": [0, 0]}}
        d = meteoblue.parse(res)["m"]["meteoblue"]
        self.assertEqual([round(v) for v in d["cloud_cover_low"]], [0, 10, 20, 30])
        self.assertAlmostEqual(d["wind_speed_10m"][0], 2.0)
        self.assertIsNone(meteoblue.parse({"data_1h": {"time": ["2026-11-18 00:00"], "foo": [1]}}))

    def test_ensemble_and_previous_runs(self):
        r = {"hourly": {"time": [0, 3600], "cloud_cover_low": [1, 2], "cloud_cover_low_member01": [3, 4]}}
        self.assertEqual(set(fetch.parse_ens(r, {"cloud_cover_low"}, "ecmwf_ifs025")["m"]), {"ens00", "ens01"})
        r2 = {"hourly": {"time": [0], "cloud_cover_low_previous_day1_gfs_seamless": [7]}}
        self.assertEqual(fetch.parse_prev(r2, ["cloud_cover_low"], ["gfs_seamless"], 1)["m"]["gfs_seamless"]["cloud_cover_low"], [7])

    def test_obs_flatten_new_and_old_format(self):
        new = {"StationId": "C0X", "GeoInfo": {"StationAltitude": "1600.0", "Coordinates": [
            {"CoordinateName": "TWD67", "StationLatitude": 1, "StationLongitude": 1},
            {"CoordinateName": "WGS84", "StationLatitude": 23.5, "StationLongitude": 120.7}]},
            "WeatherElement": {"AirTemperature": "14.2", "RelativeHumidity": "-99", "WindSpeed": "1.1"}}
        s = obs._flatten(new)
        self.assertEqual((s["lat"], s["lon"], s["alt"], s["T"], s["RH"]), (23.5, 120.7, 1600.0, 14.2, None))


# ───────────────────────── 衛星 ─────────────────────────
def scene(red_cloud=True, low_over=False, site=BALI):
    step, lat_max, lon_min = 0.04, 30.0, 112.0
    ny, nx = int(12 / step) + 1, int(16 / step) + 1
    bt = [[299.0] * nx for _ in range(ny)]
    r01 = [[1.0] * nx for _ in range(ny)]
    r03 = [[1.0] * nx for _ in range(ny)]
    sc = sat.Scene(0, lat_max, lon_min, step, bt, r01, r03)
    for i, j in sat._cells_within(sc, site["lat"], site["lon"], 40, 150, 250, 25):
        bt[i][j], r03[i][j], r01[i][j] = 240.0, 2.0, (1.0 if red_cloud else 2.0)
    if low_over:
        for i, j in sat._cells_within(sc, site["lat"], site["lon"], 0, 15):
            bt[i][j] = 292.0
    return sc


class Sat(unittest.TestCase):
    def test_detector_separates_red_lit_clouds(self):
        idx = lambda **kw: sat.glow_index(sat.glow_features(scene(**kw), BALI, 250.0, 299.0))
        self.assertGreater(idx(), 0.9)                       # 紅光雲 + 頭頂乾淨
        self.assertLess(idx(red_cloud=False), 0.05)          # 白雲:沒被紅光照亮
        self.assertLess(idx(low_over=True), 0.05)            # 頭頂低雲蓋住

    def test_s3_list_parses_prefixes_keys_and_pagination(self):
        import io
        ns = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'
        pages = [f'<ListBucketResult {ns}><Prefix>AHI</Prefix><CommonPrefixes><Prefix>AHI-L1b-FLDK/</Prefix></CommonPrefixes>'
                 f'<IsTruncated>true</IsTruncated><NextContinuationToken>T2</NextContinuationToken></ListBucketResult>',
                 f'<ListBucketResult {ns}><Prefix>AHI</Prefix><Contents><Key>AHI-x.nc</Key></Contents></ListBucketResult>']
        calls = []
        def opener(url, timeout=0):
            calls.append(url)
            return io.BytesIO(pages[len(calls) - 1].encode())
        out = sat.s3_list("AHI", opener=opener, delimiter="/")
        self.assertEqual(out, ["AHI-L1b-FLDK/", "AHI-x.nc"])           # 不含請求本身的 Prefix;分頁會接續取
        self.assertIn("continuation-token=T2", calls[1])

    def test_dusk_quantisation_noise_is_not_mistaken_for_glow(self):
        sc = scene()                                           # 扇形內全是高雲,但可見光只比零高 ~5 個計數(黃昏、地面已黑)
        for i, j in sat._cells_within(sc, BALI["lat"], BALI["lon"], 20, 220, 250, 30):
            sc.r03[i][j] = 0.003
            sc.r01[i][j] = 0.0 if (i + j) % 2 else 0.003       # 藍光最先消失 → 一半像素比值「無限大」
        f = sat.glow_features(sc, BALI, 250.0, 299.0)
        self.assertEqual(f["lit_frac"], 0.0)                   # 不再把雜訊當成「被照亮」
        self.assertLess(sat.glow_index(f, 1.4, lit_min=sat.LIT_DEFAULT), 0.05)

    def test_thin_bright_red_band_is_detected_and_table_is_monotone(self):
        sc = scene(red_cloud=False)
        cells = list(sat._cells_within(sc, BALI["lat"], BALI["lon"], 20, 220, 250, 30))
        for k, (i, j) in enumerate(cells):
            sc.bt[i][j] = 240.0                                # 整個扇形都是高雲(畫布)
            if k % 4 == 0:                                     # 25% 面積是「亮而偏紅」(紅 0.03、藍 0.008 → 比 3.75)
                sc.r03[i][j], sc.r01[i][j] = 0.03, 0.008
            else:                                              # 其餘是黃昏雜訊
                sc.r03[i][j], sc.r01[i][j] = 0.003, 0.0
        f = sat.glow_features(sc, BALI, 250.0, 299.0)
        self.assertGreater(sat.glow_index(f, 1.4, lit_min=sat.LIT_DEFAULT), 0.9)
        self.assertGreater(sat.glow_index(f, 3.0, lit_min=sat.LIT_DEFAULT), 0.9)
        tab = f["red_by_lit"]
        for L in tab:                                          # 紅度門檻越高、亮度門檻越高,比例只會越小
            vals = [tab[L][str(t)] for t in sat.THRS]
            self.assertEqual(vals, sorted(vals, reverse=True))
        lv = [tab[str(L)]["1.4"] for L in sat.LIT_LEVELS]
        self.assertEqual(lv, sorted(lv, reverse=True))

    def test_layer_fractions_and_png(self):
        sc = scene()
        la, lo = geo.dest(BALI["lat"], BALI["lon"], 250.0, 100)
        self.assertEqual(sat.layer_fracs(sc, la, lo, 299.0, 25.0)["high"], 1.0)
        cov = sat.nowcast_covers(sc, BALI, 250.0, 299.0)
        self.assertGreaterEqual(sum(c is not None for c in cov), 5)
        p = os.path.join(tempfile.mkdtemp(), "p.png")
        self.assertTrue(sat.preview(sc, p))
        with open(p, "rb") as f:
            self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")


# ───────────────────────── HSD 讀取層(用合成檔驗證:檔頭解析、校正換算、地理定位、S3 取檔、組場景) ─────────────────────────
import bz2  # noqa: E402
import io  # noqa: E402
import struct  # noqa: E402
import urllib.parse  # noqa: E402

REAL = {"sub_lon": 140.7, "cfac": 20466275, "lfac": 20466275, "coff": 2750.5, "loff": 2750.5, "rs": 42164.0, "req": 6378.137, "rpol": 6356.7523}
FAKE = dict(REAL, cfac=1023314, lfac=1023314, coff=275.5, loff=275.5)                    # 縮小版全盤:550×550,每段 55 行
FAKE_GRID = {20: (550, 1023314, 275.5), 10: (550, 1023314, 275.5), 5: (550, 1023314, 275.5)}
HC, CC, KB = 6.62606957e-34, 2.99792458e8, 1.3806488e-23


def planck_um(T, wl_um):
    lam = wl_um * 1e-6
    return 2 * HC * CC ** 2 / lam ** 5 / (math.exp(HC * CC / (lam * KB * T)) - 1) * 1e-6


def make_hsd(band, segno, field, ncol=550, nseg=10, nline=55, gain=None, cprime=0.002):
    wl = 10.4 if band >= 7 else 0.5
    gain = gain or (0.0005 if band >= 7 else 0.01)
    b1 = bytearray(282); struct.pack_into("<BHHB", b1, 0, 1, 282, 11, 0)
    hdr_len = 282 + 50 + 127 + 139 + 147 + 259 + 47 + 120
    struct.pack_into("<II", b1, 70, hdr_len, ncol * nline * 2)
    b2 = bytearray(50); struct.pack_into("<BHHHHB", b2, 0, 2, 50, 16, ncol, nline, 0)
    b3 = bytearray(127); struct.pack_into("<BHdIIffddddddd", b3, 0, 3, 127, FAKE["sub_lon"], FAKE["cfac"], FAKE["lfac"], FAKE["coff"], FAKE["loff"], FAKE["rs"], FAKE["req"], FAKE["rpol"], 0, 0, 0, 0)
    b4 = bytearray(139); struct.pack_into("<BH", b4, 0, 4, 139)
    b5 = bytearray(147); struct.pack_into("<BHHdHHHdd", b5, 0, 5, 147, band, wl, 16, 65535, 65534, gain, 0.0)
    if band >= 7:
        struct.pack_into("<9d", b5, 35, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, CC, HC, KB)
    else:
        struct.pack_into("<d", b5, 35, cprime)
    b6 = bytearray(259); struct.pack_into("<BH", b6, 0, 6, 259)
    first = (segno - 1) * nline + 1
    b7 = bytearray(47); struct.pack_into("<BHBBH", b7, 0, 7, 47, nseg, segno, first)
    cnt = []
    for line in range(first, first + nline):
        for col in range(1, ncol + 1):
            g = sat.pix_to_geo(col, line, FAKE)
            if g is None:
                cnt.append(65534)
                continue
            v = field(*g)
            rad = planck_um(v, wl) if band >= 7 else v / cprime
            cnt.append(max(0, min(65000, int(round(rad / gain)))))
    data = struct.pack("<%dH" % len(cnt), *cnt)
    return bz2.compress(bytes(b1 + b2 + b3 + b4 + b5 + b6 + b7 + bytes(120)) + data)


BT_F = lambda la, lo: 250.0 + (la - 18) * 3.0 + (lo - 112) * 0.5
R01_F = lambda la, lo: 0.2 + 0.01 * (la - 20)
R03_F = lambda la, lo: 1.5 * R01_F(la, lo)


class Hsd(unittest.TestCase):
    def test_geo_pixel_roundtrip_and_backside(self):
        for lat, lon in [(25.0, 121.5), (22.6, 120.3), (27.0, 125.0), (20.5, 115.0)]:
            c, l = sat.geo_to_pix(lat, lon, REAL)
            la, lo = sat.pix_to_geo(c, l, REAL)
            self.assertLess(abs(la - lat) + abs(lo - lon), 0.06)       # 誤差 ≤ 約 1 個 2 km 像素
        self.assertIsNone(sat.geo_to_pix(25.0, -60.0, REAL))           # 衛星背面

    def test_header_calibration_and_error_reporting(self):
        T = 270.0
        p = sat.parse_hsd(make_hsd(13, 4, lambda la, lo: T))
        self.assertEqual((p["band"], p["ncol"], p["nline"], p["segno"], p["first_line"]), (13, 550, 55, 4, 166))
        self.assertAlmostEqual(p["wl"], 10.4)
        self.assertAlmostEqual(sat.convert(p, int(round(planck_um(T, 10.4) / p["gain"]))), T, delta=0.05)
        v = sat.parse_hsd(make_hsd(1, 4, lambda la, lo: 0.3))
        self.assertAlmostEqual(sat.convert(v, int(round(0.3 / v["cprime"] / v["gain"]))), 0.3, delta=0.01)
        self.assertIsNone(sat.convert(p, 65535))                       # 錯誤像素
        bad = bytearray(bz2.decompress(make_hsd(13, 4, lambda la, lo: T)))
        bad[332 + 1] = 9                                               # 弄壞區塊 3 的長度欄
        with self.assertRaises(sat.HSDError) as cm:
            sat.parse_hsd(bytes(bad))
        self.assertIn("前 16 bytes", str(cm.exception))                # 錯誤訊息含十六進位傾印,方便遠端除錯

    def test_needed_segments_for_default_bbox(self):
        hs = sat.Himawari(blob_dir=tempfile.mkdtemp())
        for band in (13, 1, 3):
            self.assertEqual(hs.needed_segments(band), [3, 4])          # 與 sat-probe 在真實檔頭上的結果一致(台北在第 3 段)
        c, l = sat.geo_to_pix(25.05, 121.5, REAL)
        self.assertEqual((c, l), (1824, 1446))                         # 與 sat-probe 對真實 B13 檔頭算出的台北 col/line 一致

    def test_visible_albedo_never_negative_and_preview_survives(self):
        p = {"band": 1, "gain": 0.3774, "const": -7.547, "cprime": 0.00156, "err": 65535, "out": 65534}
        self.assertEqual(sat.convert(p, 5), 0.0)                       # 暗像素:校正偏移算出負輻射 → 夾成 0
        sc = scene()
        sc.r03[0][0], sc.r01[0][0] = -1.0, 0.0
        self.assertTrue(sat.preview(sc, os.path.join(tempfile.mkdtemp(), "p.png")))

    def test_preview_creates_directory_draws_outline_and_scales(self):
        sc = scene()
        path = os.path.join(tempfile.mkdtemp(), "nested", "dir", "p.png")        # 目錄不存在也要能寫(sat-probe 實際踩到的 bug)
        self.assertTrue(sat.preview(sc, path, outline=sat.TAIWAN_OUTLINE, scale=2))
        with open(path, "rb") as f:
            hdr = f.read(24)
        self.assertEqual(struct.unpack(">II", hdr[16:24]), (sc.nx * 2, sc.ny * 2))
        dark = scene()
        dark.r03 = [[0.001] * dark.nx for _ in range(dark.ny)]
        dark.r01 = [[0.001] * dark.nx for _ in range(dark.ny)]                    # 黃昏:可見光太暗 → 改畫紅外灰階,不當掉
        self.assertTrue(sat.preview(dark, os.path.join(tempfile.mkdtemp(), "d.png")))

    def _fake(self, hs_grid):
        bbox, ts = (23.0, 119.0, 26.5, 123.5), datetime(2026, 11, 23, 9, 5, tzinfo=timezone.utc).timestamp()
        blobs = {}

        def opener(url, timeout=0):
            if "list-type" in url:
                pre = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["prefix"][0]
                ns = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'
                body = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in blobs if k.startswith(pre))
                return io.BytesIO(f"<ListBucketResult {ns}>{body}</ListBucketResult>".encode())
            return io.BytesIO(blobs[urllib.parse.unquote(urllib.parse.urlparse(url).path.lstrip("/"))])
        hs = sat.Himawari(bbox=bbox, step=0.1, opener=opener, grid=hs_grid, blob_dir=tempfile.mkdtemp())
        truth_hs = sat.Himawari(bbox=bbox, step=0.1, grid=FAKE_GRID, blob_dir=tempfile.mkdtemp())
        for band, f in ((13, BT_F), (1, R01_F), (3, R03_F)):
            for seg in sorted(set(hs.needed_segments(band)) | set(truth_hs.needed_segments(band))):
                res = sat.band_res(band)
                blobs[f"AHI-L1b-FLDK/2026/11/23/0900/HS_H09_20261123_0900_B{band:02d}_FLDK_R{res:02d}_S{seg:02d}10.DAT.bz2"] = make_hsd(band, seg, f)
        return hs, ts + 120, bbox

    def test_scene_from_fake_s3(self):
        hs, ts, bbox = self._fake(FAKE_GRID)
        sc = hs.scene(ts)
        self.assertIsNotNone(sc, hs.last_error)
        cells = [(i, j) for i in range(sc.ny) for j in range(sc.nx)]
        ok = [(i, j) for i, j in cells if sc.bt[i][j] is not None]
        self.assertGreater(len(ok) / len(cells), 0.95)
        self.assertLess(max(abs(sc.bt[i][j] - BT_F(*sc.latlon(i, j))) for i, j in ok), 2.0)
        ratios = sorted(sc.r03[i][j] / sc.r01[i][j] for i, j in ok if sc.r01[i][j])
        self.assertAlmostEqual(ratios[len(ratios) // 2], 1.5, delta=0.1)       # 反照率比 = 設定的 1.5
        self.assertIsNone(sat.Himawari(bbox=bbox, opener=lambda *a, **k: io.BytesIO(b"<ListBucketResult/>"), grid=FAKE_GRID,
                                       blob_dir=tempfile.mkdtemp()).scene(ts, tries=1))      # 該時段沒檔 → None,不丟例外

    def test_wrong_grid_table_is_self_corrected_from_headers(self):
        wrong = {k: (v[0], v[1] // 2, v[2]) for k, v in FAKE_GRID.items()}          # 規格表錯 2 倍(就是 sat-probe 抓到的那種錯)
        hs, ts, bbox = self._fake(wrong)
        right = sat.Himawari(bbox=bbox, grid=FAKE_GRID, blob_dir=tempfile.mkdtemp())
        self.assertNotEqual(hs.needed_segments(13), right.needed_segments(13))      # 預估確實選錯段
        sc = hs.scene(ts)
        self.assertIsNotNone(sc, hs.last_error)
        filled = sum(v is not None for r in sc.bt for v in r) / (sc.ny * sc.nx)
        self.assertGreater(filled, 0.95)                                            # 讀到檔頭後自動補抓正確的段


# ───────────────────────── 自我學習(以合成資料驗證「真的學得到」) ─────────────────────────
class Synth:
    def __init__(self, tc, kind="sunset"):
        self.d, self.kind = tempfile.mkdtemp(), kind
        for sub in ("history", "truth", "obs"):
            os.makedirs(os.path.join(self.d, sub))
        self.hist, self.labels, self.truth = [], [], []

    def events(self, n, true_p, sites=("a", "b", "c"), noise=lambda lead: 3 + lead / 6, models=None, nc=None, seed=1,
               factors=None, label=True):
        rng = random.Random(seed)
        for i in range(n):
            day = (date(2026, 1, 1) + timedelta(days=i // len(sites))).isoformat()
            site = sites[i % len(sites)]
            S = rng.uniform(0, 90)
            y = 1 if rng.random() < true_p(S, site) else 0
            if label:
                self.labels.append({"date": day, "site": site, "kind": self.kind, "label": 2 if y else 0})
            for lead in (3.0, 12.0):
                h = {"gen_h": i * 100 + int(lead), "site": site, "kind": self.kind, "date": day, "lead_h": lead,
                     "score": max(0, S + rng.gauss(0, noise(lead)))}
                if models:
                    h["models"] = {m: max(0, S + rng.gauss(0, sd)) for m, sd in models.items()}
                if nc is not None and lead <= 3.5:
                    h["score_nc"] = max(0, S + rng.gauss(0, nc))
                if factors:
                    h["factors"] = factors(S)
                self.hist.append(h)

    def write(self):
        store.append_ndjson(os.path.join(self.d, "history", "x.ndjson"), self.hist)
        store.append_ndjson(os.path.join(self.d, "truth", "x.ndjson"), self.truth)
        with open(os.path.join(self.d, "labels.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, ["date", "site", "kind", "label"])
            w.writeheader()
            w.writerows(self.labels)
        return self.d


sig = lambda x: 1 / (1 + math.exp(-x))


class Learn(unittest.TestCase):
    def tearDown(self):
        for d in getattr(self, "dirs", []):
            shutil.rmtree(d, ignore_errors=True)

    def run_learn(self, syn, **kw):
        self.dirs = getattr(self, "dirs", []) + [syn.d]
        syn.write()
        return learn.update(syn.d, NOW + 86400 * 400, **kw)

    def test_few_events_keep_prior(self):
        s = Synth(0)
        s.events(9, lambda S, _: sig(0.09 * (S - 58)))
        cal, rep = self.run_learn(s)
        self.assertEqual(cal["sunset"]["champion"], "prior")

    def test_learns_true_calibration_and_beats_prior(self):
        s = Synth(0)
        s.events(150, lambda S, _: sig(0.09 * (S - 58)))                 # 真實中點 58,先驗是 42
        cal, rep = self.run_learn(s)
        wf = rep["sunset"]["walk_forward_brier"]
        self.assertNotEqual(cal["sunset"]["champion"], "prior")
        self.assertLess(min(v for k, v in wf.items() if k != "prior"), wf["prior"])
        row = lambda S: {"s": S, "lead": 12.0, "site": "a", "factors": None}
        m = cal["sunset"]["model"]
        self.assertLess(learn.predict(m, row(15)), 0.15)
        self.assertGreater(learn.predict(m, row(90)), 0.8)
        self.assertGreater(rep["sunset"]["BSS"], 0.05)                   # 樣本外技巧分數為正
        self.assertTrue(rep["sunset"]["oos"])

    def test_site_offsets_are_learned(self):
        s = Synth(0)
        s.events(240, lambda S, st: sig(0.09 * (S - 50) + (2.5 if st == "a" else 0.0)), seed=3)
        cal, rep = self.run_learn(s)
        wf = rep["sunset"]["walk_forward_brier"]
        self.assertLess(wf["logit_sls"], wf["logit_s"])                  # 站點偏差確實改善樣本外表現

    def test_factor_exponents_candidate_wins_when_factor_matters(self):
        rng = random.Random(5)
        s = Synth(0)
        facs = {}
        def f(S):
            v = {"canvas": max(0.05, S / 100), "overhead_open": rng.uniform(0.1, 1), "air": rng.uniform(.5, 1),
                 "light_path": rng.uniform(.2, 1), "color": rng.uniform(.4, 1)}
            return v
        # 真實發生率額外強烈依賴 overhead_open(模型分數沒完全反映)
        ev_rng = random.Random(7)
        for i in range(200):
            day = (date(2026, 1, 1) + timedelta(days=i // 3)).isoformat()
            fc = f(ev_rng.uniform(0, 90))
            S = 100 * fc["canvas"] * fc["overhead_open"]
            y = 1 if ev_rng.random() < sig(3.0 * math.log(fc["overhead_open"]) + 0.05 * (S - 20)) else 0
            s.labels.append({"date": day, "site": "abc"[i % 3], "kind": "sunset", "label": 2 if y else 0})
            for lead in (3.0, 12.0):
                s.hist.append({"gen_h": i * 100 + int(lead), "site": "abc"[i % 3], "kind": "sunset", "date": day,
                               "lead_h": lead, "score": S, "factors": fc})
        cal, rep = self.run_learn(s)
        wf = rep["sunset"]["walk_forward_brier"]
        self.assertIn("logit_fac", wf)
        self.assertLess(wf["logit_fac"], wf["logit_sl"])

    def test_source_weights_favor_accurate_model(self):
        s = Synth(0)
        s.events(90, lambda S, _: sig(0.09 * (S - 50)), models={"good": 3.0, "bad": 28.0})
        cal, rep = self.run_learn(s, base_weights={"good": 0.5, "bad": 0.5})
        mw = cal["sunset"]["model_weights"]
        self.assertGreater(mw["good"], mw["bad"] * 1.5)

    def test_nowcast_weight_learned(self):
        s = Synth(0)
        s.events(80, lambda S, _: sig(0.09 * (S - 50)), noise=lambda lead: 18 if lead <= 3.5 else 6, nc=2.0)
        cal, rep = self.run_learn(s)
        self.assertGreaterEqual(cal["sunset"]["nowcast_w"]["short"], 0.75)

    def test_detector_threshold_learned_from_human_labels(self):
        s = Synth(0)
        rng = random.Random(2)
        for i in range(40):
            pos = i % 2 == 0
            rf = {str(t): 0.0 for t in sat.THRS}
            for t in sat.THRS:                    # 正例:各紅度閾值都有紅光雲;負例:只有輕微偏紅(≤1.4)的雲
                rf[str(t)] = 0.2 if (pos or t <= 1.4) else 0.01
            ev = {"red_frac": rf, "cloud_frac": 0.4, "low_overhead": 0.0, "lit_frac": 0.3, "n": 500}
            day = (date(2026, 2, 1) + timedelta(days=i)).isoformat()
            s.truth.append({"site": "a", "kind": "sunset", "date": day, "src": "sat", "y": 0, "ev": ev, "ts": 0})
            s.labels.append({"date": day, "site": "a", "kind": "sunset", "label": 2 if pos else 0})
        cal, rep = self.run_learn(s)
        self.assertGreaterEqual(cal["detector"]["ratio_thr"], 1.5)
        self.assertGreater(cal["truth_w"]["sat"], 0.8)                   # 一致度高 → 衛星真值權重高

    def test_observation_truth_is_used_without_any_human_label(self):
        s = Synth(0, kind="sea_of_clouds")
        rng = random.Random(4)
        s.events(60, lambda S, _: 0.5, label=False)
        for h in s.hist:                          # 以測站真值取代人工回報
            pass
        for k in {(h["site"], h["date"]) for h in s.hist}:
            s.truth.append({"site": k[0], "kind": "sea_of_clouds", "date": k[1], "src": "cwa", "y": rng.random() < .5,
                            "ev": {}, "ts": 0})
        cal, rep = self.run_learn(s)
        self.assertGreater(rep["sea_of_clouds"]["by_src"]["cwa"], 40)

    def test_choose_rolls_back_and_needs_margin(self):
        self.assertEqual(learn.choose({"prior": .2, "logit_s": .25}, "logit_s")[0], "prior")      # 現任變差 → 回退
        self.assertEqual(learn.choose({"prior": .2, "logit_s": .199}, "prior")[0], "prior")       # 進步不到 2% → 不換
        self.assertEqual(learn.choose({"prior": .2, "logit_s": .15}, "prior")[0], "logit_s")

    def test_pav_monotone_and_finalize_uses_learned_nowcast_weight(self):
        ys = [y for _, y in learn.pav([1, 2, 3, 4, 5], [1, 0, 1, 0, 1], [1] * 5)]
        self.assertEqual(ys, sorted(ys))
        r = {"score": 10.0, "score_nc": 70.0, "models": {"a": 1.0}, "factors": {}}
        learn.finalize(r, "sunset", 1.0, {"sunset": {"nowcast_w": {"short": 1.0}}}, "a")
        self.assertEqual(r["score"], 70.0)
        self.assertEqual(r["score_fc"], 10.0)


# ───────────────────────── 觀測型真值 ─────────────────────────
class Truth(unittest.TestCase):
    def test_sea_truth_rules(self):
        smp = lambda *sc, ws=2.0: [{"ts": 100 + i, "score": s, "ws": ws} for i, s in enumerate(sc)]
        self.assertEqual(truth.sea_truth(smp(70, 80, 20), "sea_of_clouds", 0, 1000)["y"], 1)
        self.assertEqual(truth.sea_truth(smp(70, 20, 10), "sea_of_clouds", 0, 1000)["y"], 0)
        self.assertEqual(truth.sea_truth(smp(90, 90, 90, ws=9.0), "ryuli", 0, 1000)["y"], 0)    # 風太大 → 琉璃光不成立
        self.assertIsNone(truth.sea_truth([], "ryuli", 0, 1000))

    def test_build_from_logged_stations_and_fake_satellite(self):
        d = tempfile.mkdtemp()
        now = datetime(2026, 11, 17, 13, 0, tzinfo=timezone.utc).timestamp()
        sea_site = next(s for s in CFG["sites"] if s["id"] == "alishan_shizhuo")
        e = events._event(sea_site, "sea_of_clouds", date(2026, 11, 17), 8)
        store.append_ndjson(os.path.join(d, "obs", "m.ndjson"),
                            [{"ts": int(e["t0"] + 1800 * i), "site": "alishan_shizhuo", "score": 85.0, "ws": 1.0} for i in range(4)])

        class FakeHS:
            def scene(self, t, want_visible=True): return scene()
        gl = events._event(BALI, "sunset", date(2026, 11, 17), 8)
        rows = truth.build(CFG, now, d, {("alishan_shizhuo", "sea_of_clouds", "2026-11-17"), ("bali", "sunset", "2026-11-17")},
                           FakeHS(), lambda s: 299.0, None)
        by = {r["kind"]: r for r in rows}
        self.assertEqual(by["sea_of_clouds"]["src"], "cwa")
        self.assertEqual(by["sea_of_clouds"]["y"], 1)
        self.assertEqual(by["sunset"]["src"], "sat")
        self.assertIn("red_frac", by["sunset"]["ev"])
        self.assertEqual(truth.row_y(by["sunset"], {"ratio_thr": 1.4, "tau": 0.4}), 1)
        self.assertEqual(truth.row_y(by["sunset"], {"ratio_thr": 2.0, "tau": 1.01}), 0)           # 重新判定用的是存下來的證據


# ───────────────────────── 端到端 ─────────────────────────
class FakeHS:
    def scene(self, t, want_visible=True):
        return scene()


class EndToEnd(unittest.TestCase):
    def test_hourly_runs_with_nowcast_truth_and_learning(self):
        d = tempfile.mkdtemp()
        try:
            for i, t in enumerate(("2026-11-17T08:00:00Z", "2026-11-17T09:00:00Z", "2026-11-17T13:00:00Z")):
                self.assertEqual(run.main(["--mock", "--now", t, "--out", d], hs=FakeHS()), 0)
                if i == 0:                                    # 第一次執行時日落在 3.5h 內 → 應啟用衛星 nowcast 與集合預報
                    meta = store.load_json(os.path.join(d, "meta.json"))
                    self.assertTrue(meta["sources"]["himawari_nowcast"])
                    self.assertGreaterEqual(meta["sources"]["ensemble_events"], 1)
            nx = store.load_json(os.path.join(d, "next24h.json"))
            self.assertEqual(nx["horizon_h"], 24)
            self.assertTrue(all(-1 <= e["lead_h"] <= 24.5 for e in nx["events"]))
            self.assertEqual(set(nx["summary"]), {"sunset", "sunrise", "sea_of_clouds", "ryuli"})
            tr = list(store.read_ndjson(os.path.join(d, "truth")))
            self.assertTrue(tr and all(r["src"] == "sat" for r in tr))                          # 事件結束後自動建立觀測型真值
            hist = list(store.read_ndjson(os.path.join(d, "history")))
            self.assertTrue(any(h.get("score_nc") is not None for h in hist))                  # 保存「純預報」與「含 nowcast」兩種分數
            self.assertTrue(any(h.get("ens") is not None for h in hist))
            self.assertTrue(os.path.exists(os.path.join(d, "REPORT.md")))
            with open(os.path.join(d, "charts", "bali.svg"), encoding="utf-8") as f:
                svg = f.read()
            import xml.dom.minidom
            xml.dom.minidom.parseString(svg)                                                    # meteogram 是合法 SVG
        finally:
            shutil.rmtree(d)

    def test_backfill_creates_cold_start_history(self):
        d = tempfile.mkdtemp()
        try:
            nh, nt = backfill.run(CFG, NOW, mock.MockOpenMeteo(NOW), d, days=5, hs=FakeHS())
            self.assertGreater(nh, 10)
            self.assertGreater(nt, 10)
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
