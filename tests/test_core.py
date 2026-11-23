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
    def test_projection_roundtrip(self):
        for lat, lon in [(25.0, 121.5), (22.6, 120.3), (30, 128), (18, 112)]:
            g = sat.to_geo(*sat.to_scan(lat, lon))
            self.assertAlmostEqual(g[0], lat, places=5)
            self.assertAlmostEqual(g[1], lon, places=5)

    def test_assemble_nearest_neighbour(self):
        import numpy as np
        c = [sat.to_scan(la, lo) for la in (19.5, 27.5) for lo in (117.5, 124.5)]
        x = np.linspace(min(a[0] for a in c), max(a[0] for a in c), 400)
        y = np.linspace(max(a[1] for a in c), min(a[1] for a in c), 500)
        f = lambda la, lo: 250.0 + (la - 18) * 3.0 + (lo - 112) * 0.5
        rad = np.array([[f(*sat.to_geo(xx, yy)) for xx in x] for yy in y])
        grid, (la0, lo0, st) = sat.assemble([{"x": x, "y": y, "rad": rad}], (20.0, 118.0, 27.0, 124.0), 0.1, 0, "rad")
        err = max(abs(grid[i][j] - f(la0 - i * st, lo0 + j * st)) for i in range(len(grid)) for j in range(len(grid[0])))
        self.assertLess(err, 0.1)

    def test_detector_separates_red_lit_clouds(self):
        idx = lambda **kw: sat.glow_index(sat.glow_features(scene(**kw), BALI, 250.0, 299.0))
        self.assertGreater(idx(), 0.9)                       # 紅光雲 + 頭頂乾淨
        self.assertLess(idx(red_cloud=False), 0.05)          # 白雲:沒被紅光照亮
        self.assertLess(idx(low_over=True), 0.05)            # 頭頂低雲蓋住

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
            def scene(self, t): return scene()
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
    def scene(self, t):
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
