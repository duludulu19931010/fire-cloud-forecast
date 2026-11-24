"""主程式(每小時):取資料(Open-Meteo 預報/空品/集合 + Meteoblue + 測站 + 衛星)→ 評分 → 機率 →
建立剛結束事件的「觀測型真值」→ 輸出 → 自我學習。
用法: python -m fcf.run --config config/sites.json --out data --cache .cache/om.json"""
import argparse
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone

from . import __version__, events as ev, learn, meteoblue, meteogram, obs, sat, scoring, sea, store, truth
from .fetch import V_PATH, V_PLEV, V_SITE, OpenMeteo, log, merge

UTC = timezone.utc
HORIZON_H = 24
BUCKETS = (1, 3, 6, 12, 18, 24)                # 歷史快照:事件前 N 小時各留一筆,供回測
ATTRIBUTION = ("Weather data by Open-Meteo.com (CC BY 4.0); meteoblue (optional); CWA open data (optional); "
               "Himawari-9 via NOAA/AWS (optional); astronomy computed locally.")
load_json = store.load_json
key, path_keys = ev.key, ev.path_keys


def iso(ts, tz):
    return datetime.fromtimestamp(ts, UTC).astimezone(timezone(timedelta(hours=tz))).isoformat(timespec="minutes")


def verdict(p, thr=0.6):
    return "高" if p >= thr else "中" if p >= 0.55 * thr else "低" if p >= 0.15 else "幾乎沒有"


class Ctx:
    def __init__(self, cfg, now, site_d, path_d, plev_d, calib):
        self.cfg, self.now, self.calib = cfg, now, calib
        self.site_d, self.path_d, self.plev_d = site_d, path_d, plev_d
        self.base_w = cfg.get("models") or {"ecmwf_ifs025": 0.4, "icon_seamless": 0.3, "gfs_seamless": 0.3}
        self._ser = {}

    def weights(self, kind, extra=()):
        base = dict(self.base_w)
        if kind in ev.GLOW_SPAN:
            for m in extra:                          # meteoblue 等只參與火燒雲
                base[m] = (self.cfg.get("meteoblue") or {}).get("weight", 0.2)
        return learn.model_weights(self.calib, kind, base)

    def sfc_k(self, site):
        p = self.site_d.get(key(site["lat"], site["lon"]))
        if not p:
            return None
        vs = [v for m in self.base_w if (v := scoring.interp(p, m, "temperature_2m", self.now)) is not None]
        return sum(vs) / len(vs) + 273.15 if vs else None

    def series(self, s):
        if s["id"] not in self._ser:
            sp = self.site_d.get(key(s["lat"], s["lon"]))
            bp = self.plev_d.get(key(*s.get("base", (s["lat"], s["lon"]))))
            w = self.weights("sea_of_clouds")
            self._ser[s["id"]] = (sea.series(bp, sp, s, w, self.now - 2 * 3600, self.now + 26 * 3600,
                                             self.cfg.get("tz_offset", 8), s.get("light", 0.5))
                                  if sp and bp else None)
        return self._ser[s["id"]]


def glow_pts(ctx, e):
    s = e["site"]
    pts = [ctx.site_d.get(key(s["lat"], s["lon"]))] + [ctx.path_d.get(q) for q in path_keys(s, e["az"])]
    return None if any(p is None for p in pts) else pts


def ens_for(om, cfg, e, pts_keys):
    """集合預報:每個成員各算一次分數 → 成員比例(≥ 事件門檻)與 10/90 百分位。失敗回傳 None。"""
    ec = cfg.get("ensemble") or {}
    try:
        data = om.get_ens(pts_keys, model=ec.get("model", "ecmwf_ifs025"))
        names = sorted(next(iter(data.values()))["m"])
        mp = {n: [{"time": data[k]["time"], "m": {n: data[k]["m"][n]}} for k in pts_keys] for n in names if all(n in data[k]["m"] for k in pts_keys)}
        sc = sorted(scoring.score_members(e["kind"], e["ref"], e["site"], mp).values())
    except Exception as ex:
        log(f"[ens] {e['site']['id']} failed: {ex}")
        return None
    if len(sc) < 8:
        return None
    thr = ec.get("thr", 40.0)
    q = lambda p: sc[min(len(sc) - 1, int(p * (len(sc) - 1) + 0.5))]
    return {"frac": sum(1 for v in sc if v >= thr) / len(sc), "p10": q(0.1), "p90": q(0.9), "n": len(sc)}


def run(cfg, now, om, env=None, calib=None, hs=None, out=None):
    env, calib, tz = env or {}, calib or {}, cfg.get("tz_offset", 8)
    planned = ev.plan(cfg, now, HORIZON_H)
    site_pts = {key(s["lat"], s["lon"]) for s in cfg["sites"]}
    base_pts = {key(*s.get("base", (s["lat"], s["lon"]))) for s in cfg["sites"] if {"sea_of_clouds", "ryuli"} & set(s["types"])}
    p_pts = {q for e in planned if e["kind"] in ev.GLOW_SPAN for q in path_keys(e["site"], e["az"])}
    site_d, path_d = om.get(list(site_pts), V_SITE, "site"), om.get(list(p_pts), V_PATH, "path")
    plev_d = om.get(list(base_pts), V_PLEV, "plev")
    for k, a in om.get_air(list(site_pts)).items():            # CAMS 氣膠(失敗不影響主流程)
        merge(site_d.get(k), a)
    extra = []
    mbk = env.get("METEOBLUE_API_KEY")
    if mbk and (cfg.get("meteoblue") or {}).get("enabled", True):
        mb = meteoblue.Cached(om.cache, mbk, clock=lambda: now)
        want = [(s["lat"], s["lon"], s.get("elev")) for s in cfg["sites"]]
        if (cfg.get("meteoblue") or {}).get("path", True):
            want += [(la, lo, None) for la, lo in p_pts]
        got = mb.get(want)
        for k, d in got.items():
            tgt = site_d.get(key(*k)) or path_d.get(key(*k))
            if tgt:
                merge(tgt, d)
        extra = ["meteoblue"] if got else []
    om.flush()
    stations = obs.fetch_obs(env["CWA_API_KEY"]) if env.get("CWA_API_KEY") else {}
    ctx = Ctx(cfg, now, site_d, path_d, plev_d, calib)

    scene = None                                                  # 衛星 nowcast:只在 3.5h 內有日出/日落事件時才抓
    sc_cfg = cfg.get("sat") or {}
    if hs is not None and sc_cfg.get("enabled", True) and any(
            e["kind"] in ev.GLOW_SPAN and e["ref"] - now <= 3.5 * 3600 for e in planned):
        scene = hs.scene(now - 900, want_visible=False)

    results, skipped = {s["id"]: [] for s in cfg["sites"]}, 0
    glow_done = []
    for e in planned:
        s, k = e["site"], e["kind"]
        lead = (max(e["t0"], now) - now) / 3600
        if k in ev.GLOW_SPAN:
            pts = glow_pts(ctx, e)
            if pts is None:
                skipped += 1
                continue
            nc = None
            sk = ctx.sfc_k(s)
            if scene and sk and e["ref"] - now <= 3.5 * 3600:
                cov = sat.nowcast_covers(scene, s, e["az"], sk)
                nc = {"covers": cov, "w": 1.0} if cov else None
            r = scoring.score_glow(k, e["ref"], s, pts, ctx.weights(k, extra if pts[0]["m"].get("meteoblue") else ()), nc)
            if r:
                r["sun_ts"], r["az"] = e["ref"], round(e["az"], 1)
                glow_done.append((e, r, pts))
        else:
            sp, bp = site_d.get(key(s["lat"], s["lon"])), plev_d.get(key(*s.get("base", (s["lat"], s["lon"]))))
            ser = ctx.series(s)
            if not (sp and bp and ser):
                skipped += 1
                continue
            t0, t1 = max(e["t0"], now), min(e["t1"], now + HORIZON_H * 3600)
            nc = obs.nowcast_sea(s, stations) if stations and k == "sea_of_clouds" else None
            r = sea.pick(ser, "sea" if k == "sea_of_clouds" else "ryuli", t0, t1, k, nowcast=nc, lead_h=lead)
            if r:
                r["sun_ts"] = e["t0"]
                learn.finalize(r, k, (r["peak_ts"] - now) / 3600, calib, s["id"])
                r.update({"date": e["date"], "site": s["id"]})
                results[s["id"]].append(r)
            else:
                skipped += 1

    # 集合預報:只對「分數夠高」的前 N 個火燒雲事件做(API 額度考量),結果並入機率特徵
    ec = cfg.get("ensemble") or {}
    top = sorted(glow_done, key=lambda x: -x[1]["score"])[: ec.get("max_events", 4)] if ec.get("enabled", True) else []
    ens_by = {}
    for e, r, pts in top:
        if r["score"] >= ec.get("min_score", 15.0):
            ens_by[id(e)] = ens_for(om, cfg, e, [key(e["site"]["lat"], e["site"]["lon"])] + path_keys(e["site"], e["az"]))
    for e, r, pts in glow_done:
        lead = (r["peak_ts"] - now) / 3600
        learn.finalize(r, e["kind"], lead, calib, e["site"]["id"], ens_by.get(id(e)))
        r.update({"date": e["date"], "site": e["site"]["id"]})
        results[e["site"]["id"]].append(r)

    sites_out, events_all, names = [], [], {s["id"]: s["name"] for s in cfg["sites"]}
    charts = {}
    for s in cfg["sites"]:
        evs = sorted(results[s["id"]], key=lambda r: r["peak_ts"])
        for r in evs:
            r["lead_h"] = round((r["peak_ts"] - now) / 3600, 1)
            r["peak"] = iso(r["peak_ts"], tz)
            r["window"] = [iso(t, tz) for t in r["window_ts"]]
        events_all += evs
        rec = {"id": s["id"], "name": s["name"], "lat": s["lat"], "lon": s["lon"], "elev": s.get("elev"), "events": evs}
        ser = ctx.series(s) if {"sea_of_clouds", "ryuli"} & set(s["types"]) else None
        tl = [x for x in (ser or []) if now <= x["ts"] <= now + HORIZON_H * 3600]
        if tl:
            rec["timeline"] = [{"t": iso(x["ts"], tz), "sea": round(x["sea"]), "ryuli": round(x["ryuli"])} for x in tl]
        sp = site_d.get(key(s["lat"], s["lon"]))
        if sp:
            charts[s["id"]] = meteogram.render(s, meteogram.cloud_layers(sp, ctx.base_w, now), evs, tl, now, tz)
        sites_out.append(rec)

    summary, ranking = {}, {}
    for kind in ev.KINDS:
        es = sorted([r for r in events_all if r["kind"] == kind], key=lambda r: -r["prob"])
        if not es:
            continue
        thr, b = learn.thr_hi(calib, kind), es[0]
        ranking[kind] = [{"site": r["site"], "name": names[r["site"]], "score": r["score"], "prob": r["prob"],
                          "peak": r["peak"], "confidence": r["confidence"]} for r in es[:10]]
        summary[kind] = {"label": learn.KIND_ZH[kind], "chance": verdict(b["prob"], thr), "max_prob": b["prob"],
                         "sites_prob_ge_50": sum(1 for r in es if r["prob"] >= 0.5),
                         "best": {"site": b["site"], "name": names[b["site"]], "peak": b["peak"], "window": b["window"],
                                  "score": b["score"], "prob": b["prob"], "confidence": b["confidence"], "notes": b["notes"]}}
    meta = {"generated_at": iso(now, tz), "generated_ts": int(now), "version": __version__, "horizon_h": HORIZON_H,
            "sources": {"open_meteo": True, "meteoblue": bool(extra), "cwa": bool(stations), "himawari_nowcast": bool(scene),
                        "ensemble_events": sum(1 for v in ens_by.values() if v)},
            "events": len(events_all), "skipped": skipped, "attribution": ATTRIBUTION,
            "champions": {k: (calib.get(k) or {}).get("champion", "prior") for k in ev.KINDS}}
    nxt24 = {"generated_at": iso(now, tz), "horizon_h": HORIZON_H, "summary": summary,
             "events": [dict(r, name=names[r["site"]]) for r in sorted(events_all, key=lambda r: r["peak_ts"])]}
    return ({"meta": meta, "sites": sites_out, "ranking": ranking, "next24h": nxt24}, events_all, charts, ctx, stations)


def write_outputs(result, events_all, charts, now, outdir):
    for sub in ("sites", "history", "truth", "obs", "model", "charts"):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)
    for name, obj in (("latest", result), ("next24h", result["next24h"]), ("ranking", result["ranking"]), ("meta", result["meta"])):
        store.dump_json(os.path.join(outdir, name + ".json"), obj)
    for s in result["sites"]:
        store.dump_json(os.path.join(outdir, "sites", s["id"] + ".json"), s)
    for sid, svg in charts.items():
        with open(os.path.join(outdir, "charts", sid + ".svg"), "w", encoding="utf-8") as f:
            f.write(svg)
    month = datetime.fromtimestamp(now, UTC).strftime("%Y-%m")
    seen = {(r.get("gen_h"), r.get("site"), r.get("kind"), r.get("date")) for r in store.read_ndjson(os.path.join(outdir, "history"))}
    gh, rows = int(now // 3600), []
    for r in events_all:
        if int(math.floor((r["peak_ts"] - now) / 3600 + 0.5)) in BUCKETS and (gh, r["site"], r["kind"], r["date"]) not in seen:
            rows.append({"gen_h": gh, "site": r["site"], "kind": r["kind"], "date": r["date"], "lead_h": r["lead_h"],
                         "score": r.get("score_fc", r["score"]), "score_nc": r.get("score_nc"), "prob": r["prob"],
                         "models": r["models"], "factors": r.get("factors"), "ens": (r.get("ens") or {}).get("frac"),
                         "peak_ts": r["peak_ts"]})
    store.append_ndjson(os.path.join(outdir, "history", month + ".ndjson"), rows)
    return len(rows)


def pending_truth_keys(out, now, tz):
    cutoff = (ev.local_today(now, tz) - timedelta(days=3)).isoformat()
    have = {(r["site"], r["kind"], r["date"]) for r in store.read_ndjson(os.path.join(out, "truth"))}
    return {(h["site"], h["kind"], h["date"]) for h in store.read_ndjson(os.path.join(out, "history"))
            if h["date"] >= cutoff} - have


def main(argv=None, hs=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/sites.json")
    ap.add_argument("--out", default="data")
    ap.add_argument("--cache", default=".cache/om.json")
    ap.add_argument("--now", help="ISO8601 (UTC),測試用")
    ap.add_argument("--mock", action="store_true", help="用合成資料乾跑,不連網")
    ap.add_argument("--no-learn", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_json(a.config)
    if not cfg:
        log("config not found:", a.config)
        return 2
    tz = cfg.get("tz_offset", 8)
    now = datetime.fromisoformat(a.now.replace("Z", "+00:00")).timestamp() if a.now else time.time()
    calib = load_json(os.path.join(a.out, "model", "calibration.json"), {})
    env = {k: os.environ.get(k) for k in ("CWA_API_KEY", "METEOBLUE_API_KEY", "OPEN_METEO_API_KEY") if os.environ.get(k)}
    if a.mock:
        from .mock import MockOpenMeteo
        om, env = MockOpenMeteo(now), {}
    else:
        om = OpenMeteo(a.cache, api_key=env.get("OPEN_METEO_API_KEY"), max_age=cfg.get("forecast_max_age_h", 3) * 3600,
                       clock=lambda: now)
        if hs is None and (cfg.get("sat") or {}).get("enabled", True):
            hs = sat.from_config(cfg, os.path.join(os.path.dirname(a.cache) or ".cache", "sat"))
    result, events_all, charts, ctx, stations = run(cfg, now, om, env, calib, hs, a.out)
    if not events_all:
        log("no events produced (all fetches failed?) — keeping previous outputs")
        return 1
    nh = write_outputs(result, events_all, charts, now, a.out)
    truth.log_obs(a.out, now, cfg, stations)
    tr = truth.build(cfg, now, a.out, pending_truth_keys(a.out, now, tz), hs, ctx.sfc_k, calib.get("detector"))
    if not a.no_learn:
        learn.update(a.out, now, cfg.get("models"))
    log(f"ok: {len(events_all)} events, {result['meta']['skipped']} skipped, +{nh} history, +{len(tr)} truth rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
