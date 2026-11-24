"""冷啟動回補:用 Open-Meteo Previous Runs API 還原「N 天前發布的預報」(≈ 領先 24h),
配上衛星觀測型真值(Himawari 有多年封存),一次補出歷史快照 + 真值,讓自我學習不必從零等待。
限制:① 只回補火燒雲/朝燒(雲海與琉璃光的測站觀測沒有歷史來源);② 沒有集合預報與 nowcast 欄位;③ 衛星 IO 需先用 scripts/sat_probe.py 驗證。
用法: python -m fcf.backfill --days 30 --out data [--no-truth]"""
import argparse
import os
import sys
import time
from datetime import timedelta

from . import events as ev, learn, scoring, store, truth
from .fetch import V_PATH, OpenMeteo, log
from .sat import from_config

V_BACK = V_PATH + ["precipitation", "relative_humidity_2m", "temperature_2m", "visibility"]


def run(cfg, now, om, out, days=30, hs=None, calib=None):
    tz, weights = cfg.get("tz_offset", 8), cfg.get("models")
    today = ev.local_today(now, tz)
    jobs = []
    for s in cfg["sites"]:
        for kind in s["types"]:
            if kind not in ev.GLOW_SPAN:
                continue
            for d in range(1, days + 1):
                e = ev._event(s, kind, today - timedelta(days=d), tz)
                if e:
                    jobs.append(e)
    pts = {ev.key(e["site"]["lat"], e["site"]["lon"]) for e in jobs} | {q for e in jobs for q in ev.path_keys(e["site"], e["az"])}
    data = om.get_prev(list(pts), V_BACK, day=1, past_days=days + 1)
    have_h = {(h["site"], h["kind"], h["date"]) for h in store.read_ndjson(os.path.join(out, "history"))}
    have_t = {(t["site"], t["kind"], t["date"]) for t in store.read_ndjson(os.path.join(out, "truth"))}
    hist, trs = [], []
    for e in jobs:
        s, k = e["site"], e["kind"]
        kk = (s["id"], k, e["date"])
        p0 = data.get(ev.key(s["lat"], s["lon"]))
        pp = [data.get(q) for q in ev.path_keys(s, e["az"])]
        if p0 is None or any(p is None for p in pp):
            continue
        if kk not in have_h:
            r = scoring.score_glow(k, e["ref"], s, [p0] + pp, learn.model_weights(calib or {}, k, weights))
            if r:
                hist.append({"gen_h": int(e["ref"] // 3600) - 24, "site": s["id"], "kind": k, "date": e["date"], "lead_h": 24.0,
                             "score": r["score"], "score_nc": None, "prob": None, "models": r["models"],
                             "factors": r["factors"], "ens": None, "peak_ts": r["peak_ts"], "backfill": True})
        if hs is not None and kk not in have_t and now - e["t1"] > 3600:
            vs = [v for m in (weights or {}) if (v := scoring.interp(p0, m, "temperature_2m", e["ref"])) is not None]
            t = truth.sat_truth(hs, e, (sum(vs) / len(vs) + 273.15) if vs else 299.0, (calib or {}).get("detector"))
            if t:
                trs.append({"site": s["id"], "kind": k, "date": e["date"], "ts": int(now), **t})
    store.append_ndjson(os.path.join(out, "history", "backfill.ndjson"), hist)
    store.append_ndjson(os.path.join(out, "truth", "backfill.ndjson"), trs)
    return len(hist), len(trs)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/sites.json")
    ap.add_argument("--out", default="data")
    ap.add_argument("--cache", default=".cache/om.json")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--no-truth", action="store_true")
    a = ap.parse_args(argv)
    cfg = store.load_json(a.config)
    now = time.time()
    om = OpenMeteo(a.cache, api_key=os.environ.get("OPEN_METEO_API_KEY"), clock=lambda: now)
    hs = None if a.no_truth else from_config(cfg, os.path.join(os.path.dirname(a.cache) or ".cache", "sat"))
    calib = store.load_json(os.path.join(a.out, "model", "calibration.json"), {})
    nh, nt = run(cfg, now, om, a.out, a.days, hs, calib)
    om.flush()
    learn.update(a.out, now, cfg.get("models"))
    log(f"backfill: +{nh} history, +{nt} truth")
    return 0


if __name__ == "__main__":
    sys.exit(main())
