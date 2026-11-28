"""觀測型真值:「事件到底有沒有真的發生」,不靠模式分析場。
  雲海 / 琉璃光 → 氣象署測站(谷地近飽和 + 山上乾 + 穩定層)每小時取樣,事件結束後依窗口內取樣判定
  火燒雲 / 朝燒 → Himawari 可見光:太陽方向扇形內「被紅光照亮的雲」覆蓋 × 頭頂開口
  人工回報(labels.csv)優先於上述兩者。
每筆真值保留原始證據(evidence),偵測器閾值之後被學習修正時可重新判定,不必重抓衛星。"""
import os

from . import obs, sat, store
from .events import GLOW_SPAN, plan_ended

SEA_T = 60.0            # 測站雲海分數 ≥ 此值視為「雲海成形」(先驗,可在 config.truth 覆蓋)
CALM_WS = 5.0           # 琉璃光要求山頂風速 ≤ 此值 (m/s)


def log_obs(out, now, cfg, stations):
    """每小時記一筆各山區站點的測站取樣(測站資料只有「現在」,所以必須自己累積)。"""
    from datetime import datetime, timezone
    rows = []
    for s in cfg["sites"]:
        if {"sea_of_clouds", "ryuli"} & set(s["types"]) and stations:
            smp = obs.sea_sample(s, stations)
            if smp:
                rows.append({"ts": int(now), "site": s["id"], **smp})
    month = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m")
    store.append_ndjson(os.path.join(out, "obs", month + ".ndjson"), rows)
    return len(rows)


def sea_truth(samples, kind, t0, t1, sea_t=SEA_T):
    w0 = [x for x in samples if t0 <= x["ts"] <= t1]
    if not w0:
        return None
    w = [x for x in w0 if x.get("ws") is None or x["ws"] <= CALM_WS] if kind == "ryuli" else w0
    hits = sum(1 for x in w if x["score"] >= sea_t)
    need = 2 if len(w) >= 3 else 1
    return {"src": "cwa", "y": 1 if w and hits >= need else 0,
            "ev": {"n": len(w0), "n_ok": len(w), "hits": hits, "max": max(x["score"] for x in w0)}}


def sat_truth(hs, event, sfc_k, detector):
    """event 窗口內每 10 分鐘一景,取「紅光雲覆蓋」最大的一景作證據。"""
    if hs is None:
        return None
    site, az = event["site"], event["az"]
    best, n = None, 0
    for t in range(int(event["t0"] // 600 * 600), int(event["t1"]) + 1, 600):
        sc = hs.scene(t)
        if sc is None:
            continue
        f = sat.glow_features(sc, site, az, sfc_k)
        if f is None:
            continue
        n += 1
        if best is None or f["red_frac"]["1.4"] > best["red_frac"]["1.4"]:
            best = f
    if best is None:
        return None
    best["slots"] = n
    return {"src": "sat", "y": judge_sat(best, detector), "ev": best}


def judge_sat(feat, detector):
    d = dict(sat.DETECTOR_PRIOR, **(detector or {}))
    return 1 if (sat.glow_index(feat, d["ratio_thr"], lit_min=d.get("lit_min")) or 0) >= d["tau"] else 0


def row_y(row, detector=None):
    """真值列 → 0/1。衛星列依「現行偵測器參數」重新判定。"""
    if row["src"] == "sat":
        return judge_sat(row["ev"], detector)
    return int(row["y"])


def build(cfg, now, out, pending_keys, hs=None, sfc_of=None, detector=None):
    """對已結束的事件建立真值列。sfc_of(site)→地面溫度 K(衛星雲頂高需要)。"""
    from datetime import datetime, timezone
    events = plan_ended(cfg, pending_keys, now)
    samples = {}
    for r in store.read_ndjson(os.path.join(out, "obs")):
        samples.setdefault(r["site"], []).append(r)
    rows = []
    for e in events:
        s, k = e["site"], e["kind"]
        if k in GLOW_SPAN:
            t = sat_truth(hs, e, (sfc_of(s) if sfc_of else None) or 299.0, detector)
        else:
            t = sea_truth(samples.get(s["id"], []), k, e["t0"], e["t1"], (cfg.get("truth") or {}).get("sea_t", SEA_T))
        if t:
            rows.append({"site": s["id"], "kind": k, "date": e["date"], "ts": int(now), **t})
    month = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m")
    store.append_ndjson(os.path.join(out, "truth", month + ".ndjson"), rows)
    return rows
