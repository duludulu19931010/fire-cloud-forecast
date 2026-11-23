"""雲海 / 琉璃光:以「低地參考點」的氣壓層剖面估濕層頂高,對照觀測點海拔。
  雲海 = 濕層頂略低於觀測點(俯瞰)+ 濕層夠厚 + 上方穩定(逆溫蓋)+ 風小 + 沒下雨
  琉璃光 = 夜間雲海 × 下方有燈火 × 風柔 × 夜間時段權重"""
import math

from .astro import moon_illum, sun_pos
from .fetch import LEVELS
from .scoring import clamp, interp

HOUR = 3600


def profile(pt, model, ts):
    ps = interp(pt, model, "surface_pressure", ts)
    z0 = pt.get("elev") or 0.0
    rows = []
    t2, rh2 = interp(pt, model, "temperature_2m", ts), interp(pt, model, "relative_humidity_2m", ts)
    if t2 is not None and rh2 is not None:
        rows.append((z0 + 2.0, rh2, t2))
    for p in LEVELS:
        if ps is not None and p > ps - 3:
            continue
        z = interp(pt, model, f"geopotential_height_{p}hPa", ts)
        rh = interp(pt, model, f"relative_humidity_{p}hPa", ts)
        t = interp(pt, model, f"temperature_{p}hPa", ts)
        if None in (z, rh, t) or z <= z0 + 30:
            continue
        rows.append((z, rh, t))
    rows.sort()
    return rows


def moist_layers(rows, thr=88.0):
    """連續 RH≥thr 的濕層 → [(底, 頂, 頂端索引)],高度以 m 計,層間用線性內插定交界。"""
    layers, cur = [], None
    for i, (z, rh, _) in enumerate(rows):
        if rh >= thr:
            if cur is None:
                if i == 0 or rows[i - 1][1] >= thr:
                    base = z
                else:
                    zp, rp, _ = rows[i - 1]
                    base = zp + (thr - rp) / (rh - rp) * (z - zp)
                cur = [base, z, i]
            cur[1], cur[2] = z, i
        elif cur is not None:
            zp, rp, _ = rows[cur[2]]
            layers.append((cur[0], zp + (rp - thr) / (rp - rh) * (z - zp), cur[2]))
            cur = None
    if cur is not None:
        layers.append((cur[0], rows[cur[2]][0], cur[2]))
    return layers


def trap(x, a, b, c, d):
    if x <= a or x >= d:
        return 0.0
    if x < b:
        return (x - a) / (b - a)
    if x <= c:
        return 1.0
    return (d - x) / (d - c)


def sea_hour(base_pt, site_pt, site, model, ts):
    rows = profile(base_pt, model, ts)
    if len(rows) < 3:
        return None
    E = site["elev"]
    layers = [l for l in moist_layers(rows) if l[0] <= E]
    if not layers:
        return {"score": 0.0, "gap": None, "thick": 0.0, "why": "低地沒有濕層"}
    base, top, idx = max(layers, key=lambda l: l[1] - l[0])
    gap, thick = E - top, top - base
    if idx + 1 < len(rows):
        (z1, _, t1), (z2, _, t2) = rows[idx], rows[idx + 1]
        lapse = -(t2 - t1) / max((z2 - z1) / 1000.0, 0.05)
        stab = clamp((6.5 - lapse) / 6.5)
    else:
        stab = 0.5
    ws = interp(site_pt, model, "wind_speed_10m", ts)
    wind = 1.0 if ws is None else 1.0 - 0.65 * clamp((ws - 5) / 9)
    pr = interp(site_pt, model, "precipitation", ts)
    dry = 1.0 if pr is None else 1 - clamp(pr / 1.0)
    mid, high = interp(site_pt, model, "cloud_cover_mid", ts), interp(site_pt, model, "cloud_cover_high", ts)
    sky = clamp(1 - 0.5 * (mid or 0) / 100 - 0.2 * (high or 0) / 100, 0.2, 1.0)
    s = (trap(gap, -100, 100, 700, 1600) * (0.35 + 0.65 * clamp((thick - 150) / 450))
         * (0.4 + 0.6 * stab) * wind * dry * sky)
    return {"score": 100 * s, "gap": round(gap), "thick": round(thick), "wind": ws,
            "why": "站點在濕層頂之上" if gap > 0 else "站點位於雲中(起霧)"}


def night_weight(local_hour):
    if 18 <= local_hour < 23:
        return 1.0
    if local_hour >= 23 or local_hour < 1:
        return 0.75
    if local_hour < 4:
        return 0.45
    return 0.3


def series(base_pt, site_pt, site, weights, t_from, t_to, tz_h, light):
    """逐時序列 → [{'ts','sea','ryuli','models'}];缺資料的整點略過。"""
    out = []
    ts = int(t_from // HOUR * HOUR)
    while ts <= t_to:
        vals = {}
        for m in weights:
            r = sea_hour(base_pt, site_pt, site, m, ts)
            if r is not None:
                vals[m] = r
        if vals:
            ws = sum(weights[m] for m in vals)
            sea = sum(weights[m] * vals[m]["score"] for m in vals) / ws
            wind = interp(site_pt, next(iter(vals)), "wind_speed_10m", ts)
            gentle = 1.0 if wind is None else 1.0 - 0.6 * clamp((wind - 4) / 6)
            el, _ = sun_pos(ts, site["lat"], site["lon"])
            dark = 1.0 if el < -9 else 0.0
            hr = int(((ts + tz_h * HOUR) % 86400) // HOUR)
            out.append({"ts": ts, "sea": sea, "ryuli": sea * light * dark * night_weight(hr) * gentle,
                        "models": {m: round(vals[m]["score"], 1) for m in vals},
                        "gap": next(iter(vals.values())).get("gap")})
        ts += HOUR
    return out


def pick(ser, key, t0, t1, kind, nowcast=None, lead_h=None):
    """從序列取 [t0, t1] 內最佳時刻 → 事件字典;nowcast 為測站即時估值(只混入 0-6h 內的事件)。"""
    w = [r for r in ser if t0 <= r["ts"] <= t1]
    if not w:
        return None
    best = max(w, key=lambda r: r[key])
    score = best[key]
    if nowcast is not None and lead_h is not None and key == "sea" and lead_h <= 6:
        wt = 0.5 if lead_h <= 3 else 0.25
        score = (1 - wt) * score + wt * nowcast
    lo = hi = w.index(best)
    while lo > 0 and w[lo - 1][key] >= 0.8 * best[key] > 0:
        lo -= 1
    while hi < len(w) - 1 and w[hi + 1][key] >= 0.8 * best[key] > 0:
        hi += 1
    models = best["models"]
    spread = (max(models.values()) - min(models.values())) if models else 0
    n = len(models)
    conf = "低" if n <= 1 else "高" if spread <= 15 and n >= 3 else "中" if spread <= 30 else "低"
    notes = []
    if best.get("gap") is not None and best["gap"] < 0:
        notes.append("模式顯示觀測點可能被雲霧包住(起霧而非俯瞰雲海)")
    if nowcast is not None and key == "sea" and lead_h is not None and lead_h <= 6:
        notes.append(f"已混入測站即時觀測({nowcast:.0f} 分)")
    if kind == "ryuli":
        notes.append(f"月面照亮約 {moon_illum(best['ts']) * 100:.0f}%(月光會增亮雲面,也會稀釋燈火對比)")
    return {"kind": kind, "score": round(score, 1), "confidence": conf, "peak_ts": best["ts"], "window_ts": [w[lo]["ts"], w[hi]["ts"]],
            "models": models, "spread": round(spread, 1), "factors": {"gap_m": best.get("gap")},
            "notes": notes}
