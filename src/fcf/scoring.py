"""火燒雲 / 朝燒:把「五個環節」變成可計算特徵 → 0-100 指數。(機率校準在 learn.py)
  ① 太陽低角度(天文計算)  ② 光路(球面幾何 + 沿路雲牆)  ③ 空氣透明度(能見度/CAMS 氣膠)
  ④ 畫布(被照亮、仰角合適、雲量適中的中/高雲)  ⑤ 頭頂開口(低雲/降雨)
衛星 nowcast:短領先時,用衛星推得的雲量混入預報雲量,同時保留「純預報」與「含 nowcast」兩種分數供學習。"""
import math

from . import geo
from .astro import sun_pos

US = [0, 25, 50, 100, 150, 200, 300, 400, 500]      # 沿太陽方位的取樣距離 km
BELL = {"high": (0.55, 0.30), "mid": (0.45, 0.25)}  # 各層「最佳雲量」與寬度
CANVAS_H = {"high": (8.0, 10.5), "mid": (3.5, 5.5)}  # 各層雲底/雲體的代表高度 km
DEFAULT_CALIB = {"a": 0.06, "b": 42.0}               # 機率先驗(標註不足時使用)


def bell(x, mu, sg):
    return math.exp(-0.5 * ((x - mu) / sg) ** 2)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def interp(pt, model, var, ts):
    """逐時序列線性內插;來源可有自己的 _time;任一端缺值回傳 None。"""
    d = pt["m"].get(model)
    if not d:
        return None
    arr = d.get(var)
    if not arr:
        return None
    tm = d.get("_time") or pt["time"]
    t0 = tm[0]
    step = (tm[1] - t0) if len(tm) > 1 else 3600
    x = (ts - t0) / step
    i = int(math.floor(x))
    if i < 0 or i + 1 >= len(arr):
        return None
    a, b = arr[i], arr[i + 1]
    if a is None or b is None:
        return None
    return a + (b - a) * (x - i)


def scan_times(kind, ref):
    """日落:落日前 12 分→後 39 分;日出:前 48 分→後 6 分(每 3 分鐘一格)。"""
    ks = range(-4, 14) if kind == "sunset" else range(-16, 3)
    return [ref + k * 180 for k in ks]


def _cover(p, model, lay, ts, others):
    v = interp(p, model, "cloud_cover_" + lay, ts)
    if v is None and others:                          # 該來源在此點沒資料 → 用其他來源共識補(只補點,不補整個來源)
        vs = [x for x in (interp(p, o, "cloud_cover_" + lay, ts) for o in others) if x is not None]
        if vs:
            v = sum(vs) / len(vs)
    return v


def glow_at(ts, site, pts, model, others=(), nc=None, nc_w=0.0):
    """pts[0] 必須是觀測點本身,其後依序對應 US[1:] 的取樣點。nc=[每點 {low,mid,high}(0~1) 或 None]。"""
    if interp(pts[0], model, "cloud_cover_low", ts) is None:
        return None                                   # 該來源在觀測點就沒資料 → 不參與
    el, _ = sun_pos(ts, site["lat"], site["lon"])
    delta = -el
    covers = []
    for i, p in enumerate(pts):
        c = {}
        for lay in geo.H:
            v = _cover(p, model, lay, ts, others)
            if v is None:
                return None
            v = clamp(v / 100.0)
            if nc and nc_w > 0 and nc[i] is not None:
                v = (1 - nc_w) * v + nc_w * nc[i][lay]
            c[lay] = v
        covers.append(c)
    q_best, best = 0.0, {"layer": None, "dist": None, "T": 0.0, "eps": 0.0, "col": 0.0}
    for j, u in enumerate(US):
        for lay, hs in CANVAS_H.items():
            base = bell(covers[j][lay], *BELL[lay])
            items = []
            for h in hs:                              # 同一層取多個高度平均:雲底高度本來就不確定
                eps = geo.elev_angle(h, u)
                T = geo.path_T(h, u, delta, US, covers) if 1.5 <= eps <= 75 else 0.0
                col = geo.redness(geo.graze_alt(h, u, delta)) if T > 0 else 0.0
                items.append((base * T * bell(eps, 20, 16) * col, T, eps, col))
            q = sum(i[0] for i in items) / len(items)
            if q > q_best:
                top = max(items, key=lambda i: i[0])
                q_best, best = q, {"layer": lay, "dist": u, "T": top[1], "eps": top[2], "col": top[3]}
    low_eff = 0.65 * covers[0]["low"] + 0.35 * covers[1]["low"]
    over = (1 - 0.92 * low_eff) ** 1.3
    pr = interp(pts[0], model, "precipitation", ts)
    if pr is not None:
        over *= 1 - clamp(pr / 0.6)
    air = 1.0
    vis = interp(pts[0], model, "visibility", ts)
    if vis is not None:
        if vis < 10000:
            air *= 0.75 + 0.25 * vis / 10000
        if vis < 4000:
            over *= 0.6 + 0.4 * vis / 4000
    rh = interp(pts[0], model, "relative_humidity_2m", ts)
    if rh is not None and rh > 92:
        air *= 0.9
    aod = interp(pts[0], "cams", "aerosol_optical_depth", ts)     # CAMS 氣膠:適量增色、過量變灰
    if aod is not None:
        air *= 0.6 + 0.4 * bell(aod, 0.25, 0.25)
    pm = interp(pts[0], "cams", "pm2_5", ts)
    if pm is not None and pm > 55:
        air *= 0.85
    score = 100 * q_best * over * air
    return score, {"canvas": q_best, "T": best["T"], "over": over, "air": air,
                   "layer": best["layer"], "dist": best["dist"], "eps": best["eps"], "col": best["col"]}


def _notes(kind, comp, spread, n):
    side = "西側" if kind == "sunset" else "東側"
    out = []
    if comp["over"] < 0.35:
        out.append("頭頂低雲或降雨偏多,天邊燒了也容易被蓋住")
    if comp["canvas"] > 0.02 and comp["T"] < 0.3:
        out.append(f"{side}光路被雲牆遮擋,陽光不容易打到雲底")
    if comp["canvas"] < 0.2:
        out.append("缺少合適的中/高雲當畫布(太少或太厚)")
    if comp["air"] < 0.6:
        out.append("空氣透明度偏差(霧霾/高濕),色彩容易偏灰")
    if n >= 2 and spread > 30:
        out.append("各氣象來源分歧大,實際可能偏離分數")
    if n == 1:
        out.append("僅有 1 個來源可用,可信度受限")
    return out


def _curves(kind, ref, site, pts, weights, nc=None, nc_w=0.0):
    times, curves = scan_times(kind, ref), {}
    names = list(weights)
    for m in names:
        others = [o for o in names if o != m]
        row = []
        for t in times:
            r = glow_at(t, site, pts, m, others, nc, nc_w)
            if r is None:
                row = None
                break
            row.append(r)
        if row:
            curves[m] = row
    return times, curves


def _mean(curves, weights, k):
    ws = sum(weights[m] for m in curves)
    return sum(weights[m] * curves[m][k][0] for m in curves) / ws


def score_glow(kind, ref, site, pts, weights, nowcast=None, ens_thr=None):
    """weights: {來源: 權重}。nowcast: {"covers": [...], "w": 混入權重}。回傳含 score(純預報)與 score_nc 的結果。"""
    times, curves = _curves(kind, ref, site, pts, weights)
    if not curves:
        return None
    ws = sum(weights[m] for m in curves)
    mean = [_mean(curves, weights, k) for k in range(len(times))]
    kp = max(range(len(times)), key=lambda k: mean[k])
    peak = mean[kp]
    lo = hi = kp
    while lo > 0 and mean[lo - 1] >= 0.6 * peak:
        lo -= 1
    while hi < len(times) - 1 and mean[hi + 1] >= 0.6 * peak:
        hi += 1
    per_model = {m: round(curves[m][kp][0], 1) for m in curves}
    spread = max(per_model.values()) - min(per_model.values())
    comp = {k: sum(weights[m] * curves[m][kp][1][k] for m in curves) / ws for k in ("canvas", "T", "over", "air", "col")}
    first = next(iter(curves.values()))[kp][1]
    n = len(curves)
    conf = "低" if n == 1 else "高" if spread <= 15 and n >= 3 else "中" if spread <= 30 else "低"
    out = {"kind": kind, "score": round(peak, 1), "confidence": conf, "peak_ts": times[kp],
           "window_ts": [times[lo], times[hi]], "models": per_model, "spread": round(spread, 1),
           "factors": {"canvas": round(comp["canvas"], 3), "light_path": round(comp["T"], 3),
                       "overhead_open": round(comp["over"], 3), "air": round(comp["air"], 3),
                       "color": round(comp["col"], 3), "canvas_layer": first["layer"],
                       "canvas_dist_km": first["dist"]},
           "notes": _notes(kind, comp, spread, n)}
    if nowcast and nowcast.get("w", 0) > 0:
        _, c2 = _curves(kind, ref, site, pts, weights, nowcast["covers"], 1.0)   # 純衛星覆蓋版;混入比例由 learn 決定
        if c2:
            w2 = sum(weights[m] for m in c2)
            out["score_nc"] = round(sum(weights[m] * c2[m][kp][0] for m in c2) / w2, 1)
    return out


def score_members(kind, ref, site, member_pts):
    """集合預報:每個成員各算一次分數。member_pts: {成員名: [pts...]}(成員名即「來源名」)。回傳各成員高峰分數。"""
    out = {}
    for name, pts in member_pts.items():
        times, curves = _curves(kind, ref, site, pts, {name: 1.0})
        if curves:
            out[name] = max(curves[name][k][0] for k in range(len(times)))
    return out
