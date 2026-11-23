"""自我學習 + 每小時回測(以「事件有沒有真的發生」為準,不再拿預報跟分析場比)。

真值(truth):人工回報 labels.csv(最高權重)> 觀測型偵測(氣象署測站 / Himawari 衛星,權重由「與人工回報的一致度」自動學出)。
自我校正機制(全部每小時自動跑,且都用時間序列 walk-forward 守門,新參數不會比舊參數差才採用):
  1. 冠軍/挑戰者:prior、logit(s)、+領先時間、+站點偏差(階層收縮)、+各環節因子指數、+集合預報成員比例、isotonic
     → expanding-window walk-forward 的 Brier 最低者當冠軍;需勝過現任 2% 才換;勝不過 prior 就回退(rollback)
  2. 時間衰減:近期樣本權重大(半衰期 150 天),跟上季節/模式更新造成的漂移
  3. 來源權重學習:各氣象來源(ECMWF/ICON/GFS/meteoblue…)依自身 Brier 做指數加權(Hedge),向先驗收縮
  4. nowcast 混入比例:在標註上比較「純預報 vs 衛星修正」→ 學出最佳混入權重
  5. 偵測器自我校準:衛星紅度閾值/判定門檻用人工回報做網格搜尋(證據已存,不必重抓衛星);測站規則同理計算一致度
  6. 告警門檻學習:「高」機率門檻取使 CSI 最大者
  7. 漂移監控:近 25 場平均機率 vs 實際發生率,偏離過大在報告標示"""
import csv
import math
import os
from datetime import datetime, timezone

from . import store, truth
from .events import KINDS
from .sat import DETECTOR_PRIOR, THRS
from .scoring import DEFAULT_CALIB, clamp

KIND_ZH = {"sunset": "日落火燒雲", "sunrise": "日出朝燒", "sea_of_clouds": "雲海", "ryuli": "琉璃光"}
MIN_EVENTS, HALF_LIFE, MARGIN = 20, 150.0, 0.02
LEADS = (1, 3, 6, 12, 18, 24)
TRUTH_W0 = {"human": 1.0, "sat": 0.5, "cwa": 0.6}
GLOW = ("sunset", "sunrise")
FACTORS = ("canvas", "overhead_open", "air", "light_path", "color")


def sigmoid(x):
    return 1 / (1 + math.exp(-max(-40.0, min(40.0, x))))


# ───────────────────────── 線性代數 / 羅吉斯回歸(Newton + 高斯先驗) ─────────────────────────
def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        d = M[c][c] or 1e-12
        for r in range(c + 1, n):
            f = M[r][c] / d
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][k] * x[k] for k in range(i + 1, n))) / (M[i][i] or 1e-12)
    return x


def fit_logit(X, y, w, mu, prec, iters=30):
    d = len(mu)
    beta = list(mu)
    for _ in range(iters):
        g = [-prec[i] * (beta[i] - mu[i]) for i in range(d)]
        H = [[prec[i] if i == j else 0.0 for j in range(d)] for i in range(d)]
        for xi, yi, wi in zip(X, y, w):
            p = sigmoid(sum(b * v for b, v in zip(beta, xi)))
            r, s = wi * (yi - p), wi * p * (1 - p)
            for i in range(d):
                g[i] += r * xi[i]
                for j in range(d):
                    H[i][j] += s * xi[i] * xi[j]
        step = _solve(H, g)
        step = [max(-3.0, min(3.0, v)) for v in step]
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-6:
            break
    return beta


def pav(xs, ys, ws):
    """加權保序回歸(Pool Adjacent Violators)→ [(x_center, y)] 區塊。"""
    blocks = []
    for x, y, w in sorted(zip(xs, ys, ws)):
        blocks.append([x * w, y * w, w, x, x])
        while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
            b = blocks.pop()
            blocks[-1] = [blocks[-1][0] + b[0], blocks[-1][1] + b[1], blocks[-1][2] + b[2], blocks[-1][3], b[4]]
    return [(b[0] / b[2], b[1] / b[2]) for b in blocks]


def brier(ps, ys, ws):
    w = sum(ws)
    return sum(wi * (p - y) ** 2 for p, y, wi in zip(ps, ys, ws)) / w if w else float("nan")


def auc(ps, ys):
    pos = [p for p, y in zip(ps, ys) if y]
    neg = [p for p, y in zip(ps, ys) if not y]
    if not pos or not neg:
        return None
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(s / (len(pos) * len(neg)), 3)


# ───────────────────────── 模型(特徵 → 機率) ─────────────────────────
A0, B0 = DEFAULT_CALIB["a"] * 100, DEFAULT_CALIB["b"] / 100          # s=S/100 的先驗斜率 / 中點
PRIOR_MODEL = {"kind": "logit", "names": ["s"], "beta": [-A0 * B0, A0]}


def feat(row, names):
    x = [1.0]
    for n in names:
        if n == "s":
            x.append(row["s"] / 100.0)
        elif n == "lead":
            x.append(row["lead"] / 24.0)
        elif n == "ens":
            x.append((row.get("ens") if row.get("ens") is not None else 0.5) - 0.5)
        elif n.startswith("f_"):
            x.append(math.log(max((row.get("factors") or {}).get(n[2:], 1.0), 0.02)))
        elif n.startswith("site:"):
            x.append(1.0 if row["site"] == n[5:] else 0.0)
    return x


def predict(model, row):
    if model["kind"] == "iso":
        pts = model["pts"]
        s = row["s"]
        if s <= pts[0][0]:
            p = pts[0][1]
        elif s >= pts[-1][0]:
            p = pts[-1][1]
        else:
            p = next(pts[i][1] + (pts[i + 1][1] - pts[i][1]) * (s - pts[i][0]) / max(pts[i + 1][0] - pts[i][0], 1e-9)
                     for i in range(len(pts) - 1) if pts[i][0] <= s <= pts[i + 1][0])
        return clamp(p, 0.02, 0.98)
    return sigmoid(sum(b * v for b, v in zip(model["beta"], feat(row, model["names"]))))


def _prior_vec(names):
    mu, prec = [-A0 * B0], [0.01]
    for n in names:
        if n == "s":
            mu.append(A0); prec.append(1 / 9.0)
        elif n.startswith("site:"):
            mu.append(0.0); prec.append(4.0)           # 站點偏差:強收縮(階層式)
        else:
            mu.append(0.0); prec.append(1.0)
    return mu, prec


def fit_model(spec, rows):
    """spec: ('prior',) | ('logit', names) | ('iso',) → model 字典。"""
    if spec[0] == "prior":
        return dict(PRIOR_MODEL)
    if spec[0] == "iso":
        return {"kind": "iso", "pts": [[round(a, 2), round(b, 4)] for a, b in
                                         pav([r["s"] for r in rows], [r["y"] for r in rows], [r["w"] for r in rows])]}
    names = list(spec[1])
    mu, prec = _prior_vec(names)
    beta = fit_logit([feat(r, names) for r in rows], [r["y"] for r in rows], [r["w"] for r in rows], mu, prec)
    return {"kind": "logit", "names": names, "beta": [round(b, 4) for b in beta]}


def candidates(kind, rows):
    sites = sorted({r["site"] for r in rows})
    c = {"prior": ("prior",), "logit_s": ("logit", ["s"]), "logit_sl": ("logit", ["s", "lead"]),
         "logit_sls": ("logit", ["s", "lead"] + [f"site:{s}" for s in sites]), "iso": ("iso",)}
    if kind in GLOW and sum(1 for r in rows if r.get("factors")) >= 0.8 * len(rows):
        c["logit_fac"] = ("logit", ["s", "lead"] + [f"f_{f}" for f in FACTORS])
    if sum(1 for r in rows if r.get("ens") is not None) >= 0.6 * len(rows):
        c["logit_ens"] = ("logit", ["s", "lead", "ens"])
    return c


# ───────────────────────── 資料整理 ─────────────────────────
def read_labels(path):
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                try:
                    out[(r["site"].strip(), r["kind"].strip(), r["date"].strip())] = int(r["label"])
                except (KeyError, ValueError):
                    pass
    return out


def outcomes(truth_rows, labels, calib):
    """每個事件的最終結果 → {key: (y, w, src)}。人工回報優先;觀測型真值依學到的可信度加權。"""
    det, tw = calib.get("detector"), dict(TRUTH_W0, **(calib.get("truth_w") or {}))
    out = {}
    for r in truth_rows:
        out[(r["site"], r["kind"], r["date"])] = (truth.row_y(r, det), tw.get(r["src"], 0.5), r["src"])
    for k, lab in labels.items():
        out[k] = (1 if lab >= 2 else 0, 1.0, "human")
    return out


def build_samples(hist, oc, now):
    rows, per = [], {}
    for h in hist:
        k = (h["site"], h["kind"], h["date"])
        if k in oc:
            per.setdefault(k, []).append(h)
    for k, hs in per.items():
        y, w, src = oc[k]
        age = max(0.0, (now - datetime.fromisoformat(k[2]).replace(tzinfo=timezone.utc).timestamp()) / 86400)
        decay = 0.5 ** (age / HALF_LIFE)
        for h in hs:
            rows.append({"k": k, "date": k[2], "site": k[0], "s": h["score"], "s_nc": h.get("score_nc"),
                         "lead": h["lead_h"], "factors": h.get("factors"), "ens": h.get("ens"),
                         "models": h.get("models") or {}, "y": y, "w": w * decay / len(hs), "w_test": 1.0 / len(hs),
                         "src": src})
    return rows


# ───────────────────────── 每個機制 ─────────────────────────
def walk_forward(kind, rows, old_champ):
    ev = sorted({r["k"] for r in rows}, key=lambda k: (k[2], k))
    n = len(ev)
    cands = candidates(kind, rows)
    if n < MIN_EVENTS:
        return None
    start = max(10, int(n * 0.5))
    edges = sorted({start + round(i * (n - start) / 4) for i in range(5)})
    loss = {c: [0.0, 0.0] for c in cands}
    oos = {c: [] for c in cands}
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        tr_k, te_k = set(ev[:a]), set(ev[a:b])
        tr, te = [r for r in rows if r["k"] in tr_k], [r for r in rows if r["k"] in te_k]
        for c, spec in cands.items():
            m = fit_model(spec, tr)
            for r in te:
                p = predict(m, r)
                loss[c][0] += r["w_test"] * (p - r["y"]) ** 2
                loss[c][1] += r["w_test"]
                oos[c].append((p, r["y"], r["w_test"], r["lead"], r["k"]))
    wf = {c: (v[0] / v[1] if v[1] else float("nan")) for c, v in loss.items()}
    return cands, wf, oos


def choose(wf, old_name):
    order = list(wf)                                              # 依複雜度排序(先簡單)
    base = old_name if old_name in wf else "prior"
    order = [c for c in order if wf[c] == wf[c]]                  # 排除 NaN
    best = min(order, key=lambda c: wf[c])
    name, why = base, "沿用現任"
    if best != base and wf[best] < wf[base] * (1 - MARGIN):
        name, why = best, f"換成 {best}(walk-forward Brier {wf[best]:.4f} < {wf[base]:.4f})"
    if name != "prior" and wf[name] > wf["prior"] * (1 + MARGIN):
        name, why = "prior", f"回退 prior({name} 比 prior 差)"       # rollback
    return name, why


def learn_weights(kind, rows, base_w):
    """各來源以自身 Brier 做指數加權(Hedge),向先驗收縮。"""
    loss, cnt = {}, {}
    for r in rows:
        for m, s in r["models"].items():
            p = sigmoid(A0 * (s / 100.0 - B0))
            loss[m] = loss.get(m, 0.0) + r["w_test"] * (p - r["y"]) ** 2
            cnt[m] = cnt.get(m, 0.0) + r["w_test"]
    ms = [m for m in cnt if cnt[m] >= MIN_EVENTS and m in base_w]
    if len(ms) < 2:
        return None
    b = {m: loss[m] / cnt[m] for m in ms}
    bmin = min(b.values())
    raw = {m: base_w[m] * math.exp(-15.0 * (b[m] - bmin)) for m in ms}
    tot, tb = sum(raw.values()), sum(base_w[m] for m in ms)
    out = {}
    for m in ms:
        lam = min(cnt[m] / (cnt[m] + 40.0), 0.8)
        out[m] = round((1 - lam) * base_w[m] + lam * raw[m] / tot * tb, 4)
    return out


def learn_nowcast(rows, model):
    sel = [r for r in rows if r["s_nc"] is not None and r["lead"] <= 3.5]
    if len({r["k"] for r in sel}) < 15:
        return None, None
    res = {}
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        ps = [predict(model, dict(r, s=(1 - w) * r["s"] + w * r["s_nc"])) for r in sel]
        res[w] = brier(ps, [r["y"] for r in sel], [r["w_test"] for r in sel])
    best = min(res, key=res.get)
    return best, {str(k): round(v, 4) for k, v in res.items()}


def learn_detector(truth_rows, labels, calib):
    """衛星偵測器:對有人工回報的事件網格搜尋 (紅度閾值, 判定門檻),以 balanced accuracy 為目標。"""
    pairs = [(r["ev"], 1 if labels[(r["site"], r["kind"], r["date"])] >= 2 else 0) for r in truth_rows
             if r["src"] == "sat" and (r["site"], r["kind"], r["date"]) in labels]
    pos, neg = sum(y for _, y in pairs), sum(1 - y for _, y in pairs)
    if len(pairs) < 15 or pos < 4 or neg < 4:
        return None, None
    from .sat import glow_index

    def ba(thr, tau):
        tp = sum(1 for f, y in pairs if y and (glow_index(f, thr) or 0) >= tau)
        tn = sum(1 for f, y in pairs if not y and (glow_index(f, thr) or 0) < tau)
        return (tp / pos + tn / neg) / 2
    cur = dict(DETECTOR_PRIOR, **(calib.get("detector") or {}))
    cur_ba = ba(cur["ratio_thr"], cur["tau"])
    best, best_ba = cur, cur_ba
    for t in THRS:
        for tau in [i / 20 for i in range(1, 20)]:
            b = ba(t, tau)
            if b > best_ba + 0.03:
                best, best_ba = {"ratio_thr": t, "tau": tau}, b
    return best, round(best_ba, 3)


def agreement(truth_rows, labels, src, detector=None):
    pr = [(truth.row_y(r, detector), 1 if labels[(r["site"], r["kind"], r["date"])] >= 2 else 0)
          for r in truth_rows if r["src"] == src and (r["site"], r["kind"], r["date"]) in labels]
    pos, neg = sum(1 for _, h in pr if h), sum(1 for _, h in pr if not h)
    if len(pr) < 10 or pos < 3 or neg < 3:
        return None
    ba = (sum(1 for y, h in pr if h and y) / pos + sum(1 for y, h in pr if not h and not y) / neg) / 2
    return round(ba, 3)


def learn_threshold(oos_champ):
    pts = [(p, y, w) for p, y, w, lead, _ in oos_champ if lead <= 12]
    if len(pts) < 20 or sum(1 for _, y, _ in pts if y) < 5:
        return None
    best, bc = None, -1
    for t in [0.3 + 0.05 * i for i in range(11)]:
        h = sum(w for p, y, w in pts if p >= t and y)
        m = sum(w for p, y, w in pts if p < t and y)
        f = sum(w for p, y, w in pts if p >= t and not y)
        csi = h / (h + m + f) if h + m + f else 0
        if csi > bc:
            best, bc = round(t, 2), csi
    return best


# ───────────────────────── 套用(run.py 預報時呼叫) ─────────────────────────
def kind_cal(calib, kind):
    return (calib or {}).get(kind) or {}


def model_weights(calib, kind, base):
    mw = kind_cal(calib, kind).get("model_weights") or {}
    return {m: mw.get(m, w) for m, w in base.items()}


def thr_hi(calib, kind):
    return kind_cal(calib, kind).get("thr_hi", 0.6)


def finalize(r, kind, lead, calib, site, ens=None):
    c = kind_cal(calib, kind)
    row_s = r["score"]
    r["score_fc"] = round(row_s, 1)
    nc = r.get("score_nc")
    if nc is not None and lead <= 3.5:
        w = (c.get("nowcast_w") or {}).get("short", 0.4)
        w = w if lead <= 1.5 else w * 0.6
        row_s = (1 - w) * row_s + w * nc
    r["score"] = round(row_s, 1)
    row = {"s": row_s, "lead": lead, "site": site, "factors": r.get("factors"),
           "ens": ens["frac"] if ens else None}
    r["prob"] = round(predict(c.get("model") or PRIOR_MODEL, row), 3)
    vals = list((r.get("models") or {}).values())
    if ens:
        r["ens"] = {k: round(v, 3) if isinstance(v, float) else v for k, v in ens.items()}
        r["score_range"] = [round(ens["p10"], 1), round(ens["p90"], 1)]
    elif len(vals) >= 2:
        r["score_range"] = [round(min(vals), 1), round(max(vals), 1)]
    return r


# ───────────────────────── 每小時更新 ─────────────────────────
def update(out, now=None, base_weights=None):
    now = now or datetime.now(timezone.utc).timestamp()
    hist = list(store.read_ndjson(os.path.join(out, "history")))
    truth_rows = {}
    for r in store.read_ndjson(os.path.join(out, "truth")):
        truth_rows[(r["site"], r["kind"], r["date"])] = r
    truth_rows = list(truth_rows.values())
    labels = read_labels(os.path.join(out, "labels.csv"))
    old = store.load_json(os.path.join(out, "model", "calibration.json"), {})
    new = {"updated": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="minutes")}
    report = {}

    # 偵測器與來源可信度(全域)
    det, det_ba = learn_detector(truth_rows, labels, old)
    new["detector"] = det or old.get("detector") or dict(DETECTOR_PRIOR)
    tw = {}
    for src in ("sat", "cwa"):
        a = agreement(truth_rows, labels, src, new["detector"])
        if a is not None:
            tw[src] = round(clamp(2 * a - 1, 0.2, 1.0), 3)
    new["truth_w"] = dict(TRUTH_W0, **tw)
    oc = outcomes(truth_rows, labels, new)

    for kind in KINDS:
        rows = build_samples([h for h in hist if h["kind"] == kind], {k: v for k, v in oc.items() if k[1] == kind}, now)
        ev = {r["k"] for r in rows}
        old_c = old.get(kind) or {}
        entry, rep = {}, {"snapshots": sum(1 for h in hist if h["kind"] == kind), "events": len(ev),
                          "by_src": {s: sum(1 for k in ev if oc[k][2] == s) for s in ("human", "sat", "cwa")},
                          "base_rate": round(sum(oc[k][0] for k in ev) / len(ev), 3) if ev else None}
        wf = walk_forward(kind, rows, old_c.get("champion"))
        if wf is None:
            entry.update({"champion": "prior", "model": dict(PRIOR_MODEL)})
            rep["decision"] = f"樣本不足({len(ev)}/{MIN_EVENTS} 場有真值),使用先驗"
            oos_champ = []
        else:
            cands, wfl, oos = wf
            name, why = choose(wfl, old_c.get("champion"))
            entry["champion"], rep["decision"], rep["walk_forward_brier"] = name, why, {k: round(v, 4) for k, v in wfl.items()}
            entry["model"] = fit_model(cands[name], rows)
            oos_champ = oos[name]
            tr_w = learn_weights(kind, rows, base_weights or {m: 1.0 for r in rows for m in r["models"]})
            if tr_w:
                entry["model_weights"] = tr_w
            t = learn_threshold(oos_champ)
            if t:
                entry["thr_hi"] = t
        nw, nres = learn_nowcast(rows, entry["model"])
        if nw is not None:
            entry["nowcast_w"] = {"short": nw}
            rep["nowcast_brier_by_w"] = nres
        # ── 回測報告:以 walk-forward 樣本外預測為準(沒有時才用樣本內)
        preds = oos_champ or [(predict(entry["model"], r), r["y"], r["w_test"], r["lead"], r["k"]) for r in rows]
        if preds:
            ps, ys, ws = [p[0] for p in preds], [p[1] for p in preds], [p[2] for p in preds]
            base = sum(y * w for y, w in zip(ys, ws)) / sum(ws)
            bs, bc = brier(ps, ys, ws), brier([base] * len(ys), ys, ws)
            rep.update({"oos": bool(oos_champ), "brier": round(bs, 4), "BSS": round(1 - bs / bc, 3) if bc > 0 else None,
                        "AUC": auc(ps, [int(y) for y in ys])})
            th = thr_hi({kind: entry}, kind)
            rep["by_lead"] = {}
            for L in LEADS:
                sel = [(p, y, w) for p, y, w, lead, _ in preds if abs(lead - L) <= 1.5]
                if len(sel) >= 5:
                    h = sum(1 for p, y, _ in sel if p >= th and y)
                    m = sum(1 for p, y, _ in sel if p < th and y)
                    f = sum(1 for p, y, _ in sel if p >= th and not y)
                    b_ = brier([s[0] for s in sel], [s[1] for s in sel], [s[2] for s in sel])
                    rep["by_lead"][str(L)] = {"n": len(sel), "mean_p": round(sum(s[0] for s in sel) / len(sel), 3),
                                              "obs": round(sum(s[1] for s in sel) / len(sel), 3), "brier": round(b_, 4),
                                              "POD": round(h / (h + m), 2) if h + m else None,
                                              "FAR": round(f / (h + f), 2) if h + f else None,
                                              "CSI": round(h / (h + m + f), 2) if h + m + f else None}
            bins = [[0, 0.0, 0.0] for _ in range(5)]
            for p, y, w, _, _ in preds:
                i = min(4, int(p * 5))
                bins[i][0] += 1; bins[i][1] += p; bins[i][2] += y
            rep["reliability"] = [{"bin": f"{i * 20}-{i * 20 + 20}%", "n": n, "mean_p": round(sp / n, 3),
                                   "obs": round(so / n, 3)} for i, (n, sp, so) in enumerate(bins) if n]
            recent = sorted({p[4] for p in preds}, key=lambda k: k[2])[-25:]
            rp = [(p, y) for p, y, _, _, k in preds if k in recent]
            if len(recent) >= 15 and rp:
                gap = sum(p for p, _ in rp) / len(rp) - sum(y for _, y in rp) / len(rp)
                rep["drift"] = {"recent_events": len(recent), "mean_p_minus_obs": round(gap, 3),
                                "alert": abs(gap) > 0.2}
        new[kind], report[kind] = entry, rep
    report["_global"] = {"detector": new["detector"], "detector_BA_vs_human": det_ba, "truth_w": new["truth_w"],
                         "human_labels": len(labels), "truth_rows": len(truth_rows)}
    store.dump_json(os.path.join(out, "model", "calibration.json"), new, indent=1)
    store.dump_json(os.path.join(out, "model", "report.json"), report, indent=1)
    with open(os.path.join(out, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(render_report(report, new))
    return new, report


def render_report(report, calib):
    g = report.get("_global", {})
    L = ["# 回測與自我學習報告(以「事件是否真的發生」為準)", "", f"更新:{calib.get('updated')}", "",
         "> 真值來源:人工回報(labels.csv)> 氣象署測站(雲海/琉璃光)> Himawari 衛星(火燒雲)。", "> 指標一律使用 walk-forward 樣本外預測(沒有足夠樣本時才用樣本內,並標示)。", "",
         f"人工回報 {g.get('human_labels', 0)} 筆|觀測真值 {g.get('truth_rows', 0)} 筆|來源可信度 {g.get('truth_w')}|"
         f"衛星偵測器 {g.get('detector')}(與人工一致度 BA={g.get('detector_BA_vs_human')})", ""]
    for k in KINDS:
        r = report.get(k, {})
        c = calib.get(k, {})
        L += [f"## {KIND_ZH[k]}(`{k}`)", "",
              f"- 有真值事件 {r.get('events', 0)} 場(人工 {r.get('by_src', {}).get('human', 0)} / 衛星 {r.get('by_src', {}).get('sat', 0)} / 測站 {r.get('by_src', {}).get('cwa', 0)})|歷史發生率 {r.get('base_rate')}",
              f"- 冠軍模型:`{c.get('champion')}`|決策:{r.get('decision')}"]
        if r.get("walk_forward_brier"):
            L.append(f"- walk-forward Brier:{r['walk_forward_brier']}")
        if "brier" in r:
            L.append(f"- 回測({'樣本外' if r.get('oos') else '樣本內'}):Brier={r['brier']},BSS={r['BSS']},AUC={r['AUC']}|告警門檻 thr_hi={c.get('thr_hi', 0.6)}")
        if c.get("model_weights"):
            L.append(f"- 學到的來源權重:{c['model_weights']}")
        if c.get("nowcast_w"):
            L.append(f"- 學到的衛星 nowcast 混入權重:{c['nowcast_w']}(各權重 Brier:{r.get('nowcast_brier_by_w')})")
        if r.get("drift"):
            L.append(f"- 漂移監控:{r['drift']}{' ⚠ 偏離過大' if r['drift']['alert'] else ''}")
        if r.get("by_lead"):
            L += ["", "| 領先(h) | n | 平均預報機率 | 實際發生率 | Brier | POD | FAR | CSI |", "|---|---|---|---|---|---|---|---|"]
            L += [f"| {a} | {v['n']} | {v['mean_p']} | {v['obs']} | {v['brier']} | {v['POD']} | {v['FAR']} | {v['CSI']} |" for a, v in r["by_lead"].items()]
        if r.get("reliability"):
            L += ["", "可靠度分箱:" + "；".join(f"{b['bin']} n={b['n']} 預報{b['mean_p']}/實際{b['obs']}" for b in r["reliability"])]
        L.append("")
    return "\n".join(L)
