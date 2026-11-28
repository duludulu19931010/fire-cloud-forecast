"""衛星模組驗證(GitHub 上用 sat-probe workflow 一鍵執行;或本機 PYTHONPATH=src python scripts/sat_probe.py [UTC 時間])。
依序確認:S3 目錄 → 檔名 → HSD 檔頭(逐區塊檢查)→ 關鍵檔頭值是否合理 → 台北附近像素值 → 組場景 → 預覽圖與偵測器特徵。
請把完整輸出貼回來:任何一步 ✗ 都會附上足以修正的資訊。"""
import json
import re
import sys
import time
from datetime import datetime, timezone

from fcf import sat

ts = datetime.fromisoformat(sys.argv[1].replace("Z", "+00:00")).timestamp() if len(sys.argv) > 1 else time.time() - 3600
try:
    cfg = json.load(open("config/sites.json", encoding="utf-8"))
except OSError:
    cfg = {}
hs = sat.from_config(cfg, ".cache/sat")
print("1) bucket 根目錄:", sat.s3_list("", delimiter="/")[:30])
keys = hs._slot_keys(ts)
print(f"2) 前綴 {hs.prefix} 列出 {len(keys)} 個檔;範例:", keys[:2])
idx = {}
for k in keys:
    m = re.search(r"_B(\d\d)_FLDK_R(\d\d)_S(\d\d)(\d\d)\.DAT", k)
    if m:
        idx[(int(m.group(1)), int(m.group(3)))] = k
print("   可辨認的 (波段,段) 組合數:", len(idx), "(預期 16 波段 × 10 段 = 160)")
ok = bool(idx)
for band in (13, 1, 3):
    segs = hs.needed_segments(band)
    print(f"3) 波段 B{band:02d}:bbox {hs.bbox} 預期需要的段 = {segs}")
    if not segs or (band, segs[0]) not in idx:
        print("   ✗ 找不到對應檔案"); ok = False; continue
    t0 = time.time()
    try:
        p = sat.parse_hsd(hs._blob(idx[(band, segs[0])]))
    except Exception as e:
        print(f"   ✗ HSD 解析失敗:{type(e).__name__}: {e}"); ok = False; continue
    keep = ("band", "wl", "ncol", "nline", "bits", "sub_lon", "cfac", "lfac", "coff", "loff", "rs", "req", "rpol", "gain", "const", "err", "out", "first_line", "nseg", "segno")
    print("   檔頭:", {k: (round(p[k], 4) if isinstance(p[k], float) else p[k]) for k in keep}, f"(解析 {time.time() - t0:.1f}s)")
    print("   ", {k: round(p[k], 6) for k in ("c0", "c1", "c2", "h", "k", "cc") if k in p} or {"cprime": p.get("cprime")})
    exp = {13: 10.4, 1: 0.47, 3: 0.64}[band]
    print(f"   合理性:波長 {p['wl']:.3f}µm(預期≈{exp});sub_lon {p['sub_lon']:.2f}(預期≈140.7);ncol {p['ncol']};segno/nseg {p['segno']}/{p['nseg']}")
    pix = sat.geo_to_pix(25.05, 121.5, p)
    if pix:
        row = pix[1] - p["first_line"]
        inside = 0 <= row < p["nline"] and 1 <= pix[0] <= p["ncol"]
        v = sat.convert(p, p["counts"][row * p["ncol"] + pix[0] - 1]) if inside else None
        print(f"   台北 (25.05N,121.5E) → col/line={pix},在本段內={inside},換算值={None if v is None else round(v, 3)}"
              + ("  (紅外:應落在 ~250–305 K)" if band == 13 else "  (可見光反照率:日落時台灣在晨昏線,可能很小)"))
sc = hs.scene(ts) if ok else None
print("4) 場景:", None if sc is None else f"{sc.ny}x{sc.nx}, ts={sc.ts}", "" if sc else f"最後錯誤:{hs.last_error}")
if sc:
    flat = [v for r in sc.bt for v in r if v is not None]
    print(f"   紅外亮溫有值格數 {len(flat)}/{sc.ny * sc.nx};範圍 {min(flat):.1f}–{max(flat):.1f} K")
    print("   預覽圖(紅線=台灣粗略海岸線,檢查像素定位;白天無雲處陸海反差最明顯):", sat.preview(sc, ".cache/sat/probe.png", outline=sat.TAIWAN_OUTLINE))
    print("   偵測器特徵(八里,朝西):", sat.glow_features(sc, {"lat": 25.15, "lon": 121.43}, 250.0, 299.0))
