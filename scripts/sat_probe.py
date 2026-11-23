"""衛星模組驗證(上線前請先跑一次;GitHub 上可用 sat-probe workflow 一鍵執行)。
確認:S3 目錄結構、依假設的前綴能不能列出檔案、各波段/tile 的檔名、能否組出場景,並輸出預覽圖讓人眼檢查。
用法: PYTHONPATH=src python scripts/sat_probe.py [YYYY-MM-DDTHH:MMZ]   (需要 pip install -r requirements-sat.txt)"""
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone

from fcf import sat
from fcf.sat import PREFIX, Himawari, s3_list

ts = datetime.fromisoformat(sys.argv[1].replace("Z", "+00:00")).timestamp() if len(sys.argv) > 1 else time.time() - 3600
print("1) bucket 根目錄:", [p for p in s3_list("", delimiter="/")][:30])
top = [p for p in s3_list("AHI", delimiter="/") if p.endswith("/")]
print("2) AHI 相關頂層目錄:", top)
node, depth = next((p for p in top if "L1b-FLDK" in p), top[0] if top else None), 0
print("3) 逐層往下(各層取字典序最後 = 最新),請對照 config.sat.prefix 的日期層級:")
while node and depth < 7:
    kids = s3_list(node, delimiter="/")
    dirs = sorted(k for k in kids if k.endswith("/") and k != node)
    files = [k for k in kids if not k.endswith("/")]
    print(f"   {node}  子目錄(後3)={[d[len(node):] for d in dirs[-3:]]}  檔案(前2)={[f[len(node):] for f in files[:2]]}")
    if not dirs:
        break
    node, depth = dirs[-1], depth + 1
hs = Himawari()
keys = hs._slot_keys(ts)
print(f"4) 依假設的前綴 {PREFIX} 列出 {len(keys)} 個檔;範例:", keys[:2])
if keys:
    bands = Counter(m.group(1) for k in keys if (m := re.search(r"-B(\d\d)[-_]", k)))
    tiles = {b: {m.group(1) for k in keys if f"-B{b}" in k and (m := re.search(r"-T(\d+)[_.]", k))} for b in ("01", "03", "13")}
    print("   各波段檔案數:", dict(bands))
    print("   B01/B03/B13 的 tile 編號是否相同(程式假設相同):", tiles["01"] == tiles["03"] == tiles["13"], {b: len(v) for b, v in tiles.items()})
else:
    print("   ✗ 找不到檔案 → 依上面 3) 的實際結構調整 config/sites.json 的 sat.prefix(日期層級用 {Y}/{m}/{d}/{H}{M} 這類佔位符)")
    sys.exit(0)
sc = hs.scene(ts)
print("5) 場景:", None if sc is None else f"{sc.ny}x{sc.nx}, ts={sc.ts}")
if sc:
    print("   預覽圖:", sat.preview(sc, ".cache/sat/probe.png"))
    print("   偵測器特徵(八里,朝西):", sat.glow_features(sc, {"lat": 25.15, "lon": 121.43}, 250.0, 299.0))
else:
    print("   ✗ 組不出場景 → 看上方 [sat] 開頭的錯誤訊息(tile 讀取 / 缺依賴 / 波段無 tile)")
