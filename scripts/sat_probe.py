"""衛星模組驗證(上線前請先跑一次):確認 S3 目錄結構、tile 檔名、能否組出場景,並輸出預覽圖讓人眼檢查。
用法: PYTHONPATH=src python scripts/sat_probe.py [YYYY-MM-DDTHH:MMZ]   (需要 pip install numpy h5netcdf)
輸出: .cache/sat/probe.png(偽真彩色)與偵測器特徵;人眼確認「雲是雲、海是海、紅光雲在太陽方向」後再信任衛星真值。"""
import sys
import time
from datetime import datetime, timezone

from fcf import sat
from fcf.sat import BUCKET, PREFIX, Himawari, s3_list

ts = datetime.fromisoformat(sys.argv[1].replace("Z", "+00:00")).timestamp() if len(sys.argv) > 1 else time.time() - 3600
print("1) bucket 根目錄:", s3_list("", delimiter="/")[:20])
print("2) AHI 相關頂層目錄:", [p for p in s3_list("AHI", delimiter="/")][:20])
hs = Himawari()
keys = hs._slot_keys(ts)
print(f"3) 依假設的前綴 {PREFIX} 列出 {len(keys)} 個檔;範例:", keys[:3])
if not keys:
    print("   ✗ 找不到檔案 → 請依上面 1)2) 的實際結構調整 config.sat.prefix")
    sys.exit(1)
sc = hs.scene(ts)
print("4) 場景:", None if sc is None else f"{sc.ny}x{sc.nx}, ts={sc.ts}")
if sc:
    print("   預覽圖:", sat.preview(sc, ".cache/sat/probe.png"))
    site = {"lat": 25.15, "lon": 121.43}
    print("   偵測器特徵(八里,朝西):", sat.glow_features(sc, site, 250.0, 299.0))
