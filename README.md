# fire-cloud-forecast v2

每小時自動更新「未來 24 小時」的 **日落火燒雲 / 日出朝燒 / 雲海 / 琉璃光** 發生機率與時間點,
並在同一個 workflow 內完成 **以「事件有沒有真的發生」為準的回測 + 多重自我校正**。核心純標準庫;衛星功能選用 numpy + h5netcdf。

## 架構
```
 GitHub Actions(每小時 :17)
 ├ 1 取資料  Open-Meteo 預報(ECMWF/ICON/GFS)+ 空品(CAMS 氣膠)+ 集合預報(ECMWF 51 成員)
 │           Meteoblue(選填,共識成員,支援 asl 海拔)  氣象署測站(選填)  Himawari-9 衛星(選填)
 ├ 2 評分    光路幾何 → 0-100 分;衛星 nowcast 修正近時段;集合成員 → 發生比例與分數區間
 ├ 3 機率    冠軍模型(自動挑選)把分數校準成「真實發生機率」
 ├ 4 真值    剛結束的事件 → 觀測型真值(測站 / 衛星 / 人工回報),不是模式分析場
 ├ 5 學習    walk-forward 冠軍/挑戰者、來源權重、nowcast 權重、偵測器閾值、告警門檻、漂移監控
 └ 6 發布    commit 到 data 分支 → GitHub Pages(JSON + meteogram SVG + REPORT.md)
 main : 程式、config、tests        data : next24h.json latest.json ranking.json sites/*.json charts/*.svg
                                          history/ truth/ obs/ (ndjson)  model/calibration.json  REPORT.md  labels.csv
```

## 資料來源
| 來源 | 用途 | 需要 |
|---|---|---|
| Open-Meteo Forecast | 三模式雲層/濕度/氣壓層剖面(主資料) | 免金鑰(非商用) |
| Open-Meteo Air Quality | CAMS 氣膠光學厚度 AOD、PM2.5 → 空氣透明度 | 免金鑰 |
| Open-Meteo Ensemble | ECMWF 成員各自評分 → 發生比例、p10/p90 | 免金鑰;只對分數最高的前 N 個事件做(額度) |
| Open-Meteo Previous Runs | 冷啟動:還原「N 天前發布的預報」 | 免金鑰 |
| Meteoblue | 第 4 個共識成員(clouds-3h + basic-1h,內插為逐時) | `METEOBLUE_API_KEY`(Free API 試用 1 年) |
| 氣象署開放資料 | 山上/谷地測站 → 雲海 nowcast **與雲海/琉璃光真值** | `CWA_API_KEY`(免費) |
| Himawari-9(AWS) | 雲層 nowcast + **火燒雲真值**(紅光雲偵測) | numpy + h5netcdf;先跑 `scripts/sat_probe.py` |

## 評分模型(五個環節 → 可計算特徵)
太陽低角度(天文)→ **光路**(球面幾何:太陽俯角 δ 時,距離 x 的雲要高於 R·(δ−x/R)²/2 才被照亮;光線沿途最低高度決定穿過哪一層雲;低雲牆擋不到高層雲的光,中層雲牆才會)
→ **泛紅**(光線貼地越低越紅,高峰自然落在日落後)→ **空氣**(能見度、AOD、PM2.5)→ **畫布**(被照亮、仰角合適、雲量適中的中/高雲)→ **頭頂開口**(低雲/降雨)。
雲海:低地參考點的氣壓層剖面估濕層頂 vs 站點海拔 × 厚度 × 上方穩定度 × 風 × 降雨;琉璃光 = 夜間雲海 × 燈火 × 風柔 × 夜間時段。
任一來源在某取樣點缺資料時,用其他來源的共識補該點(不會讓整個來源掉出共識)。

## 回測:真值是什麼(重點)
**不再用「預報像不像分析場」。** 一律用事件是否真的發生:
| 現象 | 真值 | 判定 |
|---|---|---|
| 雲海 | 氣象署山上/谷地測站每小時取樣(`obs/`) | 窗口內多數取樣:谷地近飽和 + 山上乾 + 穩定層 |
| 琉璃光 | 同上 + 山頂風速 ≤ 5 m/s + 夜間 | 同上 |
| 火燒雲/朝燒 | Himawari 可見光:太陽方向扇形內「被紅光照亮的雲」(紅/藍輻射比)× 頭頂低雲 | `glow_index ≥ τ`;**原始證據(各紅度閾值的覆蓋比例)全部存檔**,閾值日後被學習修正時可重新判定,不必重抓衛星 |
| 任何 | **人工回報**(Issue 表單,0–3 級,≥2 算發生) | 優先於觀測型真值 |
觀測型真值的權重不是拍腦袋:它與人工回報重疊的事件越多,「一致度」越高,權重自動升高(`truth_w`)。

## 自我校正機制(每小時自動;每項都有 walk-forward 守門)
1. **冠軍/挑戰者**:prior、logit(s)、+領先時間、+站點偏差(階層收縮)、+各環節因子指數(學出「哪個環節比物理先驗更重要」)、+集合預報比例、isotonic;
   expanding-window walk-forward Brier 最低者上任,需勝過現任 2% 才換,勝不過 prior 就**自動回退**。
2. **時間衰減**:半衰期 150 天,跟上季節與模式更新造成的漂移。
3. **來源權重學習**:ECMWF/ICON/GFS/meteoblue 依各自 Brier 指數加權(Hedge),向先驗收縮;概念同 meteoblue 的 mLM「依近期表現加權多模式」。
4. **nowcast 混入比例**:在真值上比較「純預報 vs 衛星修正」,學出 0~1 的最佳混入權重(`score_fc` 與 `score_nc` 兩份分數都存,可重新混合)。
5. **偵測器自我校準**:衛星紅度閾值與判定門檻用人工回報網格搜尋(Balanced Accuracy);同時計算測站/衛星真值與人工回報的一致度。
6. **告警門檻**:「高」的機率門檻取使 CSI 最大者。
7. **漂移監控**:近 25 場平均機率 vs 實際發生率,偏離 >0.2 在 `REPORT.md` 標示。
8. **冷啟動**:`backfill` 工作流程用 Previous Runs + 衛星封存補出歷史快照與真值;標註 <20 場時用先驗。

報告見 data 分支 `REPORT.md`:樣本外 Brier、BSS、AUC、可靠度分箱、依領先時間的 POD/FAR/CSI、冠軍決策理由。

## 部署
1. 推上 `main`;Settings → Actions → Workflow permissions 選 **Read and write**。
2. Secrets(皆選填):`CWA_API_KEY`、`METEOBLUE_API_KEY`、`OPEN_METEO_API_KEY`。**沒有 CWA 金鑰 → 雲海/琉璃光沒有觀測型真值,只能靠人工回報學習。**
3. 先跑 `PYTHONPATH=src python scripts/sat_probe.py`(需 `pip install -r requirements-sat.txt`):確認 S3 目錄、tile 檔名、組出的場景與 `.cache/sat/probe.png`;人眼確認「雲是雲、紅光雲在太陽方向」。目錄結構不同就改 `config.sat.prefix`。
4. Actions → `hourly-forecast` → Run workflow,看 log 有沒有 `[fetch]`/`[meteoblue]`/`[sat]` 失敗訊息。
5. (建議)Actions → `backfill-cold-start` → days=30。
6. Settings → Pages → Source 選 `data` 分支。網頁:`https://<user>.github.io/<repo>/`(`web/index.html`,每小時由 workflow 複製到 data 分支);
   API:`.../next24h.json`、`.../charts/<site>.svg`、`.../REPORT.md`。
7. 校正 `config/sites.json` 的座標/海拔(`verify: true`)與山區 `base`/`light`。回報入口:Issues → 「觀測回報」。

## 前端(`web/index.html`)
單一靜態檔,零依賴,讀取同目錄的 `next24h.json` / `meta.json`:24 小時時間軸(每個色塊=一個地點的高峰窗口,越不透明機率越高)、
依現象篩選的地點清單(展開看原因、各環節分數、各來源分數、meteogram)、資料過期警示、事後回報連結。亮/暗色自動切換。
已用真實管線輸出 + 最小假 DOM 跑過渲染與互動邏輯;**尚未在真實瀏覽器/手機上目視驗證版面**。

## 已知限制(請先讀)
- **我在沙箱無法連外網**:所有「真實連線」都沒測過——Open-Meteo 新端點(集合/空品/Previous Runs)、Meteoblue 欄位名、CWA 欄位、Himawari S3 目錄與 NetCDF 格式。
  已測的是:解析函式(用合成回應)、固定網格投影往返、tile 組裝、偵測器、全部學習機制、端到端流程(假衛星場景 + 合成氣象資料)。
- Meteoblue 免費版 clouds 套件只有 ≥3 小時解析度;欄位名稱採容錯解析,對不上會把實際 keys 印到 log。
- 衛星偵測是**代理指標**:衛星從上往下看被紅光照亮的雲頂,不等於地面看到的雲底;閾值是先驗,必須靠人工回報校準(機制已內建)。
- 雲海/琉璃光的真值只有測站;測站配不到「山上+谷地」一對就沒有真值。ECMWF 集合預報只對火燒雲做。
- 全球模式 ~25 km,分不出同一區哪個機位會燒;山谷雲海是 NWP 弱項。
- Open-Meteo 免費版僅限非商用(CC BY 4.0 需標註);GitHub 共享 IP 偶爾 429(已重試並退回快取);GitHub cron 會延遲;公開 repo 60 天無活動排程會停用(附 keepalive)。
- 所有權重/係數是物理先驗,上線頭幾週請以 `REPORT.md` 為準。

## 開發
```
pip install numpy                      # 僅測試需要(衛星組裝測試)
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m fcf.run --mock --out /tmp/out     # 不連網乾跑
```
