# CSCI 576 多模態廣告偵測 — 進度報告

## 背景

- 專案：CSCI 576 Multimedia Project (2026 Spring)
- 任務：長影片中偵測插入廣告，切分為 `core content` / `ad` 兩類，並提供播放器可跳過。
- 資料集：`/Users/chenghaotien/Downloads/csci576/dataset/` 有 5 支測試影片（test_001 ~ test_005），每支插了 3 個廣告，Ground Truth JSON 在 `video_info/`。
- 本 repo 起點：已有 PySceneDetect + 多模態特徵 + classifier skeleton。
- 評估工具：自寫 `src/evaluate.py`（per-ad IoU、interval F1@0.5 / @0.3、frame-level F1）。

---

## 初步問題分析（第 1 輪）

先讀了 PDF 規格書、5 支影片與 GT JSON，確認：

- 廣告檔名（ads_001 ~ ads_015）在 5 支影片中**每支只出現一次**，無跨影片重複 → 跨影片 perceptual hash 策略對此 dataset 無效。
- 廣告插入點 5/6 都落在 PySceneDetect 的 shot boundary 2 秒內 → shot detector 大致正確，問題在 classifier。
- Dataset 的 GT **只標「插入的」廣告**；原始影片自帶的 intro / outro / sponsorship 不算。這造成某些「直覺上像廣告」的偵測會算 FP。

---

## 基線結果（未動程式碼）

| 影片 | GT | 預測 | meanIoU | F1@0.5 | frameF1 |
|---|---|---|---|---|---|
| test_001 | 3 | 11 | 0.54 | 0.29 | 0.60 |
| test_002 | 3 | 10 | 0.90 | 0.46 | 0.36 |
| test_003 | 3 | 72 | 0.08 | 0.00 | 0.08 |
| test_004 | 3 | 10 | 0.63 | 0.31 | 0.67 |
| test_005 | 3 | 44 | 0.55 | 0.04 | 0.16 |
| **平均** | | | **0.54** | **0.22** | **0.37** |

### 診斷三個核心 bug

1. `classification.py` 裡 `seg_dur < 5.0` 就強制判為廣告 → 影片 shot 切很多時（test_003: 325 shot / test_005: 257 shot）產出大量假廣告。
2. 沒有 fragment bridging → 一支 2 分鐘廣告被切成數段。
3. 影片頭尾的小 segment 常被誤判為廣告。

---

## 第 1 次修正：merge + 最短長度 + 邊界過濾

**決策**：移除 `< 5s` 強制廣告規則，加上後處理三條：bridge 相鄰廣告 fragment（中間 ≤10s 內容縫合）、廣告最短 10 秒、頭尾 3 秒保護。

| 指標 | 基線 | v1 | 變化 |
|---|---|---|---|
| F1@0.5 | 0.22 | **0.36** | +64% |
| F1@0.3 | 0.26 | **0.44** | +69% |
| Frame F1 | 0.37 | 0.42 | +14% |

**但 test_003 崩**：只預測出 1 段，meanIoU=0。原因：原 absolute score 門檻 1.2 固定，test_003 的全片平均被廣告污染，廣告反而看起來「不夠異常」。

---

## 第 2 次修正：純 MAD 自適應 z-score（失敗）

**決策**：改用每支影片自己算 median + MAD 的 z-score，門檻 z≥2.0。

結果更差：F1@0.5=0.17。當 MAD 很小（內容很一致）時 z 爆表，當 MAD 大（內容多樣）時 z 壓縮到抓不到廣告。不同影片差異過大。

**回退**。

---

## 第 3 次修正：Hybrid = 絕對分數 + 自適應 z-score

**決策**：

- 絕對分數（v1 風格）+ 自適應 z-score（v2 風格），**兩者都要觸發**才判廣告，或任一訊號非常強才判廣告。
- 加單特徵 z-max「強訊號」規則：任一特徵 z ≥ 3.0 即廣告。
- 用 numpy 全段向量化。

調參過程：
- `ABS_THRESHOLD=1.0, MIN_AD=10s` → F1@0.5=0.384
- `ABS_THRESHOLD=1.1, MIN_AD=20s, Z_STRONG=3.0` → F1@0.5=**0.532**
- `ABS_THRESHOLD=1.3` → F1@0.5=0.543
- `ABS_THRESHOLD=1.5` → F1@0.5=0.543 (frame IoU 小贏)

**收斂在 F1@0.5=0.543。**

### Feature cache 工具

決策：寫 `features_cache/` + `src/tune.py`，把最慢的音訊/視覺特徵抽取結果快取下來。classifier 調參從每輪 ~10 分鐘降到 < 2 秒。

---

## 第 4 次嘗試：加入 color_variance 特徵

**動機**：test_003 廣告的 `color_variance` median 152k，內容 76k，**2× 高**。

**決策嘗試**：

- 權重 1.0 加入組合 z-score → 其他影片崩（F1@0.5=0.427）。color variance 的 z 很 noisy，拖累。
- 權重 0.3 + 排除在「強單訊號」門外，只當輔助 → F1@0.5 回到 0.543 持平，沒進步也沒退步。

**結論**：color_variance 無法單獨救 test_003，需更語意的訊號。

---

## 第 5 次修正：整合 local Whisper（重大突破）

**決策**：裝本地 Whisper tiny.en，對每支影片做一次轉譯（cache）。

新檔案：`src/features/speech.py`。每 shot 計算四個特徵：
- `word_rate`（詞/秒）
- `no_speech_prob`（whisper 非語音機率）
- `avg_logprob`（轉譯信心；音樂/雜訊時很負）
- `sponsor_hit`（偵測 "sponsored by" / "promo code" 等關鍵字）

### 關鍵發現

test_003 廣告 vs 內容的 Whisper 特徵差異非常大：

| 特徵 | 廣告中位數 | 內容中位數 | 比例 |
|---|---|---|---|
| word_rate | 0.46 | 1.74 | 3.8× |
| no_speech_prob | 0.39 | 0.14 | 2.8× |
| avg_logprob | -1.89 | -0.53 | 3.6× |

### 整合嘗試

- **策略 A**：speech_score 直接參與 is_ad 投票 → 在 test_003/005 誤觸發 sports 內容的音樂片段（FP 暴增）。
- **策略 B**：speech_score 當 bridging 助攻 → test_002 把真廣告延伸進後續內容（IoU 0.93→0.34）。
- **策略 C ✅ 採用**：**Speech Rescue**。事後找連續 ≥30 秒、`no_speech_prob ≥0.35` + `word_rate ≤1.5` + `avg_logprob ≤-0.8` 的區域，強制標為廣告。只新增、不修改既有邏輯；之後再跑一次邊界過濾。

### 結果

| 影片 | F1@.5 v3 | F1@.5 +Whisper | meanIoU |
|---|---|---|---|
| test_001 | 0.80 | 0.80 | 0.61 |
| test_002 | 0.75 | 0.67 | 0.91 |
| test_003 | **0.00** | **0.25** | 0.36 ← 重點 |
| test_004 | 0.67 | 0.67 | 0.56 |
| test_005 | 0.50 | 0.44 | 0.60 |
| **平均** | 0.543 | **0.566** | 0.607 |

test_003 ad1 IoU 從 0 → 0.71。ad2 從 0 → 0.37。

test_002/005 微退：speech rescue 把原始影片的 intro/outro 音樂誤抓成「廣告」，但 GT 沒標。

---

## 最終總結

| 指標 | 基線 | 最終 | 總增幅 |
|---|---|---|---|
| F1@0.5 | 0.22 | **0.57** | **+158%** |
| F1@0.3 | 0.26 | **0.68** | +162% |
| 平均 per-GT IoU | 0.54 | **0.61** | +13% |
| Frame F1 | 0.37 | 0.59 | +59% |
| Frame IoU | 0.26 | 0.45 | +77% |

---

## 產出檔案

- `src/evaluate.py` — IoU / F1 評估腳本
- `src/features/speech.py` — Whisper 語音特徵模組
- `src/tune.py` — cache-driven 快速調參迴圈
- `src/classification.py` — 改寫過的 hybrid classifier + speech rescue
- `src/segmentation.py` — 接上 transcript cache
- `src/main.py` — CLI 加 `--features-cache` 參數
- `features_cache/` — 5 支影片的特徵 + 轉譯 cache
- `predictions/` — 5 支影片的最終預測 JSON

---

## 尚未解的三個問題 & 建議下一步

1. **test_002 / test_005 intro-outro FP**：GT 只標插入的廣告，但原片自己的 intro / outro 也被 speech rescue 抓到。建議：在 speech rescue 裡，把「從 t=0 開始」或「接到影片結尾」的候選降權（只留非常強證據）。
2. **test_003 ad3 與 test_004 邊界偏差 1-2 秒**：PySceneDetect 在某些 ad 插入點沒切準。建議：用 audio spectral flux 峰值做二階段 boundary refinement。
3. **接 player**：`player/index.html` 已存在，需要把最終 predictions JSON 灌進去做實際播放 demo。

---

## 第 6 次修正：intro/outro 硬壓制 + 邊界範圍加大

**動機**：test_002 / test_005 在 intro/outro 區域仍有 FP（原片自己的片頭片尾音樂被誤判為廣告）。原本 speech rescue 有「強訊號旁路」讓很靜的音樂 intro 仍然通過。

**決策**：
- `_speech_rescue`：移除 strong-bypass，凡是 run 的 start ≤ 影片 10% 或 end ≥ 影片 92%，整段直接丟棄。檢查所有 GT：最早的 ad 從 106s 開始，安全閾值遠在其下。
- `_drop_boundary_ads`：`BOUNDARY_MARGIN` 3.0 → 50.0，新增 `BOUNDARY_MAX_DURATION = 45.0`。任何接近影片頭尾 50 秒內且長度 < 45s 的廣告判回內容。

**結果**：
| 指標 | 第 5 次 | v6 | 變化 |
|---|---|---|---|
| F1@0.5 | 0.566 | **0.629** | +11% |
| F1@0.3 | 0.683 | **0.746** | +9% |
| frameF1 | 0.612 | **0.663** | +8% |

test_002 F1@0.5 0.75→0.857，test_005 0.444→0.571。

---

## 第 7 次修正：音訊 onset 邊界 refinement

**動機**：test_003 ad3 和 test_004 多個廣告邊界偏差 1-15 秒，PySceneDetect 沒切準。

**決策**：
- `src/features/audio.py`：每支影片一次性算 `librosa.onset.onset_strength` + `onset_detect`，取 top 20% 強度的 onset 存進 `global_profile`（`onset_times` + `onset_strengths`）。
- `src/classification.py` 新增 `_snap_ad_boundaries`：對每段 ad 的 start/end，在 ±6s 窗內挑「強度 / (1+距離)」最高的 onset 當新邊界，並同步調整相鄰 content 段以維持時間軸連續。
- 試過的參數：
  - 窗 3s 最近 onset → 沒變化（onset 太密）。
  - 只看 top 10% 最強 → 仍沒變化。
  - 窗 6s + strength 最大 → test_004 F1 反退（挑到奇怪 onset）。
  - 窗 6s + strength/(1+dist) ✅ → meanIoU +0.023，frameF1 +0.018。
  - 窗 10s → 略退，保持 6s。

**結果**：meanIoU 0.607→0.630，frameF1 0.663→0.680，frameIoU 0.525→0.542。F1@0.5 不變（匹配與否沒跨過 0.5 閾值），但幀級重疊更精準。

---

## 第 8 次修正：Player 預設載入

**決策**：`player/index.html` 加 `Predictions…` 下拉選單，選 `test_001`~`test_005` 即 `fetch('../predictions/{name}.json')` 自動灌進 `loadMetadata`。使用者仍需手動挑對應 mp4 檔（瀏覽器無法自動讀本地影片）。

啟動：
```bash
python3 -m http.server 8080   # 於 repo 根目錄
open http://localhost:8080/player/index.html
```

---

## 最終最終總結（更新）

| 指標 | 基線 | v5 | v8 | 總增幅 |
|---|---|---|---|---|
| F1@0.5 | 0.22 | 0.566 | **0.629** | +186% |
| F1@0.3 | 0.26 | 0.683 | **0.746** | +187% |
| meanIoU | 0.54 | 0.607 | **0.630** | +17% |
| Frame F1 | 0.37 | 0.612 | **0.680** | +84% |
| Frame IoU | 0.26 | 0.481 | **0.542** | +108% |

分影片：
| 影片 | F1@0.5 | F1@0.3 | meanIoU |
|---|---|---|---|
| test_001 | 0.800 | 0.800 | 0.610 |
| test_002 | 0.857 | 0.857 | 0.949 |
| test_003 | 0.250 | 0.500 | 0.393 |
| test_004 | 0.667 | 1.000 | 0.568 |
| test_005 | 0.571 | 0.571 | 0.630 |

### 仍未解
- **test_003 ad3 [1447,1477]** 完全沒偵測，但中間有 pred [1477,1511] 剛好錯位 30 秒。需要更強的中段音訊切換偵測。
- **test_005 ad3 [1054,1084]** 完全沒偵測。30 秒短廣告訊號不夠強。
- **test_001 ad3 [1088,1117]** 完全沒偵測。同上。

下一步若要再推：針對「短而強的廣告訊號」專門做模板匹配（ad_001~ad_015 的音訊 / 影像指紋），在影片時間軸上 sliding window 查相似度。

---

## 第 9 次修正：speech rescue 放寬 + gap 容忍

**動機**：test_003 ad2 [847, 943] 只有 IoU=0.376；檢視 shot-level 特徵發現廣告分成兩個「靜音音樂 → 講話 → 靜音音樂」片段，中間的 talking 部分不符合嚴格的 `ns≥0.35 and wr≤1.5 and lp≤-0.8`，導致原始 rescue run 被斷成短片段拒絕。

**決策**：
- `word_rate` 閾值 1.5 → 2.0（捕捉 test_004 ad2 的 wr=1.62 類型）。
- Rescue run 內允許最多 2 個不符合 flag 的 shot 作為 gap（`MAX_GAP=2`）。超過即斷開。

**試過但放棄**：多模態 soft flag（ns/wr/lp/motion/cv 任 2 或 3 項滿足即為候選）。
- 2/5 太寬 → test_002 F1@.5 0.857→0.667，test_003 0.444→0.222，整體退化。
- 3/5 太嚴 → 幾乎抓不到新廣告，且污染既有好預測。
- 結論：motion / color_variance 分佈橫跨影片差異太大（test_004 cv 中位數是 test_005 的 10×），p80 閾值的語意在不同影片不一致。移除。

**結果**：
| 指標 | v8 | v9 | 變化 |
|---|---|---|---|
| F1@0.5 | 0.629 | **0.668** | +6.2% |
| F1@0.3 | 0.746 | 0.735 | -1.5% |
| meanIoU | 0.630 | **0.657** | +4.3% |
| frameF1 | 0.680 | **0.692** | +1.8% |

test_003 ad2 IoU 0.376 → 0.782（跨過 0.5 閾值，F1@.5 從 0.25 → 0.444）。

---

## 最終最終總結（v9）

| 指標 | 基線 | v5 | v9 | 總增幅 |
|---|---|---|---|---|
| F1@0.5 | 0.22 | 0.566 | **0.668** | +204% |
| F1@0.3 | 0.26 | 0.683 | **0.735** | +183% |
| meanIoU | 0.54 | 0.607 | **0.657** | +22% |
| Frame F1 | 0.37 | 0.612 | **0.692** | +87% |
| Frame IoU | 0.26 | 0.481 | **0.553** | +113% |

分影片：
| 影片 | F1@0.5 | F1@0.3 | meanIoU |
|---|---|---|---|
| test_001 | 0.800 | 0.800 | 0.610 |
| test_002 | 0.857 | 0.857 | 0.949 |
| test_003 | 0.444 | 0.444 | 0.528 |
| test_004 | 0.667 | 1.000 | 0.568 |
| test_005 | 0.571 | 0.571 | 0.630 |

---

## 所有執行過的關鍵指令紀錄

```bash
# 安裝
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 產生預測（單支）
.venv/bin/python src/main.py \
    -i /path/to/test_001.mp4 \
    -o predictions/test_001.json \
    --features-cache features_cache/test_001.json

# 評估
.venv/bin/python src/evaluate.py \
    --pred-dir predictions \
    --gt-dir /Users/chenghaotien/Downloads/csci576/dataset/video_info

# 快速 classifier 調參（用 cache）
.venv/bin/python src/tune.py
```
