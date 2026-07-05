# business_card — 名片掃描與 vCard 工具組

## 專案地圖
純本機網頁應用(無打包工具、無套件管理檔):瀏覽器前端 + 多個 Python 後端服務。

### 現行版入口
- `scanner.html` + `app.js`:名片掃描主程式(相機、透視裁切、OCR、匯出 vCard)。
  OCR 走 Ollama(經 `dmr_proxy.py` 的 12435 CORS 代理,OpenAI 相容 API)。
- `vcf-checker.html` + `vcf_checker.py`(port 5051):VCF 逐欄審查工具,
  OCR 委派給 5050,不自己載 PaddleOCR。
- `style.css`:現行版與 beta 共用。

### beta 關係
- `scanner-beta.html` + `app-beta.js` 是實驗版:PaddleOCR 為主、DMR 為輔,另接 Docling。
- 與現行版並存、互不引用;改其中一版不影響另一版。

### 後端服務
- `opencv_backend.py`(port 5050):OpenCV 名片偵測/透視裁切 + PaddleOCR
  (延遲載入,OCR 載入失敗不影響裁切功能)。
- `dmr_proxy.py`(port 12435):CORS 代理 → 127.0.0.1:11434(Ollama);
  `/docling/` 路徑另行轉發。`dmr-proxy.js` 為早期 Node 版,已不在啟動流程中。
- `vcf_to_win.py`:CLI,批次把 vCard 轉 Windows 聯絡人相容格式;`vcf_checker.py` 會 import 它。

### 啟動與驗收
- 啟動:`run_scanner.bat`(依序啟動 Ollama、Open WebUI、12435 代理、5050、5051、8000 靜態伺服器)。
- 驗收指令:無正式自動化測試。驗收暫以手動跑 `run_scanner.bat` 成功、
  瀏覽器開 http://localhost:8000/scanner.html 能掃描為準;輔助煙霧測試見 CONVENTIONS.md。
- 指令細節與相依套件:見 `CONVENTIONS.md`。

## 寫程式守則
0. 還原點:動手前跑 git status 確認 repo 受版控且工作區乾淨;
   若無 git → 先 git init 並 commit 現狀,才准改任何檔。
1. 覆誦:這次任務是什麼、要改哪些檔、驗證指令是什麼;
   任一項與任務描述對不上 → 先問,不猜。
2. 不確定的 API/函式/欄位 → 先 grep 本 repo 現有寫法;
   查無此名 → 停下回報「repo 內查無 X」並詢問;禁止自行發明或憑記憶寫。
3. 每次改動 ≤50 行就跑驗證,並在回報附上「跑了什麼指令+輸出末 3 行」,
   無證據視同沒跑:
   - Python 改動:python -m py_compile <改過的檔>;JS 改動:node --check <改過的檔>
   - 動到服務邏輯:加跑對應煙霧腳本(test_ocr_api.py / test_morph_api.py,
     詳見 CONVENTIONS.md;服務未啟動就照實回報「未實跑,缺服務」)。
4. 同一錯誤同一改法 2 次未解 → git checkout 回滾、換方法;
   第 2 種方法再失敗 2 次 → 停手回報(試了什麼/錯誤原文一行/兩條備選路線)。
   本條為全域失敗處理在本專案的具體版,以本條為準。
5. 絕對禁止:為過驗證而改驗證或煙霧腳本、catch 吞例外、留 TODO 假裝完成、
   任務未指明時動到另一版(現行 vs beta)的檔案。
6. 完成定義 = 第 3 條驗證全過且附證據 + git diff 只含任務範圍;缺一不得宣稱完成。
