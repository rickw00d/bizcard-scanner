# 指令與相依細節(business_card)

## 啟動
- 一鍵:`run_scanner.bat`
- 手動分別啟動(各開一個視窗):
  - `ollama serve` — LLM 服務,port 11434
  - `open-webui serve` — 選用,port 8080
  - `python dmr_proxy.py` — CORS 代理,port 12435
  - `python opencv_backend.py` — OpenCV/PaddleOCR 後端,port 5050
  - `python vcf_checker.py` — VCF 審查後端,port 5051
  - `python -m http.server 8000` — 前端靜態伺服器
- 入口網址:
  - http://localhost:8000/scanner.html(現行版)
  - http://localhost:8000/scanner-beta.html(beta)
  - http://localhost:8000/vcf-checker.html

## 驗收/煙霧測試
- 無 pytest 等正式測試框架;驗收暫以手動跑 `run_scanner.bat` 成功為準。
- 後端(5050)啟動後可跑的手動煙霧腳本:
  - `python test_ocr_api.py` — 測 /api/ocr
  - `python test_morph_api.py` — 測 /api/morph-card
- `benchmark_backend.py` / `benchmark_qwen.py`:效能量測用,非驗收。

## 相依套件(無 requirements.txt,由 import 推得;完整性未確認)
- Python:flask、flask-cors、opencv-python、numpy、paddleocr(含 paddle)、requests(測試腳本用)
- 選用:torch(缺少不影響)、rapidfuzz(缺少時退回 difflib)
- 外部程式:Ollama、Open WebUI(選用)

## 環境變數(vcf_checker.py)
- `CV_BASE`(預設 http://localhost:5050)、`VCF_CHECKER_PORT`(預設 5051)
- `OCR_ENDPOINT` / `DETECT_ENDPOINT` / `MORPH_ENDPOINT` 可各自覆寫

## 其他慣例
- `opencv_backend.py` 開頭的 `FLAGS_*` 環境變數必須在 import paddle 之前設定
  (PIR/oneDNN 相容性修正),勿移動位置。
- `vcf_to_win.py` 用法:`python vcf_to_win.py [來源資料夾] [輸出資料夾]`
  (預設來源 `C:\Users\rickwu\Documents\My Biz Card`)
