#!/usr/bin/env bash
# business_card 服務啟動器(Ubuntu 版,對應 Windows 的 run_scanner.bat)
# 用 tmux 開一個名為 "scanner" 的 session,6 個 window 各跑一個服務。
#
#   啟動:  ./run_scanner.sh
#   查看:  tmux attach -t scanner   (Ctrl-b n / p 切換 window,Ctrl-b d 離開但不關閉)
#   停止:  ./run_scanner.sh stop    (或 tmux kill-session -t scanner)
#
# 對應網址:
#   Scanner      http://localhost:8001/scanner.html
#   VCF Checker  http://localhost:8001/vcf-checker.html
#   Open WebUI   http://localhost:8080
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
SESSION="scanner"
# 用相對路徑,避免專案路徑含空白(My Code)在 tmux send-keys 被拆開。
# 每個 window 都以 -c "$DIR" 為工作目錄,相對路徑即可正確解析。
PY=".venv/bin/python"
# 靜態網頁伺服器的埠(scanner.html / vcf-checker.html 從這裡開)。
# 用 8001:避開被佔用的 8000,以及 Open WebUI 的 8080。
HTTP_PORT=8001

# --- 停止模式 ---
if [[ "${1:-}" == "stop" ]]; then
  tmux kill-session -t "$SESSION" 2>/dev/null && echo "已停止 session：$SESSION" || echo "沒有正在執行的 session：$SESSION"
  exit 0
fi

# --- 前置檢查 ---
command -v tmux >/dev/null || { echo "缺少 tmux,請先:sudo apt install tmux"; exit 1; }
[[ -x "$PY" ]] || { echo "找不到 venv:$PY"; echo "請先建立:python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
command -v ollama >/dev/null || echo "警告:找不到 ollama,對應 window 會顯示錯誤(不影響其他服務)"
HAVE_WEBUI=1; command -v open-webui >/dev/null || { HAVE_WEBUI=0; echo "警告:找不到 open-webui,略過該 window(其餘服務照常啟動)"; }

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' 已在執行。先 ./run_scanner.sh stop 再重跑,或 tmux attach -t $SESSION 查看。"
  exit 0
fi

# 讓 window 在指令結束後不自動關閉,方便看錯誤訊息
tmux new-session -d -s "$SESSION" -n Ollama -c "$DIR"
tmux set-option -t "$SESSION" remain-on-exit on

run() {  # run <window-name> <command...>
  local name="$1"; shift
  tmux new-window -t "$SESSION" -n "$name" -c "$DIR"
  tmux send-keys -t "$SESSION:$name" "$*" C-m
}

# window 0(Ollama)已建立,直接送指令
tmux send-keys -t "$SESSION:Ollama" "ollama serve" C-m
[[ $HAVE_WEBUI -eq 1 ]] && run "OpenWebUI" "open-webui serve"
run "DMRProxy" "$PY dmr_proxy.py"
run "OpenCV5050" "$PY opencv_backend.py"
run "VCF5051" "$PY vcf_checker.py"
run "HTTP$HTTP_PORT" "$PY -m http.server $HTTP_PORT"

cat <<EOF

所有服務已在 tmux session「$SESSION」啟動。

請在瀏覽器開啟:
  - Scanner:     http://localhost:$HTTP_PORT/scanner.html
  - VCF Checker: http://localhost:$HTTP_PORT/vcf-checker.html
  - Open WebUI:  http://localhost:8080

查看/切換各服務:  tmux attach -t $SESSION   (Ctrl-b 再按數字切 window,Ctrl-b d 離開)
停止全部:          ./run_scanner.sh stop
EOF
