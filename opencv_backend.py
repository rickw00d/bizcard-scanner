import os
import cv2
import numpy as np
import base64
import io

# Disable PIR and OneDNN for compatibility with certain environments/versions.
# 這組必須在 import paddle/paddleocr 之前設定才有效。
# 症狀：reader 建得起來，但 reader.ocr() 推論時丟
#   "(Unimplemented) ConvertPirAttribute2RuntimeAttribute ... onednn_instruction.cc"
# 就是 oneDNN(MKLDNN)+新版 PIR 執行器的相容性 bug，全部關掉即可。
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_enable_pir_in_executor'] = '0'
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'

# torch is only needed by some PaddleOCR backends; don't let a missing
# install crash the server (detection/cropping don't need it).
try:
    import torch  # noqa: F401
except Exception as _torch_err:
    print(f"Note: torch not available ({_torch_err}); continuing without it.")

# --- DUMMY MODULE INJECTION TO FIX PADDLEX IMPORT ERROR ---
import sys
from types import ModuleType
class DummyDocument:
    pass
dummy_module = ModuleType("langchain.docstore.document")
dummy_module.Document = DummyDocument
sys.modules["langchain.docstore.document"] = dummy_module
sys.modules["langchain.docstore"] = ModuleType("langchain.docstore")
# ----------------------------------------------------------

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────────────────────────
#  Lazy, fault-tolerant PaddleOCR loader
#
#  IMPORTANT: PaddleOCR is NOT initialized at import time anymore.
#  Previously a failed model download / unavailable model version
#  raised at import and crashed the WHOLE Flask server, which took
#  down the /api/detect-card and /api/morph-card endpoints too.
#  Those endpoints are pure OpenCV and don't need OCR at all, yet
#  auto-crop would silently fail ("完全沒反應") because the server
#  never came up.
#
#  Now OCR is loaded lazily on first /api/ocr call and any failure
#  is contained, so card detection / cropping always works.
# ──────────────────────────────────────────────────────────────
_reader = None
_reader_error = None


def get_reader():
    """Return a cached PaddleOCR instance, or None if it can't load."""
    global _reader, _reader_error
    if _reader is not None:
        return _reader
    if _reader_error is not None:
        # Already tried and failed; don't retry on every request.
        return None
    # 離線環境下，paddlex 會嘗試「連線檢查模型來源」而卡住/丟錯。
    # 模型已快取在本機 (~/.paddlex)，直接跳過此檢查。
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')

    # 重點：'chinese_cht' 繁中只有 PP-OCRv3 有模型，PP-OCRv4 會丟
    # "No models are available for the language 'chinese_cht'..."。
    # 依序嘗試多組參數，任一成功即採用。
    # 重點：每一組都關 mkldnn，避免退到某組時 oneDNN 又被打開，
    # 導致 reader 建得起來、但 reader.ocr() 推論時丟 onednn 相容性錯誤。
    attempts = [
        dict(use_angle_cls=True,  lang='chinese_cht', ocr_version='PP-OCRv3', enable_mkldnn=False),
        dict(lang='chinese_cht', ocr_version='PP-OCRv3', enable_mkldnn=False),
        dict(use_textline_orientation=True, lang='chinese_cht', ocr_version='PP-OCRv3', enable_mkldnn=False),
        dict(lang='chinese_cht', enable_mkldnn=False),  # 讓 paddleocr 自選可用版本
        dict(lang='ch', enable_mkldnn=False),           # 退回簡中(也能辨識多數繁中字)
        # 最後保險：完全不帶 mkldnn 參數，靠上面的環境變數全域關閉 oneDNN
        dict(lang='chinese_cht', ocr_version='PP-OCRv3'),
    ]
    try:
        from paddleocr import PaddleOCR
    except Exception as e:
        _reader_error = f"無法匯入 paddleocr：{e}"
        print(f"WARNING: {_reader_error}")
        return None

    errors = []
    dummy = np.zeros((48, 160, 3), dtype=np.uint8)  # 煙霧測試用小圖
    for i, kw in enumerate(attempts, 1):
        try:
            print(f"Initializing PaddleOCR reader (attempt {i}): {kw}")
            cand = PaddleOCR(**kw)
        except Exception as e:
            errors.append(f"[{i}] 建構失敗 {kw}: {e}")
            print(f"  attempt {i} construct failed: {e}")
            continue
        # 煙霧測試：實際推論一次，確認不是「建得起來但一跑就爆 oneDNN」。
        try:
            ocr_call_compat(cand, dummy)
        except Exception as e:
            errors.append(f"[{i}] 推論失敗 {kw}: {e}")
            print(f"  attempt {i} inference failed: {e}")
            continue
        print(f"PaddleOCR ready (attempt {i}).")
        _reader = cand
        _reader_error = None
        return _reader

    _reader_error = " | ".join(errors)
    print("WARNING: PaddleOCR 全部初始化嘗試失敗；裁切仍可用，OCR 不可用。")
    return None


def ocr_call_compat(reader, img):
    """相容呼叫 PaddleOCR：新舊版 .ocr()/.predict() 介面不一，逐一嘗試。"""
    last = None
    for call in (
        lambda: reader.ocr(img),
        lambda: reader.ocr(img, cls=True),
        lambda: reader.predict(img),
    ):
        try:
            return call()
        except TypeError as e:
            last = e
            continue
    if last is not None:
        raise last
    return None

def _corners_to_pct(rect, w, h):
    """rect is an ordered (4,2) array [tl, tr, br, bl] in pixels."""
    return {
        "tl": {"x": float(rect[0][0] / w * 100), "y": float(rect[0][1] / h * 100)},
        "tr": {"x": float(rect[1][0] / w * 100), "y": float(rect[1][1] / h * 100)},
        "br": {"x": float(rect[2][0] / w * 100), "y": float(rect[2][1] / h * 100)},
        "bl": {"x": float(rect[3][0] / w * 100), "y": float(rect[3][1] / h * 100)},
    }


def _aspect_ok(rect):
    """Loose card-aspect gate. Accepts horizontal or vertical orientation."""
    (tl, tr, br, bl) = rect
    widthA = np.hypot(br[0] - bl[0], br[1] - bl[1])
    widthB = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    heightA = np.hypot(tr[0] - br[0], tr[1] - br[1])
    heightB = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
    avgW = (widthA + widthB) / 2.0
    avgH = (heightA + heightB) / 2.0
    if avgW == 0 or avgH == 0:
        return False
    aspect = avgW / avgH
    # Real business cards cluster near 1.6, but allow a wide band so we
    # tolerate perspective skew and odd card sizes.
    return (1.1 < aspect < 2.3) or (0.43 < aspect < 0.91)


def detect_card(image_bytes):
    """Detect the four corners of a business card.

    Strategy (best → fallback) so we return *something* usable far more
    often than the old strict 4-point-only logic:
      1. Clean 4-point quad with a sensible aspect ratio.
      2. Any 4-point quad (relaxed aspect).
      3. minAreaRect of the largest plausible contour (handles rounded
         corners / blurry edges / skew).
      4. None — only when no contour covers enough of the frame.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    orig_h, orig_w = img.shape[:2]

    # Resize for faster/more consistent detection
    target_w = 800
    scale = target_w / float(orig_w)
    target_h = int(orig_h * scale)
    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    h, w = img_resized.shape[:2]
    img_area = float(h * w)

    # Pre-processing
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Method A: Canny edges (auto thresholds from median = more robust
    # across lighting than fixed 50/200).
    v = np.median(blur)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edged = cv2.Canny(blur, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=2)
    edged = cv2.erode(edged, kernel, iterations=1)

    # Method B: Otsu threshold (complements edges on low-contrast cards)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    combined = cv2.bitwise_or(edged, thresh)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(combined.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    MIN_AREA = img_area * 0.08   # was 0.10 — relax slightly
    best_quad = None             # tier-2 result (4 points, any aspect)
    best_rect = None             # tier-3 result (minAreaRect)
    best_rect_area = 0.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = order_points(pts.astype("float32"))
            # Tier 1: clean quad with good aspect → return immediately.
            if _aspect_ok(rect):
                return _corners_to_pct(rect, w, h)
            # Tier 2: remember first 4-point quad even if aspect is odd.
            if best_quad is None:
                best_quad = rect

        # Tier 3 candidate: rotated bounding box of this contour.
        box = cv2.boxPoints(cv2.minAreaRect(c))
        rect_b = order_points(box.astype("float32"))
        ba = cv2.contourArea(box.astype("float32"))
        if _aspect_ok(rect_b) and ba > best_rect_area:
            best_rect, best_rect_area = rect_b, ba

    if best_quad is not None:
        return _corners_to_pct(best_quad, w, h)
    if best_rect is not None:
        return _corners_to_pct(best_rect, w, h)

    return None

def order_points(pts):
    # pts is (4, 2) numpy array
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

@app.route('/api/morph-card', methods=['POST'])
def morph_card_api():
    data = request.json
    if not data or 'image' not in data or 'corners' not in data:
        return jsonify({"error": "Missing image or corners"}), 400
    
    try:
        img_data = base64.b64decode(data['image'])
        corners = data['corners']
        
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image"}), 400
            
        h, w = img.shape[:2]
        
        # Convert % corners to pixel values
        src_pts = np.array([
            [corners['tl']['x'] * w / 100, corners['tl']['y'] * h / 100],
            [corners['tr']['x'] * w / 100, corners['tr']['y'] * h / 100],
            [corners['br']['x'] * w / 100, corners['br']['y'] * h / 100],
            [corners['bl']['x'] * w / 100, corners['bl']['y'] * h / 100]
        ], dtype="float32")
        
        # Determine output dimensions (Standard business card ratio ~1.6 - 1.7)
        widthA = np.sqrt(((src_pts[2][0] - src_pts[3][0]) ** 2) + ((src_pts[2][1] - src_pts[3][1]) ** 2))
        widthB = np.sqrt(((src_pts[1][0] - src_pts[0][0]) ** 2) + ((src_pts[1][1] - src_pts[0][1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((src_pts[1][0] - src_pts[2][0]) ** 2) + ((src_pts[1][1] - src_pts[2][1]) ** 2))
        heightB = np.sqrt(((src_pts[0][0] - src_pts[3][0]) ** 2) + ((src_pts[0][1] - src_pts[3][1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Prevent zero dimensions
        maxWidth = max(100, maxWidth)
        maxHeight = max(100, maxHeight)

        dst_pts = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        
        # Target size compression：調大以保留較清晰的名片影像
        # 主圖最大邊 1200px、base64 上限約 300KB
        TARGET_MAX_DIM = 1200
        wh, ww = warped.shape[:2]
        if max(wh, ww) > TARGET_MAX_DIM:
            ratio = TARGET_MAX_DIM / float(max(wh, ww))
            warped = cv2.resize(warped, (int(ww * ratio), int(wh * ratio)), interpolation=cv2.INTER_AREA)

        # Iterative encoding to hit target size if needed, starting at 90 quality
        quality = 90
        while quality > 20:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buffer = cv2.imencode('.jpg', warped, encode_param)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            if len(b64_str) < 300 * 1024: # 300KB
                break
            quality -= 10
            
        return jsonify({"image": b64_str})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/detect-card', methods=['POST'])
def detect_card_api():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data"}), 400
    
    try:
        img_data = base64.b64decode(data['image'])
        corners = detect_card(img_data)
        if corners:
            return jsonify({"corners": corners})
        else:
            return jsonify({"corners": None, "message": "No card detected"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ocr', methods=['POST'])
def ocr_api():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data"}), 400
    
    reader = get_reader()
    if reader is None:
        return jsonify({
            "error": "OCR engine unavailable",
            "detail": _reader_error or "PaddleOCR could not be initialized.",
            "text": "",
            "raw": []
        }), 503

    try:
        img_data = base64.b64decode(data['image'])

        # PaddleOCR expects a numpy array or a file path
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 解碼失敗（格式不支援/資料毀損）→ 明確回報，不要讓 img.shape 崩潰成 500
        if img is None:
            return jsonify({
                "error": "圖片解碼失敗：cv2 無法讀取此影像（格式不支援或資料毀損）",
                "text": "", "raw": []
            }), 400

        # 灰階/帶透明通道的圖轉成 3 通道 BGR，避免部分版本 OCR 報錯
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Performance Optimization: Resize for faster OCR while maintaining accuracy
        h, w = img.shape[:2]
        MAX_DIM = 800
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"DEBUG: Resized image from {w}x{h} to {new_w}x{new_h}")

        # results is usually a list of results (one per page)
        # PaddleOCR 3.x 的 .ocr() 介面與舊版略有差異；用相容呼叫。
        ocr_result = ocr_call_compat(reader, img)
        print(f"DEBUG: ocr_result type={type(ocr_result)}")
        
        full_text_parts = []
        serializable_results = []
        
        if ocr_result and isinstance(ocr_result, list):
            for page_idx, page in enumerate(ocr_result):
                print(f"DEBUG: Processing page {page_idx}, type={type(page)}")
                if not page:
                    continue
                
                if isinstance(page, dict):
                    # PP-OCRv5 / paddlex style dictionary
                    texts = page.get('rec_texts', [])
                    scores = page.get('rec_scores', [])
                    polys = page.get('rec_polys', []) or page.get('dt_polys', [])
                    
                    for i in range(len(texts)):
                        text = texts[i]
                        conf = float(scores[i]) if i < len(scores) else 1.0
                        bbox = polys[i] if i < len(polys) else [[0, 0], [0, 0], [0, 0], [0, 0]]
                        
                        full_text_parts.append(text)
                        bbox_list = [[float(p[0]), float(p[1])] for p in bbox]
                        serializable_results.append([bbox_list, str(text), float(conf)])
                
                elif isinstance(page, list):
                    # Standard PaddleOCR format: [ [bbox, (text, conf)], ... ]
                    for item_idx, item in enumerate(page):
                        try:
                            if isinstance(item, (list, tuple)) and len(item) == 2:
                                bbox = item[0]
                                text_data = item[1]
                                if isinstance(text_data, (list, tuple)) and len(text_data) >= 2:
                                    text = text_data[0]
                                    conf = text_data[1]
                                else:
                                    text = str(text_data)
                                    conf = 1.0
                            else:
                                continue
                            
                            full_text_parts.append(text)
                            bbox_list = [[float(p[0]), float(p[1])] for p in bbox]
                            serializable_results.append([bbox_list, str(text), float(conf)])
                        except Exception as item_err:
                            print(f"DEBUG: Error parsing item {item_err}: {item_err}")
                            
        full_text = "\n".join(full_text_parts)
        print(f"DEBUG: Extracted text final length: {len(full_text)}")
        print(f"DEBUG: Extracted text: {full_text[:50]}...")
        
        return jsonify({
            "text": full_text,
            "raw": serializable_results
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_api():
    """Lightweight liveness check so the frontend can tell whether the
    backend is up before blaming detection. Does NOT require OCR."""
    return jsonify({
        "status": "ok",
        "detect": True,        # OpenCV detection always available
        "ocr_ready": _reader is not None,
        "ocr_error": _reader_error
    })


if __name__ == '__main__':
    import threading

    def warmup_task():
        # Warm up OCR in the background. If it fails, the server keeps
        # running and detection/cropping stay fully functional.
        print("Warming up PaddleOCR (background)...")
        reader = get_reader()
        if reader is None:
            print("Skipping warm-up: OCR unavailable (detection still works).")
            return
        try:
            if os.path.exists('test_card.png'):
                dummy = cv2.imread('test_card.png')
            else:
                dummy = np.ones((400, 600, 3), dtype=np.uint8) * 255
            reader.ocr(dummy)
            print("PaddleOCR warm-up complete.")
        except Exception as e:
            print(f"PaddleOCR warm-up failed: {e}")

    threading.Thread(target=warmup_task, daemon=True).start()

    print("OpenCV Backend starting on http://localhost:5050")
    print("  - /api/health      (GET)  liveness + ocr status")
    print("  - /api/detect-card (POST) auto-crop corner detection")
    print("  - /api/morph-card  (POST) perspective crop")
    print("  - /api/ocr         (POST) text recognition")
    app.run(port=5050, debug=False)
