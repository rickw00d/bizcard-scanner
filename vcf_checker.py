# -*- coding: utf-8 -*-
"""
vcf_checker.py — 名片 VCF 檢查 / 審查後端 (Port 5051)

流程：
  1. 列出來源資料夾內的 .vcf 檔
  2. 解析每個 vcf 的欄位 + 取出嵌入的名片照片
  3. 把照片送到既有 OpenCV/PaddleOCR 後端 (localhost:5050/api/ocr) 重新辨識
  4. 用「模糊比對」把 vcf 每個欄位 與 OCR 重新辨識結果 左右並列
  5. 使用者在 vcf-checker.html 逐欄選擇採用哪一邊 → 寫回 vcf → 搬到「通過」資料夾

OCR 委派給 5050，避免重複載入 PaddleOCR。請先啟動 opencv_backend.py。
"""

import os
import re
import io
import json
import base64
import shutil
import difflib
import datetime
import urllib.request
import urllib.error

from flask import Flask, request, jsonify
from flask_cors import CORS

# 重用批次轉檔邏輯（vcf_to_win.py 無 flask 相依，import 安全）
try:
    import vcf_to_win
except Exception as _e:
    vcf_to_win = None
    print(f"Note: 無法載入 vcf_to_win（整批轉檔功能停用）：{_e}")

app = Flask(__name__)
CORS(app)

# ── 設定 ───────────────────────────────────────────────
CV_BASE = os.environ.get("CV_BASE", "http://localhost:5050")
OCR_ENDPOINT = os.environ.get("OCR_ENDPOINT", CV_BASE + "/api/ocr")
DETECT_ENDPOINT = os.environ.get("DETECT_ENDPOINT", CV_BASE + "/api/detect-card")
MORPH_ENDPOINT = os.environ.get("MORPH_ENDPOINT", CV_BASE + "/api/morph-card")
FUZZY_PASS = 80          # 模糊比對相似度門檻（0~100），達標即視為「一致」
PORT = int(os.environ.get("VCF_CHECKER_PORT", "5051"))

# 嘗試使用 rapidfuzz（較準），沒有就退回 difflib
try:
    from rapidfuzz import fuzz as _rf
    def _ratio(a, b):
        return float(_rf.ratio(a, b))
except Exception:
    def _ratio(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


# ── 文字正規化 / 模糊比對 ──────────────────────────────
def _to_halfwidth(s: str) -> str:
    out = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000:
            code = 0x20
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        out.append(chr(code))
    return "".join(out)


def normalize(s: str) -> str:
    if not s:
        return ""
    s = _to_halfwidth(s)
    s = s.lower()
    # 去掉空白與常見分隔符號，比對時更穩定
    s = re.sub(r"[\s　\-_.,;:/\\|()\[\]<>]", "", s)
    return s.strip()


def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    if not na and not nb:
        return 100.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 100.0
    # 子字串包含視為高度相似（OCR 常多/少字）
    if na in nb or nb in na:
        return max(90.0, _ratio(na, nb))
    return _ratio(na, nb)


# ── VCF 解析 ────────────────────────────────────────────
def unfold(text: str):
    """還原 vCard 摺行：以空白或 tab 開頭的行接續上一行。"""
    raw = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    logical = []
    for line in raw:
        if line and (line[0] == " " or line[0] == "\t") and logical:
            logical[-1] += line[1:]
        else:
            logical.append(line)
    return logical


def split_prop(line: str):
    """切出 (name, params_dict, raw_params_str, value)。"""
    idx = line.find(":")
    if idx < 0:
        return None
    head, value = line[:idx], line[idx + 1:]
    parts = head.split(";")
    name = parts[0].upper()
    params = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            params[k.upper()] = v.upper()
        else:
            params[p.upper()] = True
    return name, params, ";".join(parts[1:]), value


def parse_vcf(text: str):
    """回傳 dict：欄位值 + 照片清單 + 原始結構。"""
    props = []
    photos = []  # list of {b64, mime, label}
    for line in unfold(text):
        if not line.strip():
            continue
        parsed = split_prop(line)
        if not parsed:
            continue
        name, params, raw_params, value = parsed
        if name in ("PHOTO", "LOGO"):
            b64 = value.strip()
            if b64:
                mime = "image/jpeg"
                if "PNG" in raw_params.upper():
                    mime = "image/png"
                label = params.get("X-ABLABEL", "")
                photos.append({"b64": b64, "mime": mime, "label": label, "name": name})
            continue
        props.append({"name": name, "params": params, "raw_params": raw_params, "value": value})

    def first(nm, pred=None):
        for p in props:
            if p["name"] == nm and (pred is None or pred(p)):
                return p["value"]
        return ""

    # 姓名
    fn = first("FN")
    n_raw = first("N")
    name_val = fn or n_raw.replace(";", "").strip()

    # 公司 / 部門
    org = first("ORG")
    org_parts = org.split(";") if org else []
    company = org_parts[0].strip() if len(org_parts) >= 1 else ""
    department = org_parts[1].strip() if len(org_parts) >= 2 else ""

    title = first("TITLE")

    def is_cell(p):
        t = p["raw_params"].upper()
        return "CELL" in t or "MOBILE" in t
    def is_work(p):
        t = p["raw_params"].upper()
        return "WORK" in t and "CELL" not in t

    mobile = first("TEL", is_cell)
    work_phone = first("TEL", is_work)
    if not mobile and not work_phone:
        mobile = first("TEL")  # 至少抓一支

    email = first("EMAIL")
    website = first("URL")

    adr = first("ADR")
    address = ""
    zipcode = ""
    if adr:
        a = adr.split(";")
        # ADR: ;;Street;City;State;ZIP;Country
        address = (a[2] if len(a) > 2 else "").strip()
        zipcode = (a[5] if len(a) > 5 else "").strip()

    note = first("NOTE")
    taxid = ""
    m = re.search(r"統編[:：]?\s*([0-9]{8})", note)
    if m:
        taxid = m.group(1)

    return {
        "props": props,
        "photos": photos,
        "n_raw": n_raw,
        "zipcode": zipcode,
        "categories": first("CATEGORIES"),
        "fields": {
            "name": name_val,
            "company": company,
            "department": department,
            "title": title,
            "mobile": mobile,
            "work_phone": work_phone,
            "email": email,
            "website": website,
            "address": address,
            "taxid": taxid,
            "note": note,
        },
    }


# 欄位顯示順序與標籤
FIELD_DEFS = [
    ("name", "姓名"),
    ("company", "公司"),
    ("department", "部門"),
    ("title", "職稱"),
    ("mobile", "手機"),
    ("work_phone", "公司電話"),
    ("email", "Email"),
    ("website", "網站"),
    ("address", "地址"),
    ("taxid", "統一編號"),
    ("note", "備註"),
]


# ── 後端委派：OCR / 偵測 / 裁切 ─────────────────────────
def _post(endpoint, payload, timeout=120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # 把後端回傳的真正錯誤內容讀出來（否則只會看到 HTTP 500）
        detail = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(body).get("error") or json.loads(body).get("detail") or body
            except Exception:
                detail = body
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {detail[:500]}")


def run_ocr(b64: str):
    data = _post(OCR_ENDPOINT, {"image": b64})
    return data.get("text", ""), data.get("raw", [])


def auto_crop(b64: str):
    """偵測名片四角並做透視裁切。回傳 (裁切後 b64 或 None, corners 或 None)。"""
    try:
        det = _post(DETECT_ENDPOINT, {"image": b64}, timeout=60)
        corners = det.get("corners")
        if not corners:
            return None, None
        morph = _post(MORPH_ENDPOINT, {"image": b64, "corners": corners}, timeout=60)
        cropped = morph.get("image")
        if not cropped:
            return None, corners
        return cropped, corners
    except Exception:
        return None, None


def pick_best_photo(photos):
    """挑資料量最大的照片（通常解析度最高）做裁切/OCR，回傳 (photo, index)。"""
    best, best_area, best_i = None, -1, -1
    for i, ph in enumerate(photos):
        try:
            raw = base64.b64decode(ph["b64"])
            area = len(raw)  # 以位元組大小近似
            if area > best_area:
                best_area, best, best_i = area, ph, i
        except Exception:
            continue
    return best, best_i


def ocr_candidate(field_key, vcf_value, ocr_text, ocr_lines):
    """在 OCR 結果中找出與該欄位最相近的候選字串 + 相似度。"""
    if field_key in ("mobile", "work_phone"):
        # 電話：逐行抓數字串（不跨行），找與 vcf 數字最相近者
        cands = []
        for ln in ocr_lines:
            cands += re.findall(r"[0-9][0-9\-　 \t()]{5,}[0-9]", ln)
        target = digits_only(vcf_value)
        best, best_sc = "", 0.0
        for c in cands:
            dc = digits_only(c)
            if target and dc == target:
                sc = 100.0
            elif target and dc and len(dc) >= 7 and (dc in target or target in dc):
                sc = 92.0   # 同號碼、僅分機/前綴差異
            else:
                sc = similarity(c, vcf_value)
            if sc > best_sc:
                best_sc, best = sc, c.strip()
        return best, best_sc
    if field_key == "email":
        cands = re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+", ocr_text)
        best, best_sc = "", 0.0
        for c in cands:
            sc = similarity(c, vcf_value)
            if sc > best_sc:
                best_sc, best = sc, c
        return best, best_sc
    if field_key == "taxid":
        m = re.search(r"\b([0-9]{8})\b", ocr_text)
        if m:
            return m.group(1), (100.0 if m.group(1) == vcf_value else similarity(m.group(1), vcf_value))
        return "", 0.0
    # 一般文字：逐行找最相近
    best, best_sc = "", 0.0
    for ln in ocr_lines:
        sc = similarity(ln, vcf_value)
        if sc > best_sc:
            best_sc, best = sc, ln
    return best, best_sc


# ── 重建 VCF（套用使用者選擇值，保留照片）───────────────
def fold_line(line: str) -> str:
    if len(line) <= 75:
        return line
    out, first = [], True
    while line:
        if first:
            out.append(line[:75]); line = line[75:]; first = False
        else:
            out.append(" " + line[:74]); line = line[74:]
    return "\r\n".join(out)


def rebuild_vcf(parsed, values):
    f = dict(parsed["fields"])
    f.update(values or {})
    zipc = parsed.get("zipcode", "")
    lines = ["BEGIN:VCARD", "VERSION:3.0"]

    # N：沿用原始（若無則用姓名）
    n_raw = parsed.get("n_raw") or f"{f.get('name','')};;;;"
    lines.append(fold_line(f"N;CHARSET=UTF-8:{n_raw}"))
    if f.get("name"):
        lines.append(fold_line(f"FN;CHARSET=UTF-8:{f['name']}"))

    if f.get("company") or f.get("department"):
        org = ";".join([x for x in [f.get("company", ""), f.get("department", "")] if x is not None])
        lines.append(fold_line(f"ORG;CHARSET=UTF-8:{org}"))
    if f.get("title"):
        lines.append(fold_line(f"TITLE;CHARSET=UTF-8:{f['title']}"))
    if f.get("mobile"):
        lines.append(fold_line(f"TEL;TYPE=CELL,VOICE:{f['mobile']}"))
    if f.get("work_phone"):
        lines.append(fold_line(f"TEL;TYPE=WORK,VOICE:{f['work_phone']}"))
    if f.get("email"):
        lines.append(fold_line(f"EMAIL;TYPE=INTERNET,WORK:{f['email']}"))
    if f.get("website"):
        lines.append(fold_line(f"URL;TYPE=WORK:{f['website']}"))
    if f.get("address"):
        lines.append(fold_line(f"ADR;TYPE=WORK;CHARSET=UTF-8:;;{f['address']};;;{zipc};"))

    # NOTE（含統編）
    note = f.get("note", "") or ""
    taxid = f.get("taxid", "") or ""
    if taxid and f"統編" not in note:
        note = (f"統編: {taxid}\n" + note) if note else f"統編: {taxid}"
    if note:
        lines.append(fold_line(f"NOTE;CHARSET=UTF-8:{note}"))
    if parsed.get("categories"):
        lines.append(fold_line(f"CATEGORIES;CHARSET=UTF-8:{parsed['categories']}"))

    # 保留原本照片（LOGO/PHOTO）
    for ph in parsed["photos"]:
        mime = "PNG" if ph["mime"].endswith("png") else "JPEG"
        extra = f";X-ABLABEL={ph['label']}" if ph.get("label") else ""
        lines.append(fold_line(f"{ph['name']};ENCODING=b;TYPE={mime}{extra}:{ph['b64']}"))

    lines.append("END:VCARD")
    return "\r\n".join(lines) + "\r\n"


# ── 工具 ────────────────────────────────────────────────
def safe_dir(path):
    if not path:
        raise ValueError("資料夾路徑為空")
    path = os.path.expanduser(path)
    return path


def read_text(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    for enc in ("utf-8-sig", "utf-8", "big5", "cp950"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


# ── API ─────────────────────────────────────────────────
@app.route("/api/vcf/health", methods=["GET"])
def health():
    backend_ok = False
    ocr_ready = False
    ocr_error = None
    try:
        with urllib.request.urlopen(CV_BASE + "/api/health", timeout=4) as r:
            backend_ok = True
            try:
                d = json.loads(r.read().decode("utf-8"))
                ocr_ready = bool(d.get("ocr_ready"))
                ocr_error = d.get("ocr_error")
            except Exception:
                pass
    except Exception:
        backend_ok = False
    return jsonify({
        "status": "ok",
        "ocr_backend": backend_ok,   # 5050 是否在線（裁切可用）
        "ocr_ready": ocr_ready,      # PaddleOCR 是否載入成功
        "ocr_error": ocr_error,
        "fuzzy_pass": FUZZY_PASS,
    })


@app.route("/api/vcf/list", methods=["POST"])
def list_vcf():
    d = safe_dir((request.json or {}).get("dir", ""))
    if not os.path.isdir(d):
        return jsonify({"error": f"找不到資料夾：{d}"}), 400
    files = []
    for name in sorted(os.listdir(d)):
        if name.lower().endswith(".vcf"):
            fp = os.path.join(d, name)
            files.append({"name": name, "size": os.path.getsize(fp)})
    return jsonify({"dir": d, "count": len(files), "files": files})


@app.route("/api/vcf/inspect", methods=["POST"])
def inspect():
    body = request.json or {}
    d = safe_dir(body.get("dir", ""))
    fname = body.get("file", "")
    fp = os.path.join(d, fname)
    if not os.path.isfile(fp):
        return jsonify({"error": f"找不到檔案：{fname}"}), 400

    parsed = parse_vcf(read_text(fp))

    # 取最佳照片 → 自動裁切 → 用裁切後的圖做 OCR
    ocr_text, ocr_raw = "", []
    photo_orig_url = None      # 原始嵌入照片
    photo_crop_url = None      # 自動裁切後照片
    crop_b64 = None            # 裁切後 base64（供 approve 寫回）
    crop_applied = False
    best_index = -1
    ocr_error = None

    best, best_index = pick_best_photo(parsed["photos"])
    if best:
        photo_orig_url = f"data:{best['mime']};base64,{best['b64']}"
        # 自動裁切
        try:
            cropped, corners = auto_crop(best["b64"])
        except Exception:
            cropped = None
        if cropped:
            crop_b64 = cropped
            crop_applied = True
            photo_crop_url = f"data:image/jpeg;base64,{cropped}"
        # OCR：優先用裁切後的圖（文字更清楚），否則用原圖
        ocr_src = cropped if cropped else best["b64"]
        try:
            ocr_text, ocr_raw = run_ocr(ocr_src)
        except Exception as e:
            ocr_error = f"OCR 失敗：{e}"
    else:
        ocr_error = "此 vcf 沒有可用的嵌入照片"

    ocr_lines = [r[1] for r in ocr_raw] if ocr_raw else [x for x in ocr_text.split("\n") if x.strip()]

    fields = []
    for key, label in FIELD_DEFS:
        vcf_val = parsed["fields"].get(key, "")
        cand, score = ("", 0.0)
        if ocr_text:
            cand, score = ocr_candidate(key, vcf_val, ocr_text, ocr_lines)
        # 兩邊都空 → 視為一致（無此欄位）
        if not vcf_val and not cand:
            status = "empty"
            score = 100.0
        elif score >= FUZZY_PASS:
            status = "match"
        else:
            status = "diff"
        fields.append({
            "key": key, "label": label,
            "vcf": vcf_val, "ocr": cand,
            "score": round(score, 1), "status": status,
        })

    n_diff = sum(1 for x in fields if x["status"] == "diff")
    overall = "pass" if (n_diff == 0 and not ocr_error) else "review"
    return jsonify({
        "file": fname,
        "fields": fields,
        "ocr_text": ocr_text,
        "ocr_error": ocr_error,
        "photo": photo_orig_url,          # 原始照片
        "photo_cropped": photo_crop_url,  # 自動裁切後（無則 null）
        "crop_applied": crop_applied,
        "crop_b64": crop_b64,
        "best_index": best_index,
        "photo_count": len(parsed["photos"]),
        "diff_count": n_diff,
        "overall": overall,
    })


@app.route("/api/vcf/approve", methods=["POST"])
def approve():
    body = request.json or {}
    d = safe_dir(body.get("dir", ""))
    dest = safe_dir(body.get("dest", ""))
    fname = body.get("file", "")
    values = body.get("values", {})       # {field_key: 採用值}
    use_crop = body.get("use_crop", False) # 是否採用裁切後照片
    crop_b64 = body.get("crop_b64", "")    # 裁切後 base64
    best_index = body.get("best_index", -1)
    fp = os.path.join(d, fname)
    if not os.path.isfile(fp):
        return jsonify({"error": f"找不到檔案：{fname}"}), 400
    if not dest:
        return jsonify({"error": "未指定目標資料夾"}), 400
    os.makedirs(dest, exist_ok=True)

    parsed = parse_vcf(read_text(fp))

    # 採用裁切後照片：替換最佳照片，並同步替換與它相同的重複照片（LOGO/PHOTO 那對）
    if use_crop and crop_b64 and 0 <= best_index < len(parsed["photos"]):
        orig_b64 = parsed["photos"][best_index]["b64"]
        for ph in parsed["photos"]:
            if ph["b64"] == orig_b64:
                ph["b64"] = crop_b64
                ph["mime"] = "image/jpeg"

    new_text = rebuild_vcf(parsed, values)

    out_path = os.path.join(dest, fname)
    # 目標已存在 → 加時間戳避免覆蓋
    if os.path.exists(out_path):
        stem, ext = os.path.splitext(fname)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(dest, f"{stem}_{ts}{ext}")

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(new_text)

    # 搬移：寫到目標後刪除原檔
    try:
        os.remove(fp)
    except Exception as e:
        return jsonify({"warning": f"已寫入目標但原檔刪除失敗：{e}", "dest": out_path}), 200

    return jsonify({"status": "moved", "dest": out_path})


@app.route("/api/vcf/convert_all", methods=["POST"])
def convert_all():
    """整批把來源資料夾的 vcf 轉成 Windows 聯絡人相容 (vCard 2.1 + Quoted-Printable)。"""
    if vcf_to_win is None:
        return jsonify({"error": "轉檔模組未載入（vcf_to_win.py 不在同資料夾？）"}), 500
    body = request.json or {}
    d = safe_dir(body.get("dir", ""))
    out = body.get("out", "") or os.path.join(d, "win_compat")
    out = safe_dir(out)
    if not os.path.isdir(d):
        return jsonify({"error": f"找不到資料夾：{d}"}), 400
    os.makedirs(out, exist_ok=True)

    vcfs = [f for f in sorted(os.listdir(d)) if f.lower().endswith(".vcf")]
    ok, fail, errors = 0, 0, []
    for name in vcfs:
        try:
            new_text = vcf_to_win.convert_one(read_text(os.path.join(d, name)))
            with open(os.path.join(out, name), "w", encoding="ascii", newline="") as fh:
                fh.write(new_text)
            ok += 1
        except Exception as e:
            fail += 1
            errors.append(f"{name}: {e}")
    return jsonify({"status": "done", "total": len(vcfs), "ok": ok,
                    "fail": fail, "out": out, "errors": errors[:20]})


@app.route("/api/vcf/recrop_all", methods=["POST"])
def recrop_all():
    """整批重新裁切：對每個 vcf 的嵌入名片照片做去背/透視裁切，寫回照片後輸出。
    注意：只能把「還帶背景、沒裁乾淨」的照片裁正，無法提高原本已壓縮小圖的解析度。"""
    body = request.json or {}
    d = safe_dir(body.get("dir", ""))
    out = body.get("out", "") or os.path.join(d, "recropped")
    out = safe_dir(out)
    if not os.path.isdir(d):
        return jsonify({"error": f"找不到資料夾：{d}"}), 400
    os.makedirs(out, exist_ok=True)

    vcfs = [f for f in sorted(os.listdir(d)) if f.lower().endswith(".vcf")]
    cropped, no_border, no_photo, fail, errors = 0, 0, 0, 0, []
    for name in vcfs:
        try:
            parsed = parse_vcf(read_text(os.path.join(d, name)))
            best, best_index = pick_best_photo(parsed["photos"])
            status_note = "no_photo"
            if best:
                new_b64, corners = auto_crop(best["b64"])
                if new_b64:
                    orig_b64 = parsed["photos"][best_index]["b64"]
                    for ph in parsed["photos"]:
                        if ph["b64"] == orig_b64:
                            ph["b64"] = new_b64
                            ph["mime"] = "image/jpeg"
                    cropped += 1
                    status_note = "cropped"
                else:
                    no_border += 1
                    status_note = "no_border"
            else:
                no_photo += 1
            # 一律輸出（裁到的換新圖，裁不到的維持原圖），保持整批完整
            new_text = rebuild_vcf(parsed, {})
            with open(os.path.join(out, name), "w", encoding="utf-8", newline="") as fh:
                fh.write(new_text)
        except Exception as e:
            fail += 1
            errors.append(f"{name}: {e}")
    return jsonify({"status": "done", "total": len(vcfs), "cropped": cropped,
                    "no_border": no_border, "no_photo": no_photo, "fail": fail,
                    "out": out, "errors": errors[:20]})


@app.route("/api/vcf/make_test_variants", methods=["POST"])
def make_test_variants():
    """對單一 vcf 產生多種格式變體，供實機匯入 Windows 聯絡人比對。
    A = vCard 3.0 + UTF-8 BOM（照片 ENCODING=b）
    B = vCard 2.1 + Quoted-Printable（照片 2.1 BASE64 區塊）
    C = vCard 3.0 + UTF-8 無 BOM（對照組，＝目前會亂碼的格式）"""
    body = request.json or {}
    d = safe_dir(body.get("dir", ""))
    fname = body.get("file", "")
    fp = os.path.join(d, fname)
    if not os.path.isfile(fp):
        return jsonify({"error": f"找不到檔案：{fname}"}), 400
    out = os.path.join(d, "wintest")
    os.makedirs(out, exist_ok=True)
    text = read_text(fp)
    parsed = parse_vcf(text)
    v30 = rebuild_vcf(parsed, {})
    stem, ext = os.path.splitext(fname)
    made = []

    pa = os.path.join(out, f"{stem}_A_30bom{ext}")
    with open(pa, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write(v30)
    made.append(pa)

    if vcf_to_win is not None:
        pb = os.path.join(out, f"{stem}_B_21qp{ext}")
        with open(pb, "w", encoding="ascii", newline="") as fh:
            fh.write(vcf_to_win.convert_one(text))
        made.append(pb)

    pc = os.path.join(out, f"{stem}_C_30nobom{ext}")
    with open(pc, "w", encoding="utf-8", newline="") as fh:
        fh.write(v30)
    made.append(pc)

    return jsonify({"status": "done", "out": out, "files": made})


if __name__ == "__main__":
    print("=" * 50)
    print(" VCF Checker 後端啟動")
    print(f"  - Port      : {PORT}")
    print(f"  - OCR 端點  : {OCR_ENDPOINT}")
    print(f"  - 模糊門檻  : {FUZZY_PASS}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=PORT, debug=False)
