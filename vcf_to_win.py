# -*- coding: utf-8 -*-
"""
vcf_to_win.py — 批次把 vCard 轉成「Windows 聯絡人相容」格式

問題：scanner 匯出的 .vcf 是 UTF-8 無 BOM 的 vCard 3.0。Windows 聯絡人
      在沒有 BOM 時會用系統 ANSI 代碼頁(繁中=Big5/CP950)解讀 → 中文變亂碼。

解法：轉成 vCard 2.1 + Quoted-Printable(CHARSET=UTF-8)。
      中文以 =XX 編碼、整個檔案變純 ASCII，任何軟體都不會判錯字碼；
      照片改用 vCard 2.1 的 ENCODING=BASE64 區塊保留。

用法：
    python vcf_to_win.py [來源資料夾] [輸出資料夾]

    省略參數時：
      來源 = C:\\Users\\rickwu\\Documents\\My Biz Card
      輸出 = 來源\\win_compat
"""

import os
import sys

DEFAULT_SRC = r"C:\Users\rickwu\Documents\My Biz Card"

TEXT_KEEP_ASCII = True  # 純 ASCII 值(電話/email/url)不做 QP，保持易讀


# ── 讀檔（自動判斷編碼）────────────────────────────────
def read_text(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    for enc in ("utf-8-sig", "utf-8", "big5", "cp950"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="replace")


# ── vCard 摺行還原 + 屬性切分 ───────────────────────────
def unfold(text):
    raw = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    logical = []
    for line in raw:
        if line and line[0] in " \t" and logical:
            logical[-1] += line[1:]
        else:
            logical.append(line)
    return logical


def split_prop(line):
    idx = line.find(":")
    if idx < 0:
        return None
    head, value = line[:idx], line[idx + 1:]
    parts = head.split(";")
    name = parts[0].upper()
    params = parts[1:]
    return name, params, value


# ── Quoted-Printable 編碼 (vCard 2.1) ──────────────────
def qp_encode(s, soft_wrap=75):
    """把字串以 UTF-8 → Quoted-Printable。非 ASCII 變 =XX，過長以 =\\r\\n 軟斷行。"""
    out = []
    line_len = 0
    for byte in s.encode("utf-8"):
        if byte == 0x3D:                 # '='
            tok = "=3D"
        elif byte in (0x0D, 0x0A):        # 換行 → 編碼
            tok = "=0D" if byte == 0x0D else "=0A"
        elif 33 <= byte <= 126:           # 可列印 ASCII 原樣保留（含 ; , 分隔符）
            tok = chr(byte)
        elif byte == 0x20:                # 空白保留
            tok = " "
        else:                             # 其餘(含所有中文位元組) → =XX
            tok = "=%02X" % byte
        # 軟斷行：不切斷 =XX 這種 token
        if line_len + len(tok) > soft_wrap:
            out.append("=\r\n")
            line_len = 0
        out.append(tok)
        line_len += len(tok)
    return "".join(out)


def is_ascii(s):
    try:
        s.encode("ascii")
        return True
    except Exception:
        return False


def strip_params(params, drop=("CHARSET", "ENCODING")):
    """移除舊的 CHARSET/ENCODING 參數，保留 TYPE 等其他參數。"""
    kept = []
    for p in params:
        key = p.split("=", 1)[0].upper()
        if key in drop:
            continue
        kept.append(p)
    return kept


# ── 照片：輸出 vCard 2.1 BASE64 區塊 ───────────────────
def photo_block(name, b64, type_hint="JPEG", label=""):
    lines = []
    label_p = f";X-ABLABEL={label}" if label else ""
    lines.append(f"{name};ENCODING=BASE64;TYPE={type_hint}{label_p}:")
    # 每 74 字一行、前面加一個空白（2.1 摺行）
    for i in range(0, len(b64), 74):
        lines.append(" " + b64[i:i + 74])
    lines.append("")   # 2.1 規定 BASE64 屬性以空白行結束
    return lines


# ── 單檔轉換 ────────────────────────────────────────────
def convert_one(text):
    out = ["BEGIN:VCARD", "VERSION:2.1"]
    for line in unfold(text):
        if not line.strip():
            continue
        pr = split_prop(line)
        if not pr:
            continue
        name, params, value = pr
        up = name.upper()
        if up in ("BEGIN", "END", "VERSION"):
            continue
        if up in ("PHOTO", "LOGO"):
            b64 = value.strip()
            if not b64:
                continue
            type_hint = "PNG" if any("PNG" in p.upper() for p in params) else "JPEG"
            label = ""
            for p in params:
                if p.upper().startswith("X-ABLABEL="):
                    label = p.split("=", 1)[1]
            out += photo_block(up, b64, type_hint, label)
            continue

        base_params = strip_params(params)
        if TEXT_KEEP_ASCII and is_ascii(value):
            # 純 ASCII(電話/email/url…)：不編碼，保持可讀
            head = ";".join([name] + base_params)
            out.append(f"{head}:{value}")
        else:
            # 含中文：CHARSET=UTF-8 + QUOTED-PRINTABLE
            head = ";".join([name] + base_params + ["CHARSET=UTF-8", "ENCODING=QUOTED-PRINTABLE"])
            out.append(f"{head}:{qp_encode(value)}")

    out.append("END:VCARD")
    return "\r\n".join(out) + "\r\n"


# ── 批次處理 ────────────────────────────────────────────
def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(src, "win_compat")

    if not os.path.isdir(src):
        print(f"[錯誤] 找不到來源資料夾：{src}")
        sys.exit(1)
    os.makedirs(dst, exist_ok=True)

    vcfs = [f for f in sorted(os.listdir(src)) if f.lower().endswith(".vcf")]
    if not vcfs:
        print(f"[提醒] {src} 內沒有 .vcf 檔")
        return

    print(f"來源：{src}")
    print(f"輸出：{dst}")
    print(f"共 {len(vcfs)} 個 vcf，開始轉檔…\n")

    ok, fail = 0, 0
    for name in vcfs:
        sp = os.path.join(src, name)
        dp = os.path.join(dst, name)
        try:
            new_text = convert_one(read_text(sp))
            # 純 ASCII 內容，用 latin-1 寫出確保不再帶任何多位元組
            with open(dp, "w", encoding="ascii", newline="") as fh:
                fh.write(new_text)
            ok += 1
            print(f"  ✔ {name}")
        except Exception as e:
            fail += 1
            print(f"  x {name} 失敗：{e}")

    print(f"\n完成：成功 {ok}、失敗 {fail}")
    print(f"轉好的檔在：{dst}")
    print("可直接在 Windows 聯絡人匯入測試，中文應正常顯示。")


if __name__ == "__main__":
    main()
