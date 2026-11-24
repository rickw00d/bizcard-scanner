from paddleocr import PaddleOCR
import traceback

print("Testing PP-OCRv4...")
try:
    reader = PaddleOCR(use_angle_cls=True, lang='chinese_cht', ocr_version='PP-OCRv4', enable_mkldnn=True)
    print("PP-OCRv4 Success!")
except Exception as e:
    print("PP-OCRv4 Failed!")
    traceback.print_exc()

print("\nTesting PP-OCRv3...")
try:
    reader = PaddleOCR(use_angle_cls=True, lang='chinese_cht', ocr_version='PP-OCRv3', enable_mkldnn=True)
    print("PP-OCRv3 Success!")
except Exception as e:
    print("PP-OCRv3 Failed!")
    traceback.print_exc()
