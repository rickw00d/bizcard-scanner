import requests
import base64
import cv2
import numpy as np
import os

CV_BACKEND_BASE = "http://localhost:5000"

def test_morph_api():
    # 1. Create a dummy test image (white rectangle on black background)
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    # Draw a skewed rectangle to simulate a card
    pts = np.array([[50, 50], [450, 100], [400, 400], [100, 450]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))
    
    _, buffer = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # 2. Preparation for API call
    payload = {
        "image": img_b64,
        "corners": {
            "tl": {"x": 10.0, "y": 10.0},
            "tr": {"x": 90.0, "y": 20.0},
            "br": {"x": 80.0, "y": 80.0},
            "bl": {"x": 20.0, "y": 90.0}
        }
    }
    
    print("Testing /api/morph-card...")
    try:
        res = requests.post(f"{CV_BACKEND_BASE}/api/morph-card", json=payload)
        res.raise_for_status()
        data = res.json()
        
        if "image" in data:
            image_len = len(data["image"])
            print(f"SUCCESS: Received morphed image. Base64 length: {image_len} bytes")
            if image_len < 100 * 1024:
                print(f"VERIFIED: Image size ({image_len/1024:.1f}KB) is under 100KB limit.")
            else:
                print(f"WARNING: Image size ({image_len/1024:.1f}KB) EXCEEDS 100KB limit.")
        else:
            print("FAILED: No image in response.")
            
    except Exception as e:
        print(f"ERROR: API call failed: {e}")

if __name__ == "__main__":
    test_morph_api()
