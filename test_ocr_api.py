import requests
import base64
import os

def test_ocr():
    url = "http://127.0.0.1:5050/api/ocr"
    image_path = "test_card.png"
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')

    payload = {"image": img_b64}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print("OCR Result:")
            print(result.get("text"))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_ocr()
