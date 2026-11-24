import requests
import base64
import time
import os

def benchmark_ocr():
    url = "http://localhost:5000/api/ocr"
    test_image_path = "c:\\Users\\rickwu\\Documents\\My Code\\business_card\\test_card.png"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}")
        return

    with open(test_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    start_time = time.time()
    try:
        response = requests.post(url, json={"image": encoded_string})
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"OCR Success!")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            print(f"Text extracted length: {len(result.get('text', ''))}")
            # print(f"Preview: {result.get('text', '')[:100]}...")
        else:
            print(f"OCR Failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error connecting to backend: {e}")

if __name__ == "__main__":
    benchmark_ocr()
