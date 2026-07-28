import urllib.request
import json
import base64

def test_ollama():
    try:
        with open('test_card.png', 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')

        prompt = """Detect the four corners of the business card in this image. 
Return the coordinates as a JSON object with exactly these keys: 'tl', 'tr', 'bl', 'br' (Top-Left, Top-Right, Bottom-Left, Bottom-Right).
Each value must be an object with 'x' and 'y' properties representing the coordinates as a percentage (0 to 100) of the image width and height.
Output ONLY valid JSON. Do not add any markdown, comments, or extra text.
Example: {"tl": {"x": 10, "y": 10}, "tr": {"x": 90, "y": 10}, "bl": {"x": 10, "y": 90}, "br": {"x": 90, "y": 90}}"""

        data = {
            'model': 'qwen3-vl:4b',
            'messages': [{'role': 'user', 'content': prompt, 'images': [b64]}],
            'stream': False
        }

        req = urllib.request.Request('http://localhost:11434/api/chat', json.dumps(data).encode('utf-8'), {'Content-Type': 'application/json'})
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read().decode('utf-8'))
        print("Model Output:\n", result['message']['content'])
    except Exception as e:
        print("Error:", e)

test_ollama()
