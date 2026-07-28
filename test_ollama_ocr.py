import urllib.request
import json
import base64

def test_ollama():
    try:
        with open('test_card.png', 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')

        prompt = """You are a data extraction assistant. Extract all contact information from the following business card image.
Return ONLY a valid JSON object with exactly these keys (use empty string "" if not found or unsure):
{ "familyNameCn": "", "givenNameCn": "", "givenNameEn": "", "familyNameEn": "", "company": "", "department": "", "title": "", "taxId": "", "workPhone": "", "ext": "", "mobile": "", "email": "", "website": "", "addressCn": "", "postalCodeCn": "", "addressEn": "" }"""

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
