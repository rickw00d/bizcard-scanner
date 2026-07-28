import requests
import base64
import time
import json
import os
import sys

# Constants
OLLAMA_URL = "http://localhost:11434/api/chat"
IMAGE_PATH = r"C:\Users\rickwu\Documents\My Code\business_card\test_card.png"
MODELS = ["qwen3.5:0.8b", "qwen2.5:7b"]
TRIALS = 3

# Ground Truth for test_card.png
GROUND_TRUTH = {
    "name": "Jane Doe",
    "title": "CEO",
    "company": "Tech Solutions Inc.",
    "email": "jane.doe@techsolutions.com",
    "phone": "+1-555-123-4567",
    "address": "123 Innovation Way, Silicon Valley, CA 94000"
}

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def run_ollama(model, prompt, image_b64=None):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64] if image_b64 else []
            }
        ],
        "stream": False
    }
    
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        elapsed = time.time() - start_time
        if response.status_code == 200:
            return response.json().get("message", {}).get("content", ""), elapsed
        else:
            return f"Error: HTTP {response.status_code}", elapsed
    except Exception as e:
        return f"Error: {str(e)}", time.time() - start_time

def calculate_accuracy(text):
    score = 0
    total = len(GROUND_TRUTH)
    found_details = {}
    
    text_lower = text.lower()
    for key, val in GROUND_TRUTH.items():
        if val.lower() in text_lower:
            score += 1
            found_details[key] = True
        else:
            found_details[key] = False
    return (score / total) * 100, found_details

def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return

    print(f"--- Benchmarking Business Card Recognition ---")
    print(f"Image: {IMAGE_PATH}")
    print(f"Models: {MODELS}")
    print(f"Trials per model: {TRIALS}")
    print("-" * 47)

    image_b64 = encode_image(IMAGE_PATH)
    prompt = "Extract all text from this business card image. Mention name, title, company, email, phone, and address."

    final_results = {}

    for model in MODELS:
        print(f"\nTesting Model: {model}")
        times = []
        accuracies = []
        outputs = []

        for i in range(TRIALS):
            print(f"  Trial {i+1}...", end="", flush=True)
            output, elapsed = run_ollama(model, prompt, image_b64)
            accuracy, _ = calculate_accuracy(output)
            
            times.append(elapsed)
            accuracies.append(accuracy)
            outputs.append(output)
            print(f" Done ({elapsed:.2f}s, Accuracy: {accuracy:.1f}%)")

        avg_time = sum(times) / len(times)
        avg_acc = sum(accuracies) / len(accuracies)
        
        final_results[model] = {
            "avg_time": avg_time,
            "avg_acc": avg_acc,
            "last_output": outputs[-1]
        }

    print("\n" + "="*50)
    print(f"{'Model':<15} | {'Avg Speed':<12} | {'Avg Accuracy':<12}")
    print("-" * 50)
    for model, res in final_results.items():
        print(f"{model:<15} | {res['avg_time']:>9.2f}s | {res['avg_acc']:>11.1f}%")
    print("="*50)

    # Recommendation
    print("\nRecommendation:")
    best_acc = -1
    best_model = None
    for model, res in final_results.items():
        if res['avg_acc'] > best_acc:
            best_acc = res['avg_acc']
            best_model = model
        elif res['avg_acc'] == best_acc:
            if best_model and res['avg_time'] < final_results[best_model]['avg_time']:
                best_model = model
    
    if best_model:
        print(f"Best performance model is {best_model} with {final_results[best_model]['avg_acc']:.1f}% accuracy.")

if __name__ == "__main__":
    main()
