import http.server
import socketserver
import urllib.request
import urllib.error
import urllib.parse
import json

PROXY_PORT = 12435
DMR_HOST = '127.0.0.1'   # IPv4 literal: avoids localhost→::1 (IPv6) mismatch
DMR_PORT = 11434  # Ollama native port

# IMPORTANT: ignore any system / corporate HTTP_PROXY settings.
# urllib normally honors HTTP_PROXY/HTTPS_PROXY and does NOT bypass
# localhost the way browsers do, so a corporate proxy would intercept
# our loopback call to Ollama and return "connection refused" even
# though Ollama is running. This opener forces a direct connection.
DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

class CORSProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Access-Control-Max-Age', '86400')
        self.end_headers()

    def proxy_request(self):
        # Route to Docling or Docker Model Runner based on path
        if self.path.startswith('/docling/'):
            target_path = self.path.replace('/docling', '', 1)
            target_url = f"http://127.0.0.1:5001{target_path}"
        else:
            target_url = f"http://{DMR_HOST}:{DMR_PORT}{self.path}"
        
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Prepare headers for target
        headers = {}
        for key, value in self.headers.items():
            if key.lower() not in ['host', 'connection', 'content-length', 'origin', 'referer', 'sec-fetch-mode', 'sec-fetch-dest', 'sec-fetch-site']:
                headers[key] = value

        req = urllib.request.Request(target_url, data=body, headers=headers, method=self.command)

        try:
            with DIRECT_OPENER.open(req, timeout=300) as response:
                self.send_response(response.status)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
                
                # Forward other headers
                for key, value in response.headers.items():
                    if key.lower() not in ['transfer-encoding']: # Avoid issues with chunked encoding mapping
                        self.send_header(key, value)
                self.end_headers()
                
                self.wfile.write(response.read())
                
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.send_header('Access-Control-Allow-Origin', '*')
            for key, value in e.headers.items():
                if key.lower() not in ['transfer-encoding']:
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(e.read())
            
        except urllib.error.URLError as e:
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            print(f"[PROXY] Failed reaching {target_url}: {e.reason}")
            err_msg = json.dumps({
                "error": "Ollama unreachable",
                "target": target_url,
                "detail": str(e.reason),
                "hint": "Ollama 沒在這個位址監聽,或被系統 Proxy 攔截。"
                        "請確認 ollama serve 已啟動,且 OLLAMA_HOST 為 127.0.0.1:11434。"
            }).encode('utf-8')
            self.wfile.write(err_msg)

    def do_GET(self):
        self.proxy_request()

    def do_POST(self):
        self.proxy_request()

    # Disable logging to avoid clutter with polling
    def log_message(self, format, *args):
        pass

with socketserver.ThreadingTCPServer(("", PROXY_PORT), CORSProxyHandler) as httpd:
    print(f"CORS Proxy running on http://localhost:{PROXY_PORT}")
    print(f"   Forwarding to Ollama at http://{DMR_HOST}:{DMR_PORT}")
    print(f"   Press Ctrl+C to stop...\n")
    httpd.serve_forever()
