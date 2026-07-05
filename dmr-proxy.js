// ═══════════════════════════════════════════════════════
//  CORS Proxy — Bridges browser requests to Ollama
//  Adds CORS headers so scanner.html can access localhost:11434
//  Usage: node dmr-proxy.js
//  Proxy listens on :12435, forwards to Ollama :11434
// ═══════════════════════════════════════════════════════

const http = require('http');

const DMR_HOST = '127.0.0.1';
const DMR_PORT = 11434;  // Ollama native port
const PROXY_PORT = 12435;

const server = http.createServer((req, res) => {
    // --- Handle CORS preflight ---
    if (req.method === 'OPTIONS') {
        res.writeHead(204, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
            'Access-Control-Max-Age': '86400',
        });
        res.end();
        return;
    }

    // --- Forward request to Docker Model Runner ---
    const options = {
        hostname: DMR_HOST,
        port: DMR_PORT,
        path: req.url,
        method: req.method,
        headers: { ...req.headers, host: `${DMR_HOST}:${DMR_PORT}` },
    };

    const proxyReq = http.request(options, (proxyRes) => {
        // Add CORS headers to the response
        const headers = {
            ...proxyRes.headers,
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization',
        };
        res.writeHead(proxyRes.statusCode, headers);
        proxyRes.pipe(res);
    });

    proxyReq.on('error', (err) => {
        console.error(`Proxy error: ${err.message}`);
        res.writeHead(502, {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
        });
        res.end(JSON.stringify({ error: 'Ollama unreachable', detail: err.message }));
    });

    req.pipe(proxyReq);
});

server.listen(PROXY_PORT, () => {
    console.log(`✅ CORS Proxy running on http://localhost:${PROXY_PORT}`);
    console.log(`   Forwarding to Ollama at http://${DMR_HOST}:${DMR_PORT}`);
    console.log(`   Press Ctrl+C to stop\n`);
});
