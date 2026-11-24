const fs = require('fs');

async function testOllama() {
    try {
        const img = fs.readFileSync('test_card.png').toString('base64');
        const prompt = `Detect the four corners of the business card in this image. 
Return the coordinates as a JSON object with exactly these keys: 'tl', 'tr', 'bl', 'br' (Top-Left, Top-Right, Bottom-Left, Bottom-Right).
Each value must be an object with 'x' and 'y' properties representing the coordinates as a percentage (0 to 100) of the image width and height.
Output ONLY valid JSON. Do not add any markdown, comments, or extra text.
Example: {"tl": {"x": 10, "y": 10}, "tr": {"x": 90, "y": 10}, "bl": {"x": 10, "y": 90}, "br": {"x": 90, "y": 90}}`;

        const res = await fetch('http://localhost:11434/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'qwen3-vl:4b',
                messages: [{ role: 'user', content: prompt, images: [img] }],
                stream: false
            })
        });
        const data = await res.json();
        console.log("qwen3-vl:4b Auto-Crop Response:\n", data.message?.content);
    } catch (e) {
        console.error(e);
    }
}
testOllama();
