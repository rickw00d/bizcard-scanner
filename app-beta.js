// ═══════════════════════════════════════════════════════
//  CardScanner — app-beta.js
//  Camera, Zoom, Flip, Perspective Crop, Docker Model Runner OCR, vCard Export
//  OCR: PaddleOCR (primary) + Docker Model Runner (fallback/parsing)
// ═══════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════
//  Docker Model Runner & PaddleOCR API Helpers
// ═══════════════════════════════════════════════════════
const DMR_BASE = 'http://localhost:12435/v1';  // CORS proxy → Docker Model Runner :12434
const CV_BACKEND_BASE = 'http://localhost:5050';

// Note: Docling expects multipart/form-data
async function doclingFetch(imageDataUrl) {
    try {
        const res = await fetch(imageDataUrl);
        const blob = await res.blob();
        const formData = new FormData();
        formData.append("files", blob, "business_card.jpg");

        const response = await fetch("http://localhost:12435/docling/v1/convert/file", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Docling Error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        let extractedText = "";
        if (data && data.document && data.document.items) {
            extractedText = data.document.items
                .map(item => item.text || '')
                .filter(text => text.trim() !== '')
                .join('\n');
        } else if (data && data.document && data.document.texts) {
             extractedText = data.document.texts.map(t => t.text).join('\n');
        } else {
             extractedText = JSON.stringify(data.document || data, null, 2);
        }
        return extractedText.trim();
    } catch (e) {
        console.error("Docling Fetch Failed:", e);
        return "";
    }
}

async function dmrChat(model, prompt, imageBase64 = null, opts = {}) {
    const messages = [];
    if (opts.systemPrompt) {
        messages.push({ role: 'system', content: opts.systemPrompt });
    }

    // Build OpenAI-compatible message content
    const content = [];
    content.push({ type: 'text', text: prompt });
    if (imageBase64) {
        content.push({
            type: 'image_url',
            image_url: { url: `data:image/jpeg;base64,${imageBase64}` }
        });
    }
    messages.push({ role: 'user', content });

    const payload = {
        model,
        messages,
        stream: false,
        temperature: opts.temperature !== undefined ? opts.temperature : 0.2
    };

    if (opts.format === 'json') {
        payload.response_format = { type: "json_object" };
    }

    const res = await fetch(`${DMR_BASE}/chat/completions`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer local-key'
        },
        body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error(`Docker Model Runner error: ${res.status} ${res.statusText}`);
    const data = await res.json();
    return data.choices?.[0]?.message?.content || '';
}

function imageDataUrlToBase64(dataUrl) {
    return dataUrl.split(',')[1] || '';
}

// XSS protection for dynamic HTML
function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

async function parseTextToJsonViaLLM(rawText) {
    if (!rawText || !rawText.trim() || rawText.startsWith('ERROR:')) return {};

    const parseModel = (textModelSelect && textModelSelect.value) || 'ai/gemma3';

    try {
        const qwenPrompt = `You are a data extraction assistant. Extract all contact information from the following OCR text.

CRITICAL RULES:
1. NEVER translate between languages. English text must stay in English exactly as it appears; do NOT translate English into Chinese, and do NOT translate Chinese into English.
2. Chinese text must be written in Traditional Chinese (繁體中文): if the OCR text contains Simplified Chinese characters, convert them to Traditional (character conversion only, not translation).
   Examples: 财务部→財務部、经理→經理、100号5楼→100號5樓。
3. Put English values in the *En fields and Chinese values in the *Cn fields. If there is no content in that language, leave the field "".
Keep digits, punctuation, emails and URLs unchanged.

Return ONLY a valid JSON object with these fields (use empty string "" if not found or unsure):
    { "familyNameCn": "", "givenNameCn": "", "givenNameEn": "", "familyNameEn": "", "company": "", "department": "", "title": "", "taxId": "", "workPhone": "", "ext": "", "mobile": "", "email": "", "website": "", "addressCn": "", "postalCodeCn": "", "addressEn": "" }

OCR TEXT:
    """
${rawText}
    """`;

        const llmResp = await dmrChat(parseModel, qwenPrompt, null, {
            temperature: 0.2,
            format: 'json'
        });

        let extractedJson = {};
        const jsonMatch = llmResp.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            try {
                let jsonString = jsonMatch[0].replace(/,\s*([}\]])/g, '$1');
                extractedJson = JSON.parse(jsonString);
            } catch (jsonErr) {
                console.warn('JSON parse error, falling back to basic extraction:', jsonErr);
                extractedJson = parseContactData(llmResp);
            }
        } else {
            extractedJson = parseContactData(llmResp);
        }
        return extractedJson;
    } catch (err) {
        console.warn('Text LLM Error during parsing:', err);
        return parseContactData(rawText);
    }
}

async function paddleOcrFetch(imageDataUrl) {
    try {
        console.log('DEBUG: Starting paddleOcrFetch');
        const b64 = imageDataUrlToBase64(imageDataUrl);
        console.log('DEBUG: Image Base64 length:', b64.length);
        
        const res = await fetch(`${CV_BACKEND_BASE}/api/ocr`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: b64 })
        });
        
        if (res.ok) {
            const data = await res.json();
            console.log('DEBUG: OCR Response Success, text length:', (data.text || '').length);
            return data.text || '';
        } else {
            console.error('DEBUG: OCR API error status:', res.status);
            const errText = await res.text();
            console.error('DEBUG: OCR API error body:', errText);
            // 把後端真正的錯誤訊息顯示出來，方便診斷（例如 PaddleOCR 未就緒 / oneDNN 錯誤）
            let detail = errText;
            try {
                const j = JSON.parse(errText);
                detail = j.error || j.detail || errText;
            } catch (e) { /* 非 JSON 就用原文 */ }
            if (res.status === 503) {
                alert('OCR 引擎尚未就緒（PaddleOCR 未載入）。\n請確認 opencv_backend.py 已啟動並完成暖機。\n\n後端訊息：' + detail);
            } else {
                alert('OCR 失敗 (HTTP ' + res.status + ')：\n' + (detail || '').slice(0, 500));
            }
        }
    } catch (err) {
        console.error('DEBUG: PaddleOCR Fetch Error:', err);
        alert('無法連線到本地 OCR 後端 (OpenCV Backend)。請確認 opencv_backend.py 正在執行。');
    }
    return '';
}

// ═══════════════════════════════════════════════════════
//  Image Pre-processing (Resize & Auto Brightness)
// ═══════════════════════════════════════════════════════
function processImageForDMR(dataUrl, doAutoBrightness = false, maxWidth = 1000) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            let width = img.width;
            let height = img.height;
            // Downscale to save processing time
            if (width > maxWidth) {
                height = Math.round((height * maxWidth) / width);
                width = maxWidth;
            }

            const canvas = document.createElement('canvas');
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, width, height);

            if (doAutoBrightness) {
                const imageData = ctx.getImageData(0, 0, width, height);
                const data = imageData.data;
                let sum = 0;
                for (let i = 0; i < data.length; i += 4) {
                    // Simple perceived brightness formula
                    sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                }
                const avgBrightness = sum / (width * height);

                // Target a brightness of 130
                const targetBrightness = 130;
                let offset = 0;
                if (avgBrightness < 100 || avgBrightness > 160) {
                    offset = targetBrightness - avgBrightness;
                }

                if (offset !== 0) {
                    for (let i = 0; i < data.length; i += 4) {
                        data[i] = Math.min(255, Math.max(0, data[i] + offset));
                        data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + offset));
                        data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + offset));
                    }
                    ctx.putImageData(imageData, 0, 0);
                }
            }

            resolve(canvas.toDataURL('image/jpeg', 0.85));
        };
        img.src = dataUrl;
    });
}

// ─── DOM Elements ───
const videoFeed = document.getElementById('videoFeed');
const videoContainer = document.getElementById('videoContainer');
const captureBtn = document.getElementById('captureBtn');
const cameraSelect = document.getElementById('cameraSelect');
const rotate180Btn = document.getElementById('rotate180Btn');
const zoomSlider = document.getElementById('zoomSlider');
const zoomLabel = document.getElementById('zoomLabel');
const zoomInBtn = document.getElementById('zoomInBtn');
const zoomOutBtn = document.getElementById('zoomOutBtn');
const fileInput = document.getElementById('fileInput');
const cameraError = document.getElementById('cameraError');
const captureCanvas = document.getElementById('captureCanvas');
const previewImage = document.getElementById('previewImage');

// Quality Assessment Removed

// Single OCR Status
const singleOcrStatus = document.getElementById('singleOcrStatus');
const activeModelLabel = document.getElementById('activeModelLabel');
const activeModelStatus = document.getElementById('activeModelStatus');
const ollamaModelSelect = document.getElementById('ollamaModelSelect');
const textModelSelect = document.getElementById('textModelSelect');
const ocrEngineSelect = document.getElementById('ocrEngineSelect');
// 主 OCR 引擎選擇的持久化 key(值可為 paddle / docling / ollama:<model>)
const OCR_ENGINE_STORE_KEY = 'betaOcrEngine';
// 文字解析(LLM)模型選擇的持久化 key
const TEXT_MODEL_STORE_KEY = 'betaTextModel';


// Sections
const cameraSection = document.getElementById('cameraSection');
const cropSection = document.getElementById('cropSection');
const processingSection = document.getElementById('processingSection');
const resultSection = document.getElementById('resultSection');
const successSection = document.getElementById('successSection');
const backCameraSection = document.getElementById('backCameraSection');
const backCropSection = document.getElementById('backCropSection');
const backProcessingSection = document.getElementById('backProcessingSection');

// Crop Elements
const cropContainer = document.getElementById('cropContainer');
const cropCanvas = document.getElementById('cropCanvas');
const cropOverlaySvg = document.getElementById('cropOverlaySvg');
const cropPolygon = document.getElementById('cropPolygon');
const cropConfirmBtn = document.getElementById('cropConfirmBtn');
const cropSkipBtn = document.getElementById('cropSkipBtn');
const cropRetakeBtn = document.getElementById('cropRetakeBtn');
const autoCropBtn = document.getElementById('autoCropBtn');

// Progress/Processing
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');

// Form Fields
const familyNameCnField = document.getElementById('familyNameCnField');
const givenNameCnField = document.getElementById('givenNameCnField');
const givenNameEnField = document.getElementById('givenNameEnField');
const familyNameEnField = document.getElementById('familyNameEnField');
const companyField = document.getElementById('companyField');
const departmentField = document.getElementById('departmentField');
const titleField = document.getElementById('titleField');
const taxIdField = document.getElementById('taxIdField');
const workPhoneField = document.getElementById('workPhoneField');
const extField = document.getElementById('extField');
const mobileField = document.getElementById('mobileField');
const emailField = document.getElementById('emailField');
const websiteField = document.getElementById('websiteField');
const prefixHttpBtn = document.getElementById('prefixHttpBtn');
const addressCnField = document.getElementById('addressCnField');
const postalCodeCnField = document.getElementById('postalCodeCnField');
const addressEnField = document.getElementById('addressEnField');
const postalCodeEnField = document.getElementById('postalCodeEnField');
const noteField = document.getElementById('noteField');
const categoriesField = document.getElementById('categoriesField');

// Back Photo
const backPhotoInput = document.getElementById('backPhotoInput');
const captureBackBtn = document.getElementById('captureBackBtn');
const removeBackPhotoBtn = document.getElementById('removeBackPhotoBtn');
const backPhotoPreview = document.getElementById('backPhotoPreview');
const backPreviewImg = document.getElementById('backPreviewImg');
const backVideoFeed = document.getElementById('backVideoFeed');
const backCaptureCanvas = document.getElementById('backCaptureCanvas');
const backCaptureTakeBtn = document.getElementById('backCaptureTakeBtn');
const backCaptureCancelBtn = document.getElementById('backCaptureCancelBtn');
const backRotate180Btn = document.getElementById('backRotate180Btn');
const backZoomSlider = document.getElementById('backZoomSlider');
const backZoomLabel = document.getElementById('backZoomLabel');
const backZoomInBtn = document.getElementById('backZoomInBtn');
const backZoomOutBtn = document.getElementById('backZoomOutBtn');

// Back Crop Elements
const backCropContainer = document.getElementById('backCropContainer');
const backCropCanvas = document.getElementById('backCropCanvas');
const backCropOverlaySvg = document.getElementById('backCropOverlaySvg');
const backCropPolygon = document.getElementById('backCropPolygon');
const backCropConfirmBtn = document.getElementById('backCropConfirmBtn');
const backCropSkipBtn = document.getElementById('backCropSkipBtn');
const backAutoCropBtn = document.getElementById('backAutoCropBtn');


// Front Crop Elements


// Back Processing Elements
const backProgressFill = document.getElementById('backProgressFill');
const backProgressText = document.getElementById('backProgressText');
const backLiveOCRText = document.getElementById('backLiveOCRText');

// Actions
const retakeBtn = document.getElementById('retakeBtn');
const exportBtn = document.getElementById('exportBtn');
const scanAnotherBtn = document.getElementById('scanAnotherBtn');
const contactForm = document.getElementById('contactForm');

// Raw Text & Live Display
const liveOCRText = document.getElementById('liveOCRText');

// Service Status Elements
const serviceStatusBanner = document.getElementById('serviceStatusBanner');
const proxyStatusEl = document.getElementById('proxyStatus');
const cvBackendStatusEl = document.getElementById('cvBackendStatus');
const statusRetryBtn = document.getElementById('statusRetryBtn');

const manualCropBtn = document.getElementById('manualCropBtn');
const manualBackCropBtn = document.getElementById('manualBackCropBtn');

// ─── State ───
let stream = null;
let backStream = null;
let flipH = false;
let flipV = false;
let currentZoom = 1;
let nativeZoomSupported = false;
let backPhotoDataUrl = null;
let isProcessing = false;
let previousSection = null;
let capturedImageDataUrl = null; // Stored for crop → OCR flow
let backCapturedImageDataUrl = null; // Stored for back crop → OCR flow
let backFlipH = false;
let backFlipV = false;
let backCurrentZoom = 1;
let backNativeZoomSupported = false;

// Crop corner positions (in % of container)
let cropCorners = {
    tl: { x: 10, y: 10 },
    tr: { x: 90, y: 10 },
    bl: { x: 10, y: 90 },
    br: { x: 90, y: 90 }
};
let backCropCorners = {
    tl: { x: 10, y: 10 },
    tr: { x: 90, y: 10 },
    bl: { x: 10, y: 90 },
    br: { x: 90, y: 90 }
};
let dragTarget = null;
let dragIsBack = false; // track which crop is being dragged

// ═══════════════════════════════════════════════════════
//  Initialization
// ═══════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', async () => {
    // Listen for camera hardware changes (plug/unplug)
    if (navigator.mediaDevices && navigator.mediaDevices.ondevicechange !== undefined) {
        navigator.mediaDevices.ondevicechange = populateCameraList;
    }
    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
        navigator.mediaDevices.addEventListener('devicechange', populateCameraList);
    }

    // Refresh labels on user interaction (select click)
    cameraSelect.addEventListener('mousedown', populateCameraList);
    cameraSelect.addEventListener('touchstart', populateCameraList);

    // Initial scan
    await populateCameraList();
    initCamera();

    // Fetch available Docker Model Runner models
    await populateModels();

    // 記住主 OCR 引擎與解析模型選擇，重新整理後由 populateModels 恢復
    if (ocrEngineSelect) {
        ocrEngineSelect.addEventListener('change', () => {
            localStorage.setItem(OCR_ENGINE_STORE_KEY, ocrEngineSelect.value);
        });
    }
    if (textModelSelect) {
        textModelSelect.addEventListener('change', () => {
            localStorage.setItem(TEXT_MODEL_STORE_KEY, textModelSelect.value);
        });
    }

    // Check backend service status
    await checkServiceStatus();
    if (statusRetryBtn) {
        statusRetryBtn.addEventListener('click', checkServiceStatus);
    }

    // Event Listeners
    captureBtn.addEventListener('click', captureImage);
    retakeBtn.addEventListener('click', resetToCamera);
    scanAnotherBtn.addEventListener('click', () => {
        resetForm();
        resetToCamera();
    });

    fileInput.addEventListener('change', handleFileUpload);

    if (prefixHttpBtn) {
        prefixHttpBtn.addEventListener('click', () => {
            if (websiteField.value && !websiteField.value.startsWith('http')) {
                websiteField.value = 'http://' + websiteField.value;
            }
        });
    }

    cameraSelect.addEventListener('change', () => {
        currentZoom = 1;
        zoomSlider.value = 1;
        zoomLabel.textContent = '1.0x';
        nativeZoomSupported = false;
        initCamera();
    });

    // Rotate 180 (Combine H and V flip)
    if (rotate180Btn) {
        rotate180Btn.addEventListener('click', () => {
            flipH = !flipH;
            flipV = !flipV;
            rotate180Btn.classList.toggle('active', flipH && flipV);
            applyVideoTransform();
        });
    }

    // Zoom
    zoomSlider.addEventListener('input', handleZoomChange);
    zoomInBtn.addEventListener('click', () => {
        zoomSlider.value = Math.min(parseFloat(zoomSlider.max), parseFloat(zoomSlider.value) + 0.1).toFixed(1);
        handleZoomChange();
    });
    zoomOutBtn.addEventListener('click', () => {
        zoomSlider.value = Math.max(parseFloat(zoomSlider.min), parseFloat(zoomSlider.value) - 0.1).toFixed(1);
        handleZoomChange();
    });

    // Form Submission (Export)
    contactForm.addEventListener('submit', (e) => {
        e.preventDefault();
        exportVCF();
    });


    // Back photo
    backPhotoInput.addEventListener('change', handleBackPhotoUpload);
    captureBackBtn.addEventListener('click', openBackCamera);
    removeBackPhotoBtn.addEventListener('click', removeBackPhoto);
    backCaptureTakeBtn.addEventListener('click', captureBackPhoto);
    backCaptureCancelBtn.addEventListener('click', closeBackCamera);

    // Back camera rotate
    if (backRotate180Btn) {
        backRotate180Btn.addEventListener('click', () => {
            backFlipH = !backFlipH;
            backFlipV = !backFlipV;
            backRotate180Btn.classList.toggle('active', backFlipH && backFlipV);
            applyBackVideoTransform();
        });
    }
    backZoomSlider.addEventListener('input', handleBackZoomChange);
    backZoomInBtn.addEventListener('click', () => {
        backZoomSlider.value = Math.min(parseFloat(backZoomSlider.max), parseFloat(backZoomSlider.value) + 0.1).toFixed(1);
        handleBackZoomChange();
    });
    backZoomOutBtn.addEventListener('click', () => {
        backZoomSlider.value = Math.max(parseFloat(backZoomSlider.min), parseFloat(backZoomSlider.value) - 0.1).toFixed(1);
        handleBackZoomChange();
    });

    // Back crop controls
    backCropConfirmBtn.addEventListener('click', confirmBackCrop);
    backCropSkipBtn.addEventListener('click', skipBackCrop);


    // Front crop controls
    cropConfirmBtn.addEventListener('click', confirmCrop);
    cropSkipBtn.addEventListener('click', skipCrop);
    cropRetakeBtn.addEventListener('click', resetToCamera);
    autoCropBtn.addEventListener('click', () => runAutoCrop(false));
    backAutoCropBtn.addEventListener('click', () => runAutoCrop(true));


    // Crop handle dragging (covers both front and back)
    initCropDragging();

    // Re-crop buttons
    if (manualCropBtn) {
        manualCropBtn.addEventListener('click', () => {
            if (capturedImageDataUrl) openCropSection(capturedImageDataUrl);
        });
    }
    if (manualBackCropBtn) {
        manualBackCropBtn.addEventListener('click', () => {
            if (backCapturedImageDataUrl) openBackCropSection(backCapturedImageDataUrl);
        });
    }
});


// ═══════════════════════════════════════════════════════
//  Service Status Check
// ═══════════════════════════════════════════════════════
async function checkServiceStatus() {
    let anyOffline = false;

    // Check CORS Proxy (DMR)
    try {
        await fetch(`${DMR_BASE}/models`, {
            headers: { 'Authorization': 'Bearer local-key' },
            signal: AbortSignal.timeout(3000)
        });
        if (proxyStatusEl) proxyStatusEl.classList.add('hidden');
    } catch {
        if (proxyStatusEl) proxyStatusEl.classList.remove('hidden');
        anyOffline = true;
    }

    // Check OpenCV Backend via lightweight health endpoint
    try {
        await fetch(`${CV_BACKEND_BASE}/api/health`, {
            method: 'GET',
            signal: AbortSignal.timeout(3000)
        });
        // Any response means the server is up (detection is available
        // even if OCR isn't ready).
        if (cvBackendStatusEl) cvBackendStatusEl.classList.add('hidden');
    } catch {
        if (cvBackendStatusEl) cvBackendStatusEl.classList.remove('hidden');
        anyOffline = true;
    }

    if (serviceStatusBanner) {
        serviceStatusBanner.classList.toggle('hidden', !anyOffline);
    }
}

// ═══════════════════════════════════════════════════════
//  Camera Enumeration
// ═══════════════════════════════════════════════════════
async function populateCameraList() {
    try {
        const currentSelected = cameraSelect.value;
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devices.filter(d => d.kind === 'videoinput');
        console.log(`[Camera Beta Debug] Found ${videoInputs.length} video inputs. Permissions: ${videoInputs.some(v => v.label) ? 'Granted' : 'Denied/Wait'}`);

        cameraSelect.innerHTML = '';
        videoInputs.forEach((device, index) => {
            const opt = document.createElement('option');
            opt.value = device.deviceId;
            opt.textContent = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(opt);
        });

        if (currentSelected && [...cameraSelect.options].some(o => o.value === currentSelected)) {
            cameraSelect.value = currentSelected;
        }

        if (videoInputs.length === 0) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = '找不到相機';
            cameraSelect.appendChild(opt);
        }
    } catch (err) {
        console.error('Error enumerating cameras:', err);
        cameraSelect.innerHTML = '<option value="">相機不可用</option>';
    }
}

async function populateModels() {
    try {
        const res = await fetch(`${DMR_BASE}/models`, {
            headers: { 'Authorization': 'Bearer local-key' }
        });
        if (res.ok) {
            const data = await res.json();
            // Docker Model Runner returns { data: [{id, ...}, ...] }
            // Strip 'docker.io/' prefix for cleaner display
            const models = (data.data || []).map(m => {
                const cleanId = m.id.replace(/^docker\.io\//, '');
                return { name: cleanId };
            });
            window.availableModels = models;

            // Handle both select elements
            if (ollamaModelSelect && ollamaModelSelect.tagName === 'SELECT') {
                ollamaModelSelect.innerHTML = '<option value="">不使用模型</option>';
            }
            if (textModelSelect) {
                textModelSelect.innerHTML = '';
            }

            if (models.length > 0) {
                const filteredModels = models.filter(m => !m.name.includes('embed'));

                filteredModels.forEach(model => {
                    const opt = document.createElement('option');
                    opt.value = model.name;
                    opt.textContent = model.name;

                    if (ollamaModelSelect && ollamaModelSelect.tagName === 'SELECT') {
                        ollamaModelSelect.appendChild(opt.cloneNode(true));
                    }
                    if (textModelSelect) {
                        textModelSelect.appendChild(opt.cloneNode(true));
                    }
                });

                // 解析模型優先序：實測 qwen3.5:0.8b 對欄位抽取常回「全空 JSON」
                // （解析等於沒做），7B instruct 才能正確抽出欄位；
                // 已驗證可用的放前面，小模型留作備援。
                const PREFERRED_TEXT_MODELS = [
                    'qwen2.5-7B-instruct',   // qcwind/qwen2.5-7B-instruct-Q4_K_M
                    'qwen2.5:3b', 'qwen2.5:1.5b',
                    'gemma2:2b', 'phi3:mini', 'llama3.2:1b', 'qwen3.5:0.8b'
                ];
                const storedTextModel = localStorage.getItem(TEXT_MODEL_STORE_KEY);
                const preferred = PREFERRED_TEXT_MODELS.find(p =>
                    filteredModels.some(m => m.name === p || m.name.includes(p))
                );

                if (storedTextModel && filteredModels.some(m => m.name === storedTextModel) && textModelSelect) {
                    // 使用者選過的解析模型優先恢復
                    textModelSelect.value = storedTextModel;
                } else if (preferred && textModelSelect) {
                    textModelSelect.value = filteredModels.find(m => m.name === preferred || m.name.includes(preferred))?.name || preferred;
                } else if (textModelSelect) {
                    textModelSelect.value = '';
                }

                if (ollamaModelSelect) ollamaModelSelect.value = '';

                // 把 Ollama 模型清單接進主 OCR 引擎下拉：
                // 除了 PaddleOCR / Docling，也可直接選 Ollama 視覺模型當 OCR 引擎。
                if (ocrEngineSelect) {
                    ocrEngineSelect.querySelectorAll('option[value^="ollama:"]').forEach(o => o.remove());
                    filteredModels.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = 'ollama:' + m.name;
                        opt.textContent = 'Ollama: ' + m.name;
                        ocrEngineSelect.appendChild(opt);
                    });
                    // 恢復上次選的引擎(選項還在清單裡才恢復)
                    const storedEngine = localStorage.getItem(OCR_ENGINE_STORE_KEY);
                    if (storedEngine && Array.from(ocrEngineSelect.options).some(o => o.value === storedEngine)) {
                        ocrEngineSelect.value = storedEngine;
                    }
                }
            }
        }
    } catch (err) {
        console.error('Failed to fetch Ollama models:', err);
        if (textModelSelect) textModelSelect.innerHTML = '<option value="">Ollama 或 Proxy 未連線</option>';
        if (ollamaModelSelect && ollamaModelSelect.tagName === 'SELECT') {
            ollamaModelSelect.innerHTML = '<option value="">Ollama 或 Proxy 未連線</option>';
        }
    }
}


// ═══════════════════════════════════════════════════════
//  Camera Init (Fixed: use 'ideal' instead of 'exact')
// ═══════════════════════════════════════════════════════
async function initCamera() {
    const selectedDeviceId = cameraSelect.value;

    // Check if we already have an active stream
    if (stream) {
        const activeTrack = stream.getVideoTracks()[0];
        const activeDeviceId = activeTrack ? activeTrack.getSettings().deviceId : null;

        // If the stream is active and matches the selected camera (or no specific camera was forced), reuse it.
        if (!selectedDeviceId || activeDeviceId === selectedDeviceId) {
            if (videoFeed.srcObject !== stream) {
                videoFeed.srcObject = stream;
            }
            videoFeed.play().catch(e => console.warn('Could not auto-play video:', e));
            return;
        } else {
            // Camera changed. Stop the old stream.
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    }

    try {
        const selectedDeviceId = cameraSelect.value;
        const constraints = {
            video: {
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            }
        };

        // Use 'ideal' so the browser can fall back if exact match fails
        if (selectedDeviceId) {
            constraints.video.deviceId = { ideal: selectedDeviceId };
        }

        stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoFeed.srcObject = stream;
        cameraError.classList.add('hidden');

        // Immediately update list to get labels (now that permission is granted)
        await populateCameraList();

        // Check native zoom support
        const track = stream.getVideoTracks()[0];
        if (track) {
            const capabilities = track.getCapabilities ? track.getCapabilities() : {};
            if (capabilities.zoom) {
                nativeZoomSupported = true;
                zoomSlider.min = capabilities.zoom.min || 1;
                zoomSlider.max = Math.min(capabilities.zoom.max || 5, 10);
                zoomSlider.step = capabilities.zoom.step || 0.1;
            } else {
                nativeZoomSupported = false;
                zoomSlider.min = 1;
                zoomSlider.max = 5;
                zoomSlider.step = 0.1;
            }
        }

        applyVideoTransform();

    } catch (err) {
        console.error("Error accessing camera:", err);
        cameraError.classList.remove('hidden');
    }
}


// ═══════════════════════════════════════════════════════
//  Zoom
// ═══════════════════════════════════════════════════════
function handleZoomChange() {
    currentZoom = parseFloat(zoomSlider.value);
    zoomLabel.textContent = `${currentZoom.toFixed(1)}x`;

    if (nativeZoomSupported && stream) {
        const track = stream.getVideoTracks()[0];
        if (track) {
            track.applyConstraints({ advanced: [{ zoom: currentZoom }] })
                .catch(() => {
                    // Fallback to CSS zoom if native fails
                    nativeZoomSupported = false;
                    applyVideoTransform();
                });
            // Don't apply CSS zoom when native works
            return;
        }
    }

    // CSS-based digital zoom
    applyVideoTransform();
}

function applyVideoTransform() {
    const scaleX = flipH ? -1 : 1;
    const scaleY = flipV ? -1 : 1;
    const cssZoom = nativeZoomSupported ? 1 : currentZoom;
    videoFeed.style.transform = `scale(${scaleX * cssZoom}, ${scaleY * cssZoom})`;
}


// ═══════════════════════════════════════════════════════
//  Section Navigation
// ═══════════════════════════════════════════════════════
function showSection(sectionToShow) {
    [cameraSection, cropSection, processingSection, resultSection, successSection, backCameraSection, backCropSection, backProcessingSection].forEach(sec => {
        sec.classList.add('hidden');
        sec.classList.remove('active');
    });
    sectionToShow.classList.remove('hidden');

    // Auto-resume video if stream exists
    if (sectionToShow === cameraSection && videoFeed && videoFeed.srcObject) {
        videoFeed.play().catch(e => console.warn('Could not auto-play video:', e));
    }
    if (sectionToShow === backCameraSection && backVideoFeed && backVideoFeed.srcObject) {
        backVideoFeed.play().catch(e => console.warn('Could not auto-play video:', e));
    }

    setTimeout(() => {
        sectionToShow.classList.add('active');
    }, 10);
}


// ═══════════════════════════════════════════════════════
//  Capture Image → go to Crop Section
// ═══════════════════════════════════════════════════════
async function captureImage() {
    if (!videoFeed.videoWidth) return;

    const vw = videoFeed.videoWidth;
    const vh = videoFeed.videoHeight;
    const cssZoom = nativeZoomSupported ? 1 : currentZoom;
    const cropW = vw / cssZoom;
    const cropH = vh / cssZoom;
    const sx = (vw - cropW) / 2;
    const sy = (vh - cropH) / 2;

    captureCanvas.width = Math.round(cropW);
    captureCanvas.height = Math.round(cropH);
    const ctx = captureCanvas.getContext('2d');
    ctx.save();
    if (flipH || flipV) {
        ctx.translate(flipH ? captureCanvas.width : 0, flipV ? captureCanvas.height : 0);
        ctx.scale(flipH ? -1 : 1, flipV ? -1 : 1);
    }
    ctx.drawImage(videoFeed, sx, sy, cropW, cropH, 0, 0, captureCanvas.width, captureCanvas.height);
    ctx.restore();

    capturedImageDataUrl = captureCanvas.toDataURL('image/jpeg', 0.92);
    previewImage.src = capturedImageDataUrl;

    if (videoFeed) {
        videoFeed.pause();
    }

    // New: Try automated cycle
    autoCycleCrop(capturedImageDataUrl, false);
}

function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
        capturedImageDataUrl = event.target.result;
        previewImage.src = capturedImageDataUrl;
        if (videoFeed) {
            videoFeed.pause();
        }
        // New: Try automated cycle
        autoCycleCrop(capturedImageDataUrl, false);
        fileInput.value = '';
    };
    reader.readAsDataURL(file);
}





// ═══════════════════════════════════════════════════════
//  API Call for OpenCV Edge Detection
// ═══════════════════════════════════════════════════════
// Tracks whether the last detect call failed because the backend was
// unreachable (vs. the backend running but finding no card). Lets the UI
// show the right message instead of always blaming "no edges".
let lastDetectBackendDown = false;

async function detectCardFetch(imageDataUrl) {
    lastDetectBackendDown = false;
    try {
        const b64 = imageDataUrlToBase64(imageDataUrl);
        const res = await fetch(`${CV_BACKEND_BASE}/api/detect-card`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: b64 })
        });
        if (res.ok) {
            const data = await res.json();
            if (data && data.corners) {
                return data.corners;
            }
            // 200 but no corners → backend ran, just didn't find a card.
            return null;
        }
        // Non-OK HTTP → treat as backend problem.
        lastDetectBackendDown = true;
        console.error('CV Backend returned HTTP', res.status);
    } catch (err) {
        // fetch threw → server not running / not reachable.
        lastDetectBackendDown = true;
        console.error('CV Backend Error (is opencv_backend.py running on :5050?):', err);
    }
    return null;
}


// ═══════════════════════════════════════════════════════
//  API Call for Vision Model Edge Detection
// ═══════════════════════════════════════════════════════
async function detectCardWithVisionModel(imageDataUrl, model) {
    if (!model) {
        alert("請先選擇模型來進行自動裁切偵測。");
        return null;
    }

    try {
        const processedImageUrl = await processImageForDMR(imageDataUrl, true, 800);
        const b64 = imageDataUrlToBase64(processedImageUrl);
        const prompt = `Detect the four corners of the business card in this image. 
Return the coordinates as a JSON object with exactly these keys: 'tl', 'tr', 'bl', 'br' (Top-Left, Top-Right, Bottom-Left, Bottom-Right).
Each value must be an object with 'x' and 'y' properties representing the coordinates as a percentage (0 to 100) of the image width and height.
Output ONLY valid JSON. Do not add any markdown, comments, or extra text.
Example: {"tl": {"x": 10, "y": 10}, "tr": {"x": 90, "y": 10}, "bl": {"x": 10, "y": 90}, "br": {"x": 90, "y": 90}}`;

        const responseText = await dmrChat(model, prompt, b64);

        console.log("Vision Model Raw Crop Response:", responseText);

        // Sanitize response: Sometimes LLM wraps in markdown or adds text
        let cleanedText = responseText.replace(/```json/gi, '').replace(/```/g, '').trim();

        // Use greedy match to capture nested JSON objects properly
        const jsonMatch = cleanedText.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            let jsonString = '';
            try {
                // Fix potential trailing commas before closing braces/brackets which breaks JSON.parse
                jsonString = jsonMatch[0].replace(/,\s*([}\]])/g, '$1');
                const corners = JSON.parse(jsonString);
                if (corners.tl && corners.tr && corners.bl && corners.br) {
                    return corners;
                }
            } catch (jsonErr) {
                console.warn('Vision Model Crop JSON parse failed:', jsonString, jsonErr);
            }
        }
    } catch (err) {
        console.error('Vision Model Edge Detection Error:', err);
    }
    return null;
}

// ═══════════════════════════════════════════════════════
//  Manual Auto Crop Trigger
// ═══════════════════════════════════════════════════════
async function runAutoCrop(isBack) {
    const btn = isBack ? backAutoCropBtn : autoCropBtn;
    const originalText = btn.innerHTML;
    const imageData = isBack ? backCapturedImageDataUrl : capturedImageDataUrl;
    const selectedModel = ollamaModelSelect.value;

    if (!imageData) return;

    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> 偵測中...';

    // In Beta, use OpenCV backend for auto-crop as default and reliable source
    const corners = await detectCardFetch(imageData);

    console.log("Auto-crop returned corners:", corners);
    if (corners && corners.tl && corners.tr && corners.bl && corners.br) {
        if (isBack) {
            backCropCorners = corners;
            updateBackCropHandles();
        } else {
            cropCorners = corners;
            updateCropHandles();
        }
    } else if (lastDetectBackendDown) {
        alert('無法連線到自動裁切後端服務 (localhost:5050)。\n請確認 opencv_backend.py 已啟動（執行 run_scanner.bat），再試一次。');
    } else {
        alert('未偵測到明顯的名片邊緣，請手動調整裁切框。');
    }

    btn.disabled = false;
    btn.innerHTML = originalText;
}

/**
 * Automated Crop Cycle:
 * 1. Detect corners via backend
 * 2. If found, morph and go to processing
 * 3. If failed, fallback to manual crop screen
 */
async function autoCycleCrop(imageDataUrl, isBack) {
    console.log(`[Auto-Crop Beta] Starting automated cycle for ${isBack ? 'Back' : 'Front'}...`);
    
    try {
        const corners = await detectCardFetch(imageDataUrl);
        if (corners) {
            console.log('[Auto-Crop Beta] Card detected! Moving to perspective transform.');
            if (isBack) {
                backCropCorners = corners;
                confirmBackCrop();
            } else {
                cropCorners = corners;
                confirmCrop();
            }
        } else {
            console.warn('[Auto-Crop Beta] No card detected. Falling back to manual mode.');
            // 後端沒開時原本靜默退回手動裁切，使用者只會覺得「自動裁切壞了」；
            // 明確告知原因（與 runAutoCrop 的提示一致）。
            if (lastDetectBackendDown) {
                alert('無法連線到自動裁切後端服務 (localhost:5050)。\n請確認 opencv_backend.py 已啟動（run_scanner.bat），先以手動裁切繼續。');
            }
            if (isBack) {
                openBackCropSection(imageDataUrl);
            } else {
                openCropSection(imageDataUrl);
            }
        }
    } catch (err) {
        console.error('[Auto-Crop Beta] Error during cycle:', err);
        if (isBack) openBackCropSection(imageDataUrl);
        else openCropSection(imageDataUrl);
    }
}

// ═══════════════════════════════════════════════════════
//  Manual Crop & Perspective Correction
// ═══════════════════════════════════════════════════════
async function openCropSection(imageDataUrl) {
    showSection(cropSection);

    // Default to 10% margin initially
    cropCorners = {
        tl: { x: 10, y: 10 },
        tr: { x: 90, y: 10 },
        bl: { x: 10, y: 90 },
        br: { x: 90, y: 90 }
    };

    const img = new Image();
    img.onload = () => {
        const maxW = cropContainer.clientWidth;
        const scale = maxW / img.width;
        const displayW = maxW;
        const displayH = img.height * scale;

        cropCanvas.width = displayW;
        cropCanvas.height = displayH;
        cropContainer.style.height = displayH + 'px';

        const ctx = cropCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0, displayW, displayH);

        cropOverlaySvg.setAttribute('width', displayW);
        cropOverlaySvg.setAttribute('height', displayH);
        cropOverlaySvg.style.width = displayW + 'px';
        cropOverlaySvg.style.height = displayH + 'px';

        updateCropHandles();
    };
    img.src = imageDataUrl;
}

function updateCropHandles() {
    const w = cropCanvas.width;
    const h = cropCanvas.height;

    for (const key of ['tl', 'tr', 'bl', 'br']) {
        const handle = document.getElementById(`handle-${key}`);
        const px = (cropCorners[key].x / 100) * w;
        const py = (cropCorners[key].y / 100) * h;
        handle.style.left = px + 'px';
        handle.style.top = py + 'px';
    }

    // Update polygon
    const points = ['tl', 'tr', 'br', 'bl'].map(key => {
        const px = (cropCorners[key].x / 100) * w;
        const py = (cropCorners[key].y / 100) * h;
        return `${px},${py}`;
    }).join(' ');
    cropPolygon.setAttribute('points', points);
}

// ─── Drag Handling (shared for front + back crop) ───
function initCropDragging() {
    // Register handles for BOTH front and back crop containers
    const frontHandles = cropContainer.querySelectorAll('.crop-handle');
    const backHandles = backCropContainer.querySelectorAll('.crop-handle');

    frontHandles.forEach(handle => {
        handle.addEventListener('mousedown', (e) => startDrag(e, false));
        handle.addEventListener('touchstart', (e) => startDrag(e, false), { passive: false });
    });

    backHandles.forEach(handle => {
        handle.addEventListener('mousedown', (e) => startDrag(e, true));
        handle.addEventListener('touchstart', (e) => startDrag(e, true), { passive: false });
    });

    document.addEventListener('mousemove', onDrag);
    document.addEventListener('touchmove', onDrag, { passive: false });
    document.addEventListener('mouseup', endDrag);
    document.addEventListener('touchend', endDrag);
}

function startDrag(e, isBack) {
    e.preventDefault();
    dragTarget = e.currentTarget.dataset.corner;
    dragIsBack = isBack;
}

function onDrag(e) {
    if (!dragTarget) return;
    e.preventDefault();

    const canvas = dragIsBack ? backCropCanvas : cropCanvas;
    const corners = dragIsBack ? backCropCorners : cropCorners;
    const rect = canvas.getBoundingClientRect();

    let clientX, clientY;
    if (e.touches) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
    } else {
        clientX = e.clientX;
        clientY = e.clientY;
    }

    let px = ((clientX - rect.left) / rect.width) * 100;
    let py = ((clientY - rect.top) / rect.height) * 100;
    px = Math.max(0, Math.min(100, px));
    py = Math.max(0, Math.min(100, py));

    corners[dragTarget] = { x: px, y: py };

    if (dragIsBack) {
        updateBackCropHandles();
    } else {
        updateCropHandles();
    }
}

function endDrag() {
    dragTarget = null;
    dragIsBack = false;
}

// ─── Apply Perspective Transform ───
// ─── Apply Perspective Transform (Via Backend) ───
async function confirmCrop() {
    if (!capturedImageDataUrl) return;

    try {
        showSection(processingSection);
        progressText.textContent = '正在進行透視校正與影像優化...';
        progressFill.style.width = '20%';

        const b64 = imageDataUrlToBase64(capturedImageDataUrl);
        const res = await fetch(`${CV_BACKEND_BASE}/api/morph-card`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: b64,
                corners: cropCorners
            })
        });

        if (res.ok) {
            const data = await res.json();
            if (data && data.image) {
                const correctedUrl = `data:image/jpeg;base64,${data.image}`;
                capturedImageDataUrl = correctedUrl;
                previewImage.src = correctedUrl;
                
                // Store for vCard Logo
                window.cardLogoBase64 = data.image;

                // Update preview in result section
                const logoPreview = document.getElementById('cardLogoPreview');
                if (logoPreview) {
                    logoPreview.src = correctedUrl;
                    logoPreview.classList.remove('hidden');
                }

                progressFill.style.width = '40%';
                progressText.textContent = '影像優化完成，開始 OCR 辨識...';
                performOCR(correctedUrl);
                return;
            }
        }
        throw new Error('Morphing failed');
    } catch (err) {
        console.error('Perspective Transform Error:', err);
        // Fallback to skip crop or alert user
        alert('透視校正失敗，將使用原始影像進行辨識。');
        skipCrop();
    }
}



function skipCrop() {
    if (!capturedImageDataUrl) return;
    showSection(processingSection);
    performOCR(capturedImageDataUrl);
}

// ─── Homography Math ───
function dist(a, b) {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

/**
 * Compute 3×3 homography matrix H such that for each i:
 *   dst[i] = H * src[i]  (in homogeneous coordinates)
 * We solve dst→src so we can do inverse mapping.
 * Uses DLT (Direct Linear Transform) method.
 *
 * @param {Array} from - 4 source points [{x,y}, ...]
 * @param {Array} to   - 4 destination points [{x,y}, ...]
 * @returns {number[]} 9-element array representing 3×3 matrix row-major
 */
function computeHomography(from, to) {
    // Build the 8×9 matrix A for Ah=0
    const A = [];
    for (let i = 0; i < 4; i++) {
        const sx = from[i].x, sy = from[i].y;
        const dx = to[i].x, dy = to[i].y;

        A.push([-sx, -sy, -1, 0, 0, 0, dx * sx, dx * sy, dx]);
        A.push([0, 0, 0, -sx, -sy, -1, dy * sx, dy * sy, dy]);
    }

    // Solve using simplified SVD for 8×9 system
    // We use Gaussian elimination to find the null space
    const h = solveHomographySystem(A);

    return h;
}

/**
 * Solve the 8×9 homogeneous system Ah=0 via Gaussian elimination
 * Returns the 9-element solution vector (null space)
 */
function solveHomographySystem(A) {
    const rows = 8;
    const cols = 9;

    // Create augmented matrix copy
    const M = A.map(row => [...row]);

    // Forward elimination
    for (let col = 0; col < rows; col++) {
        // Find pivot
        let maxVal = 0;
        let maxRow = col;
        for (let row = col; row < rows; row++) {
            if (Math.abs(M[row][col]) > maxVal) {
                maxVal = Math.abs(M[row][col]);
                maxRow = row;
            }
        }

        // Swap rows
        [M[col], M[maxRow]] = [M[maxRow], M[col]];

        if (Math.abs(M[col][col]) < 1e-10) continue;

        // Scale pivot row
        const pivot = M[col][col];
        for (let j = col; j < cols; j++) {
            M[col][j] /= pivot;
        }

        // Eliminate column
        for (let row = 0; row < rows; row++) {
            if (row === col) continue;
            const factor = M[row][col];
            for (let j = col; j < cols; j++) {
                M[row][j] -= factor * M[col][j];
            }
        }
    }

    // The solution: h[8] = 1 (free variable), back-substitute
    const h = new Array(9);
    h[8] = 1;
    for (let i = 7; i >= 0; i--) {
        h[i] = -M[i][8]; // Since M[i][i]=1 and h[8]=1
    }

    return h;
}

/**
 * Apply the 3×3 homography to a point (x,y).
 * Returns the transformed {x, y}.
 */
function applyHomography(H, x, y) {
    const w = H[6] * x + H[7] * y + H[8];
    return {
        x: (H[0] * x + H[1] * y + H[2]) / w,
        y: (H[3] * x + H[4] * y + H[5]) / w
    };
}


// ═══════════════════════════════════════════════════════
//  Back Photo Handling
// ═══════════════════════════════════════════════════════
function handleBackPhotoUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (event) => {
        backCapturedImageDataUrl = event.target.result;
        await openBackCropSection(backCapturedImageDataUrl);
        backPhotoInput.value = '';
    };
    reader.readAsDataURL(file);
}

async function openBackCamera() {
    previousSection = resultSection;
    backFlipH = false;
    backFlipV = false;
    backCurrentZoom = 1;
    backNativeZoomSupported = false;
    backZoomSlider.value = 1;
    backZoomLabel.textContent = '1.0x';
    if (backRotate180Btn) backRotate180Btn.classList.remove('active');

    showSection(backCameraSection);

    const selectedDeviceId = cameraSelect.value;

    // Reuse back camera stream
    if (backStream) {
        const activeTrack = backStream.getVideoTracks()[0];
        const activeDeviceId = activeTrack ? activeTrack.getSettings().deviceId : null;

        if (!selectedDeviceId || activeDeviceId === selectedDeviceId) {
            if (backVideoFeed.srcObject !== backStream) {
                backVideoFeed.srcObject = backStream;
            }
            backVideoFeed.play().catch(e => console.warn('Could not auto-play back video:', e));
            return;
        } else {
            backStream.getTracks().forEach(track => track.stop());
            backStream = null;
        }
    }

    try {
        const selectedDeviceId = cameraSelect.value;
        const constraints = {
            video: {
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            }
        };
        if (selectedDeviceId) {
            constraints.video.deviceId = { ideal: selectedDeviceId };
        }

        backStream = await navigator.mediaDevices.getUserMedia(constraints);
        backVideoFeed.srcObject = backStream;

        // Check native zoom support for back camera
        const track = backStream.getVideoTracks()[0];
        if (track) {
            const capabilities = track.getCapabilities ? track.getCapabilities() : {};
            if (capabilities.zoom) {
                backNativeZoomSupported = true;
                backZoomSlider.min = capabilities.zoom.min || 1;
                backZoomSlider.max = Math.min(capabilities.zoom.max || 5, 10);
                backZoomSlider.step = capabilities.zoom.step || 0.1;
            }
        }

        applyBackVideoTransform();
    } catch (err) {
        console.error('Error opening back camera:', err);
        alert('無法開啟相機，請改用上傳方式。');
        closeBackCamera();
    }
}

function applyBackVideoTransform() {
    const scaleX = backFlipH ? -1 : 1;
    const scaleY = backFlipV ? -1 : 1;
    const cssZoom = backNativeZoomSupported ? 1 : backCurrentZoom;
    backVideoFeed.style.transform = `scale(${scaleX * cssZoom}, ${scaleY * cssZoom})`;
}

function handleBackZoomChange() {
    backCurrentZoom = parseFloat(backZoomSlider.value);
    backZoomLabel.textContent = `${backCurrentZoom.toFixed(1)}x`;

    if (backNativeZoomSupported && backStream) {
        const track = backStream.getVideoTracks()[0];
        if (track) {
            track.applyConstraints({ advanced: [{ zoom: backCurrentZoom }] })
                .catch(() => {
                    backNativeZoomSupported = false;
                    applyBackVideoTransform();
                });
            return;
        }
    }
    applyBackVideoTransform();
}

async function captureBackPhoto() {
    if (!backVideoFeed.videoWidth) return;

    const vw = backVideoFeed.videoWidth;
    const vh = backVideoFeed.videoHeight;
    const cssZoom = backNativeZoomSupported ? 1 : backCurrentZoom;
    const cropW = vw / cssZoom;
    const cropH = vh / cssZoom;
    const sx = (vw - cropW) / 2;
    const sy = (vh - cropH) / 2;

    backCaptureCanvas.width = Math.round(cropW);
    backCaptureCanvas.height = Math.round(cropH);

    const ctx = backCaptureCanvas.getContext('2d');
    ctx.save();
    if (backFlipH || backFlipV) {
        ctx.translate(
            backFlipH ? backCaptureCanvas.width : 0,
            backFlipV ? backCaptureCanvas.height : 0
        );
        ctx.scale(backFlipH ? -1 : 1, backFlipV ? -1 : 1);
    }
    ctx.drawImage(backVideoFeed, sx, sy, cropW, cropH, 0, 0, backCaptureCanvas.width, backCaptureCanvas.height);
    ctx.restore();

    backCapturedImageDataUrl = backCaptureCanvas.toDataURL('image/jpeg', 0.92);

    if (backStream) {
        backStream.getTracks().forEach(t => t.stop());
        backStream = null;
    }

    // New: Try automated cycle
    autoCycleCrop(backCapturedImageDataUrl, true);
}

function closeBackCamera() {
    if (backStream) {
        backStream.getTracks().forEach(t => t.stop());
        backStream = null;
    }
    showSection(previousSection || resultSection);
}

// ─── Back Crop Section ───
async function openBackCropSection(imageDataUrl) {
    showSection(backCropSection);

    // Default to 10% margin initially
    backCropCorners = {
        tl: { x: 10, y: 10 },
        tr: { x: 90, y: 10 },
        bl: { x: 10, y: 90 },
        br: { x: 90, y: 90 }
    };

    const img = new Image();
    img.onload = () => {
        const maxW = backCropContainer.clientWidth;
        const scale = maxW / img.width;
        const displayW = maxW;
        const displayH = img.height * scale;

        backCropCanvas.width = displayW;
        backCropCanvas.height = displayH;
        backCropContainer.style.height = displayH + 'px';

        const ctx = backCropCanvas.getContext('2d');
        ctx.drawImage(img, 0, 0, displayW, displayH);

        backCropOverlaySvg.setAttribute('width', displayW);
        backCropOverlaySvg.setAttribute('height', displayH);
        backCropOverlaySvg.style.width = displayW + 'px';
        backCropOverlaySvg.style.height = displayH + 'px';

        updateBackCropHandles();
    };
    img.src = imageDataUrl;
}

function updateBackCropHandles() {
    const w = backCropCanvas.width;
    const h = backCropCanvas.height;

    for (const key of ['tl', 'tr', 'bl', 'br']) {
        const handle = document.getElementById(`back-handle-${key}`);
        const px = (backCropCorners[key].x / 100) * w;
        const py = (backCropCorners[key].y / 100) * h;
        handle.style.left = px + 'px';
        handle.style.top = py + 'px';
    }

    const points = ['tl', 'tr', 'br', 'bl'].map(key => {
        const px = (backCropCorners[key].x / 100) * w;
        const py = (backCropCorners[key].y / 100) * h;
        return `${px},${py}`;
    }).join(' ');
    backCropPolygon.setAttribute('points', points);
}

// ─── Apply Perspective Transform (Via Backend) ───
async function confirmBackCrop() {
    if (!backCapturedImageDataUrl) return;

    try {
        showSection(backProcessingSection);
        backProgressText.textContent = '正在進行背面透視校正與影像優化...';
        backProgressFill.style.width = '30%';

        const b64 = imageDataUrlToBase64(backCapturedImageDataUrl);
        const res = await fetch(`${CV_BACKEND_BASE}/api/morph-card`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: b64,
                corners: backCropCorners
            })
        });

        if (res.ok) {
            const data = await res.json();
            if (data && data.image) {
                const correctedUrl = `data:image/jpeg;base64,${data.image}`;
                backCapturedImageDataUrl = correctedUrl;
                backPhotoDataUrl = correctedUrl;
                showBackPhotoPreview(correctedUrl);

                backProgressFill.style.width = '60%';
                backProgressText.textContent = '背面影像優化完成，開始 OCR...';
                performBackOCR(correctedUrl);
                return;
            }
        }
        throw new Error('Back morphing failed');
    } catch (err) {
        console.error('Back Perspective Transform Error:', err);
        alert(`背面校正失敗: ${err.message || '未知錯誤'}`);
        skipBackCrop();
    }
}

function skipBackCrop() {
    console.log('DEBUG: Skipping back crop, proceeding to OCR');
    if (!backCapturedImageDataUrl) return;
    showSection(backProcessingSection);
    backPhotoDataUrl = backCapturedImageDataUrl;
    showBackPhotoPreview(backCapturedImageDataUrl);
    performBackOCR(backCapturedImageDataUrl);
}

// ─── Back OCR ───
async function performBackOCR(imageDataUrl) {
    showSection(backProcessingSection);
    backProgressFill.style.width = '10%';
    backProgressText.textContent = '載入本地 OCR 引擎...';
    backLiveOCRText.value = '';

    try {
        const backStartTime = Date.now();
        const backTimer = setInterval(() => {
            const elapsed = ((Date.now() - backStartTime) / 1000).toFixed(0);
            backProgressText.textContent = `使用本地 OCR 辨識背面文字中... (${elapsed}s)`;
        }, 1000);

        // Use Selected OCR Engine
        const primaryEngine = (ocrEngineSelect && ocrEngineSelect.value) || 'paddle';
        let text = '';
        if (primaryEngine === 'docling') {
            text = await doclingFetch(imageDataUrl);
        } else if (primaryEngine.startsWith('ollama:')) {
            text = await ollamaVisionOcr(primaryEngine.slice(7), imageDataUrl);
        } else {
            // 與正面流程一致：先縮到 800px 再送 OCR。背面原本漏了這步，
            // 透視校正後的大圖（~1200px）在 CPU 上一跑就是 20 秒以上。
            const downscaled = await processImageForDMR(imageDataUrl, false, 800);
            text = await paddleOcrFetch(downscaled);
        }
        clearInterval(backTimer);

        // --- Fallback to Vision Model if PaddleOCR returns nothing ---
        if (!text || text.trim() === '') {
            console.warn('DEBUG: Back PaddleOCR returned no text, falling back to Vision Model');
            backProgressText.textContent = `本地 OCR 無結果，改用 Vision 模型辨識背面...`;
            
            const b64 = imageDataUrlToBase64(imageDataUrl);
            const ocrPrompt = "請辨識名片影像。務必保留原始字體（繁體維持繁體，簡體維持簡體），禁止變更字形。Extract all text exactly as it appears, preserving Traditional/Simplified script. No JSON, raw text only.";
            const visionModel = (textModelSelect && textModelSelect.value) || 'ai/gemma3';
            text = await dmrChat(visionModel, ocrPrompt, b64);
            console.log('DEBUG: Back Vision Model Fallback result length:', (text || '').length);
        }

        backProgressFill.style.width = '60%';
        backLiveOCRText.value = text;

        if (!text || text.trim() === '') {
             console.warn('DEBUG: No text extracted from back of card');
        }

        const parseModel = (textModelSelect && textModelSelect.value) || 'qcwind/qwen2.5-7B-instruct-Q4_K_M';
        const backParseStartTime = Date.now();
        const backParseTimer = setInterval(() => {
            const elapsed = ((Date.now() - backParseStartTime) / 1000).toFixed(0);
            backProgressText.textContent = `使用 ${parseModel} 解析背面表單資料中... (${elapsed}s)`;
        }, 1000);

        const jsonResult = await parseTextToJsonViaLLM(text);

        clearInterval(backParseTimer);
        backProgressFill.style.width = '100%';
        backProgressText.textContent = '背面掃描完成';

        mergeBackOCRWithLLM(jsonResult);

        setTimeout(() => {
            showSection(resultSection);
        }, 500);
    } catch (err) {
        console.error('Back OCR Error:', err);
        alert(`背面辨識發生錯誤。請確認後端服務正在運行。\n${err.message || ''}`);
        showSection(resultSection);
    }
}

// Merge back card OCR into empty fields (only fill if field is currently empty)
function mergeBackOCR(text) {
    const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 1);
    const emailRegex = /([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)/i;
    const phoneRegex = /(?:\+?\d{1,4}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}/;
    const urlRegex = /(?:https?:\/\/)?(?:www\.)?[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:\/[^\s]*)*/i;
    const cnAddressKeywords = ['路', '街', '段', '巷', '弄', '號', '樓', '市', '區', '縣', '鄉', '鎮', '村', '里', '鄰'];
    const enAddressKeywords = ['road', 'rd.', 'rd', 'street', 'st.', 'st', 'avenue', 'ave', 'blvd', 'lane', 'floor', 'suite', 'building', 'bldg', 'no.', 'sec.', 'district', 'city', 'county'];

    for (const line of lines) {
        const lowerLine = line.toLowerCase();

        // Email
        if (!emailField.value) {
            const match = line.replace(/\s+/g, '').match(emailRegex);
            if (match) { emailField.value = match[1]; continue; }
        }

        // Website
        if (!websiteField.value && (lowerLine.includes('www.') || lowerLine.includes('http') || lowerLine.includes('.com'))) {
            const match = line.match(urlRegex);
            if (match && !match[0].includes('@')) {
                websiteField.value = match[0].startsWith('http') ? match[0] : 'https://' + match[0];
                continue;
            }
        }

        // Phone
        const phoneMatch = line.match(phoneRegex);
        if (phoneMatch && !lowerLine.includes('fax')) {
            if (!mobileField.value) { mobileField.value = phoneMatch[0]; continue; }
            if (!workPhoneField.value) { workPhoneField.value = phoneMatch[0]; continue; }
        }

        // Chinese address
        if (!addressCnField.value && cnAddressKeywords.some(kw => line.includes(kw))) {
            const result = extractPostalCode(line);
            addressCnField.value = result.address;
            if (result.postalCode && !postalCodeCnField.value) postalCodeCnField.value = result.postalCode;
            continue;
        }

        // English address
        if (!addressEnField.value && enAddressKeywords.some(kw => lowerLine.includes(kw))) {
            const result = extractPostalCode(line);
            addressEnField.value = result.address;
            if (result.postalCode && !postalCodeEnField.value) postalCodeEnField.value = result.postalCode;
            continue;
        }
    }
}

function showBackPhotoPreview(dataUrl) {
    backPreviewImg.src = dataUrl;
    backPhotoPreview.classList.remove('hidden');
    removeBackPhotoBtn.classList.remove('hidden');
    if (manualBackCropBtn) manualBackCropBtn.classList.remove('hidden');
}

function removeBackPhoto() {
    backPhotoDataUrl = null;
    backCapturedImageDataUrl = null;
    backPreviewImg.src = '';
    backPhotoPreview.classList.add('hidden');
    removeBackPhotoBtn.classList.add('hidden');
    if (manualBackCropBtn) manualBackCropBtn.classList.add('hidden');
}


// ═══════════════════════════════════════════════════════
//  Reset
// ═══════════════════════════════════════════════════════
function resetToCamera() {
    isProcessing = false;
    capturedImageDataUrl = null;
    showSection(cameraSection);
    initCamera();
}

function resetForm() {
    contactForm.reset();
    liveOCRText.value = '';
    removeBackPhoto();
}


// ═══════════════════════════════════════════════════════
//  OCR — PaddleOCR + Docker Model Runner Pipeline
// ═══════════════════════════════════════════════════════

// 以 Ollama 視覺模型直接做整張名片 OCR(主引擎選 ollama:<model> 時使用)
const VISION_OCR_PROMPT = '請辨識名片影像。務必保留原始字體（繁體維持繁體，簡體維持簡體），禁止變更字形。Extract all text exactly as it appears, preserving Traditional/Simplified script. No JSON, raw text only.';
async function ollamaVisionOcr(model, imageDataUrl) {
    const downscaled = await processImageForDMR(imageDataUrl, false, 800);
    const b64 = imageDataUrlToBase64(downscaled);
    return dmrChat(model, VISION_OCR_PROMPT, b64);
}

const OCR_FIELDS = [
    { key: 'familyNameCn', label: '中文姓' },
    { key: 'givenNameCn', label: '中文名' },
    { key: 'givenNameEn', label: 'First Name' },
    { key: 'familyNameEn', label: 'Last Name' },
    { key: 'company', label: '公司' },
    { key: 'department', label: '部門' },
    { key: 'title', label: '職稱' },
    { key: 'taxId', label: '統一編號' },
    { key: 'workPhone', label: '公司電話' },
    { key: 'ext', label: '分機' },
    { key: 'mobile', label: '手機' },
    { key: 'email', label: 'Email' },
    { key: 'website', label: '網址' },
    { key: 'addressCn', label: '中文地址' },
    { key: 'postalCodeCn', label: '郵遞區號' },
    { key: 'addressEn', label: 'English Address' },
];

let qwenResult = {};
let glmResult = {};

function getFieldElement(key) {
    const map = {
        familyNameCn: familyNameCnField, givenNameCn: givenNameCnField,
        givenNameEn: givenNameEnField, familyNameEn: familyNameEnField,
        company: companyField, department: departmentField, title: titleField,
        taxId: taxIdField,
        workPhone: workPhoneField, ext: extField, mobile: mobileField,
        email: emailField, website: websiteField,
        addressCn: addressCnField, postalCodeCn: postalCodeCnField,
        addressEn: addressEnField,
    };
    return map[key] || null;
}

async function performOCR(imageDataUrl) {
    try {
        const textModelName = (textModelSelect && textModelSelect.value) || 'ai/gemma3';
        console.log('DEBUG: performOCR started, model:', textModelName);

        progressFill.style.width = '10%';
        progressText.textContent = `正在啟動本地 OCR 引擎...`;

        if (singleOcrStatus) {
            singleOcrStatus.classList.remove('hidden');
            if (activeModelLabel) activeModelLabel.textContent = 'PaddleOCR:';
            if (activeModelStatus) {
                activeModelStatus.textContent = '辨識中...';
                activeModelStatus.style.color = '';
            }
        }

        const startTime = Date.now();
        const timer = setInterval(() => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
            progressText.textContent = `快速文字提取中... (${elapsed}s)`;
        }, 1000);

        // --- Step 1: Primary OCR Engine ---
        const primaryEngine = (ocrEngineSelect && ocrEngineSelect.value) || 'paddle';
        const engineLabel = primaryEngine.startsWith('ollama:') ? primaryEngine.slice(7)
            : (primaryEngine === 'docling' ? 'Docling' : 'PaddleOCR');
        if (singleOcrStatus && activeModelLabel) {
            activeModelLabel.textContent = engineLabel + ':';
        }

        let rawText = '';
        if (primaryEngine === 'docling') {
            rawText = await doclingFetch(imageDataUrl);
        } else if (primaryEngine.startsWith('ollama:')) {
            rawText = await ollamaVisionOcr(primaryEngine.slice(7), imageDataUrl);
        } else {
            const downscaled = await processImageForDMR(imageDataUrl, false, 800);
            rawText = await paddleOcrFetch(downscaled);
        }
        clearInterval(timer);
        
        // --- Fallback to Vision Model if PaddleOCR returns nothing ---
        if (!rawText || rawText.trim() === '') {
            console.warn('DEBUG: PaddleOCR returned no text, falling back to Vision Model');
            progressText.textContent = `本地 OCR 無結果，改用 Vision 模型辨識中...`;
            if (activeModelStatus) activeModelStatus.textContent = '切換至 Vision 模型...';
            
            const b64 = imageDataUrlToBase64(imageDataUrl);
            const visionModel = textModelName || 'ai/gemma3';
            rawText = await dmrChat(visionModel, "Please perform OCR on this business card image. Extract all text exactly as it appears. Do not format as JSON, just raw text.", b64);
            console.log('DEBUG: Vision Model Fallback result length:', (rawText || '').length);
        }

        if (activeModelStatus) {
            activeModelStatus.textContent = '✓ OCR 完成';
        }

        liveOCRText.value = rawText;

        if (!rawText || rawText.trim() === '') {
            alert('無法從影像中提取文字，請確認圖片清晰度並重試。');
            resetToCamera();
            return;
        }

        // --- Step 2: Regex Fast Path ---
        console.log('DEBUG: Starting Regex Fast Path');
        // Clear form first
        contactForm.reset();
        
        const regexExtracted = parseContactData(rawText);
        fillFormFromObj(regexExtracted);

        if (activeModelStatus) {
            activeModelStatus.textContent = '✓ OCR 完成 (背景 LLM 解析中...)';
            activeModelStatus.style.color = 'var(--accent, #6366f1)';
        }

        // Show result section immediately
        progressFill.style.width = '100%';
        progressText.textContent = '已套用快速解析。LLM 深度解析在背景執行中...';
        showSection(resultSection);
        if (singleOcrStatus) singleOcrStatus.classList.add('hidden');

        // --- Step 3: Background LLM Parsing ---
        (async () => {
            try {
                console.log('DEBUG: Starting LLM Background Parsing');
                const extractedJson = await parseTextToJsonViaLLM(rawText);
                console.log('DEBUG: LLM Parsing completed, data keys:', Object.keys(extractedJson));
                
                // Trigger Comparison UI
                window.llmExtractedData = extractedJson;
                window.currentFormData = getFormData();
                
                const banner = document.getElementById('llmResultBanner');
                if (banner) {
                    banner.classList.remove('hidden');
                    banner.classList.add('pulse-animation');
                }
            } catch (err) {
                console.error('LLM Background Error:', err);
            }
        })();

    } catch (err) {
        console.error('OCR Error:', err);
        alert(`文字辨識發生錯誤。請確認後端服務正在運行。\n${err.message || ''}`);
        resetToCamera();
    }
}

// ═══════════════════════════════════════════════════════
//  Infer English Name from Email
//  When OCR doesn't extract English names, parse the
//  email local part for common business patterns.
// ═══════════════════════════════════════════════════════
function inferEnglishNameFromEmail() {
    // Only run if both English name fields are empty
    const givenEn = givenNameEnField.value.trim();
    const familyEn = familyNameEnField.value.trim();
    if (givenEn || familyEn) return;

    const email = emailField.value.trim();
    if (!email || !email.includes('@')) return;

    const localPart = email.split('@')[0].toLowerCase();

    // Skip if local part looks like pure numbers or generic (info, admin, sales, etc.)
    const genericPrefixes = ['info', 'admin', 'sales', 'support', 'contact', 'service',
        'hello', 'office', 'hr', 'marketing', 'help', 'noreply', 'no-reply', 'webmaster'];
    if (/^\d+$/.test(localPart) || genericPrefixes.includes(localPart)) return;

    // Attempt to split by common separators: . _ -
    let parts = localPart.split(/[.\-_]+/).filter(p => p.length > 0);

    // Filter out pure-number segments (e.g. rick.wu123 → ['rick', 'wu'])
    parts = parts.map(p => p.replace(/\d+$/g, '')).filter(p => p.length > 0);

    // Only pure-alpha parts count as name candidates
    parts = parts.filter(p => /^[a-z]+$/i.test(p));

    let firstName = '';
    let lastName = '';

    if (parts.length >= 2) {
        // Most common pattern: firstname.lastname or first_last
        firstName = parts[0];
        lastName = parts[parts.length - 1];
    } else if (parts.length === 1) {
        const single = parts[0];
        // Check if Chinese name is available to help determine last name
        const familyCn = familyNameCnField.value.trim();
        if (single.length <= 3) {
            // Too short to reliably split (e.g. "rwu", "jli") — leave as given name only
            firstName = single;
        } else {
            // Single long word — just set as given name, user can adjust
            firstName = single;
        }
    }

    if (firstName) {
        // Capitalize first letter
        givenNameEnField.value = firstName.charAt(0).toUpperCase() + firstName.slice(1);
        console.log(`[Name Infer] Inferred English given name "${givenNameEnField.value}" from email "${email}"`);
    }
    if (lastName) {
        familyNameEnField.value = lastName.charAt(0).toUpperCase() + lastName.slice(1);
        console.log(`[Name Infer] Inferred English family name "${familyNameEnField.value}" from email "${email}"`);
    }
}

function fillFormFromObj(obj) {
    for (const f of OCR_FIELDS) {
        const el = getFieldElement(f.key);
        if (el && obj[f.key]) el.value = obj[f.key];
    }
    if (!postalCodeEnField.value && postalCodeCnField.value) {
        postalCodeEnField.value = postalCodeCnField.value;
    }
    // If no English name was extracted by OCR, try to infer from email
    inferEnglishNameFromEmail();
}

function getFormData() {
    const data = {};
    for (const f of OCR_FIELDS) {
        const el = getFieldElement(f.key);
        if (el) data[f.key] = el.value;
    }
    return data;
}

// ═══════════════════════════════════════════════════════
//  Comparison UI Logic
// ═══════════════════════════════════════════════════════
function initComparisonUI() {
    const banner = document.getElementById('llmResultBanner');
    const modal = document.getElementById('llmComparisonModal');
    const closeBtn = document.getElementById('closeComparisonBtn');
    const applyAllBtn = document.getElementById('applyAllLlmBtn');
    
    if (banner) {
        banner.addEventListener('click', () => {
            openComparisonModal();
            banner.classList.remove('pulse-animation');
            banner.classList.add('hidden');
        });
    }
    
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.classList.add('hidden');
        });
    }
    
    if (applyAllBtn) {
        applyAllBtn.addEventListener('click', () => {
            if (window.llmExtractedData) {
                fillFormFromObj(window.llmExtractedData);
                modal.classList.add('hidden');
            }
        });
    }
}

document.addEventListener('DOMContentLoaded', initComparisonUI);

function openComparisonModal() {
    const modal = document.getElementById('llmComparisonModal');
    const listContainer = document.getElementById('comparisonList');
    if (!modal || !listContainer) return;
    
    window.currentFormData = getFormData();
    listContainer.innerHTML = '';
    
    const llmData = window.llmExtractedData || {};
    const currentData = window.currentFormData || {};
    
    for (const field of OCR_FIELDS) {
        const llmVal = llmData[field.key] || '';
        const curVal = currentData[field.key] || '';
        
        // Skip if both are empty or if they are identical
        if (!llmVal && !curVal) continue;
        
        const row = document.createElement('div');
        row.className = 'comparison-row';
        if (llmVal !== curVal && llmVal) {
            row.classList.add('diff-row');
        }
        
        row.innerHTML = `
            <div class="comp-label">${field.label}</div>
            <div class="comp-value cur-value">${escapeHtml(curVal) || '<span class="empty-val">空</span>'}</div>
            <div class="comp-value llm-value">${escapeHtml(llmVal) || '<span class="empty-val">空</span>'}</div>
            <div class="comp-action">
                ${(llmVal && llmVal !== curVal) ? `<button class="apply-btn" data-key="${field.key}">套用</button>` : ''}
            </div>
        `;
        listContainer.appendChild(row);
    }
    
    // Add event listeners to apply buttons
    const applyBtns = listContainer.querySelectorAll('.apply-btn');
    applyBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const key = e.target.getAttribute('data-key');
            const el = getFieldElement(key);
            if (el && llmData[key]) {
                el.value = llmData[key];
                
                // Visual feedback
                const row = e.target.closest('.comparison-row');
                row.querySelector('.cur-value').textContent = llmData[key];
                row.classList.remove('diff-row');
                e.target.remove();
            }
        });
    });
    
    modal.classList.remove('hidden');
}

// Deprecated table functions removed


// ═══════════════════════════════════════════════════════
//  Parsing
// ═══════════════════════════════════════════════════════

function mergeBackOCRWithLLM(parsedJson) {
    if (!parsedJson || Object.keys(parsedJson).length === 0) return;

    for (const f of OCR_FIELDS) {
        const el = getFieldElement(f.key);
        if (el && parsedJson[f.key]) {
            // Only fill if the current field is empty
            if (!el.value || el.value.trim() === '') {
                el.value = parsedJson[f.key].trim();
            }
        }
    }

    if (!postalCodeEnField.value && postalCodeCnField.value) {
        postalCodeEnField.value = postalCodeCnField.value;
    }
}

// Helper: check if a string is predominantly Chinese
function isChinese(str) {
    const cleaned = str.replace(/[\s\d\W]/g, '');
    if (!cleaned) return false;
    const cnChars = (cleaned.match(/[\u4e00-\u9fa5]/g) || []).length;
    return cnChars / cleaned.length > 0.5;
}

// Helper: check if a string is predominantly English/Latin
function isEnglish(str) {
    const cleaned = str.replace(/[\s\d\W]/g, '');
    if (!cleaned) return false;
    const enChars = (cleaned.match(/[a-zA-Z]/g) || []).length;
    return enChars / cleaned.length > 0.5;
}

// Helper: extract postal code from an address string
// Supports Taiwan 3-digit/5-digit codes and international codes
function extractPostalCode(addressStr) {
    // Taiwan postal code: 3 to 6 digits at the start
    const twMatch = addressStr.match(/^(\d{3,6})[\s,，]?/);
    if (twMatch) {
        return {
            postalCode: twMatch[1],
            address: addressStr.replace(twMatch[0], '').trim()
        };
    }
    // International: postal/zip code patterns at end or embedded
    const intMatch = addressStr.match(/[,，\s]+(\d{4,6})\s*$/)
        || addressStr.match(/\b(\d{5}(?:-\d{4})?)\b/)  // US ZIP
        || addressStr.match(/\b([A-Z]\d[A-Z]\s?\d[A-Z]\d)\b/i); // Canada
    if (intMatch) {
        return {
            postalCode: intMatch[1],
            address: addressStr.replace(intMatch[0], '').trim()
        };
    }
    return { postalCode: '', address: addressStr };
}

function parseContactData(rawText) {
    const lines = rawText.split('\n')
        .map(line => line.trim())
        .filter(line => line.length > 1)
        .filter(line => /[a-zA-Z0-9\u4e00-\u9fa5]/.test(line));

    // Candidate name lines (we'll pick from these later)
    let chineseNameCandidates = [];
    let englishNameCandidates = [];

    // ─── Regex patterns ───
    const emailRegex = /([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)/i;
    const phoneRegex = /(?:\+?\d{1,4}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}/;
    const urlRegex = /(?:https?:\/\/)?(?:www\.)?[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:\/[^\s]*)*/i;

    // ─── Keywords ───
    const titleKeywords = ['ceo', 'cto', 'cfo', 'coo', 'cio', 'cmo', 'president', 'vp', 'vice president',
        'manager', 'director', 'engineer', 'developer', 'consultant', 'founder', 'co-founder',
        'partner', 'owner', 'specialist', 'executive', 'supervisor', 'coordinator', 'lead',
        'analyst', 'architect', 'associate', 'assistant', 'secretary', 'head', 'officer',
        '理', '長', '員', '師', '總裁', '主席', '主任', '代表', '設計', '企劃', '顧問', '特助',
        '負責人', '老闆', '副總', '協理', '經理', '課長', '組長', '襄理', '專員', '秘書', '工程師', '專案'];

    const companyKeywords = ['inc', 'corp', 'llc', 'ltd', 'company', 'solutions', 'technologies',
        'group', 'international', 'enterprise', 'co.',
        '股份有限公司', '有限公司', '公司', '企業', '實業', '廠', '工作室',
        '集團', '商行', '科技', '工業', '銀行', '事務所', '基金會', '協會'];

    const departmentKeywords = ['department', 'dept', 'division', 'team', 'unit', 'section', 'bureau',
        '部', '處', '室', '科', '課', '組', '部門', '中心',
        '研發', '行銷', '業務', '財務', '人資', '法務', '客服',
        '資訊', '技術', '品管', '企劃', '管理'];

    const cnAddressKeywords = ['路', '街', '段', '巷', '弄', '號', '樓', '市', '區', '縣', '鄉', '鎮', '村', '里', '鄰'];
    const enAddressKeywords = ['road', 'rd.', 'rd', 'street', 'st.', 'st', 'avenue', 'ave', 'blvd',
        'lane', 'floor', 'suite', 'building', 'bldg', 'no.', 'sec.',
        'district', 'city', 'county', 'taipei', 'taichung', 'kaohsiung'];

    let email = '';
    let workPhone = '';
    let ext = '';
    let mobile = '';
    let website = '';
    let title = '';
    let company = '';
    let department = '';
    let addressCn = '';
    let addressEn = '';
    let postalCodeCn = '';

    // ═══ Pass 1: Extract structured data (email, phone, URL) ═══
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const lowerLine = line.toLowerCase();

        // Email
        if (!email) {
            const spacelessLine = line.replace(/\s+/g, '');
            const emailMatch = spacelessLine.match(emailRegex);
            if (emailMatch) {
                email = emailMatch[1];
                lines[i] = '';
                continue;
            }
        }

        // Website URL (must not be an email line)
        if (!website && (lowerLine.includes('www.') || lowerLine.includes('http') || lowerLine.includes('.com'))) {
            const urlMatch = line.match(urlRegex);
            if (urlMatch && !urlMatch[0].includes('@')) {
                website = urlMatch[0];
                if (!website.startsWith('http')) website = 'https://' + website;
                lines[i] = '';
                continue;
            }
        }

        // Phone numbers
        const phoneMatch = line.match(phoneRegex);
        if (phoneMatch && !lowerLine.includes('fax') && !lowerLine.includes('傳真')) {
            const isMobile = lowerLine.includes('mob') || lowerLine.includes('cell') || lowerLine.includes('手機') || line.match(/09\d{2}/) || line.match(/\+886\s*9/);

            if (isMobile && !mobile) {
                mobile = phoneMatch[0].replace(phoneMatch[1], '').replace(/(ext|ext\.|分機|#)\s*$/i, '').trim();
                lines[i] = '';
                continue;
            } else if (!isMobile && !workPhone) {
                workPhone = phoneMatch[0].replace(phoneMatch[1], '').replace(/(ext|ext\.|分機|#)\s*$/i, '').trim();
                if (phoneMatch[1]) {
                    ext = phoneMatch[1];
                }
                lines[i] = '';
                continue;
            } else if (!workPhone) {
                // Secondary fallback
                workPhone = phoneMatch[0].replace(phoneMatch[1], '').replace(/(ext|ext\.|分機|#)\s*$/i, '').trim();
                if (phoneMatch[1]) {
                    ext = phoneMatch[1];
                }
                lines[i] = '';
                continue;
            }
        }
    }

    // ═══ Pass 2: Classify remaining lines ═══
    const remainingLines = lines.filter(l => l !== '');

    for (let i = 0; i < remainingLines.length; i++) {
        const line = remainingLines[i];
        if (!line) continue;
        const lowerLine = line.toLowerCase();

        // Chinese address
        if (!addressCn && cnAddressKeywords.some(kw => line.includes(kw))) {
            const result = extractPostalCode(line);
            addressCn = result.address;
            postalCodeCn = result.postalCode;
            remainingLines[i] = '';
            continue;
        }

        // English address
        if (!addressEn && enAddressKeywords.some(kw => lowerLine.includes(kw))) {
            const result = extractPostalCode(line);
            addressEn = result.address;
            if (result.postalCode && !postalCodeCn) postalCodeCn = result.postalCode;
            remainingLines[i] = '';
            continue;
        }

        // Department (check before title since some keywords overlap)
        if (!department && departmentKeywords.some(kw => lowerLine.includes(kw))) {
            // Exclude if it also matches company keywords (e.g. "科技公司研發部")
            if (!companyKeywords.some(kw => lowerLine.includes(kw))) {
                department = line;
                remainingLines[i] = '';
                continue;
            }
        }

        // Title/Position
        if (!title && titleKeywords.some(kw => lowerLine.includes(kw))) {
            title = line;
            remainingLines[i] = '';
            continue;
        }

        // Company
        if (!company && companyKeywords.some(kw => lowerLine.includes(kw))) {
            // Check if line also contains department info (e.g. "科技公司 研發部")
            if (!department) {
                for (const deptKw of departmentKeywords) {
                    const idx = line.indexOf(deptKw);
                    if (idx > 0) {
                        // Try to split company and department
                        const possibleDept = line.substring(idx).trim();
                        const possibleCompany = line.substring(0, idx).trim();
                        if (possibleCompany.length > 0 && possibleDept.length > 0) {
                            company = possibleCompany;
                            department = possibleDept;
                            remainingLines[i] = '';
                            break;
                        }
                    }
                }
                if (!company) {
                    company = line;
                    remainingLines[i] = '';
                }
            } else {
                company = line;
                remainingLines[i] = '';
            }
            continue;
        }
    }

    // ═══ Pass 3: Identify names from leftover lines ═══
    const leftOvers = remainingLines.filter(l => l !== '');

    for (const line of leftOvers) {
        const cleaned = line.replace(/[^\w\s\u4e00-\u9fa5.-]/gi, '').trim();
        if (!cleaned) continue;

        // Pure or mostly Chinese → Chinese name candidate
        if (isChinese(cleaned) && cleaned.length <= 5) {
            chineseNameCandidates.push(cleaned);
        }
        // Pure or mostly English → English name candidate
        else if (isEnglish(cleaned) && cleaned.split(/\s+/).length <= 4 && cleaned.length <= 40) {
            englishNameCandidates.push(cleaned);
        }
        // Mixed line: try to separate Chinese and English parts
        else {
            const cnPart = (cleaned.match(/[\u4e00-\u9fa5]+/g) || []).join('');
            const enPart = (cleaned.match(/[a-zA-Z][a-zA-Z.\s-]*/g) || []).map(s => s.trim()).filter(Boolean).join(' ');

            if (cnPart && cnPart.length <= 5) chineseNameCandidates.push(cnPart);
            if (enPart && enPart.split(/\s+/).length <= 4) englishNameCandidates.push(enPart);
        }
    }

    // ─── Split Chinese name into family + given ───
    let familyNameCn = '';
    let givenNameCn = '';
    if (chineseNameCandidates.length > 0) {
        const cnName = chineseNameCandidates[0].replace(/\s/g, '');
        if (cnName.length >= 2) {
            familyNameCn = cnName.charAt(0);
            givenNameCn = cnName.substring(1);
        } else {
            familyNameCn = cnName;
        }
    }

    // ─── Split English name into first + last ───
    let givenNameEn = '';
    let familyNameEn = '';
    if (englishNameCandidates.length > 0) {
        const enName = englishNameCandidates[0].trim();
        const parts = enName.split(/\s+/);
        if (parts.length > 1) {
            familyNameEn = parts[parts.length - 1];
            givenNameEn = parts.slice(0, -1).join(' ');
        } else {
            givenNameEn = enName;
        }
    }

    // ─── Fallback: if no title/company found, use remaining leftover lines ───
    const unusedLeftOvers = leftOvers.filter(l => {
        const cleaned = l.replace(/[^\w\s\u4e00-\u9fa5.-]/gi, '').trim();
        return cleaned &&
            !chineseNameCandidates.includes(cleaned) &&
            !englishNameCandidates.includes(cleaned);
    });

    if (!title && unusedLeftOvers.length > 0) {
        title = unusedLeftOvers.shift();
    }
    if (!company && unusedLeftOvers.length > 0) {
        company = unusedLeftOvers.shift();
    }
    if (!department && unusedLeftOvers.length > 0) {
        department = unusedLeftOvers.shift();
    }

    // 純函式：只回傳解析結果，絕不直接寫表單。
    // 過去這裡會把 16 個欄位全部覆寫（含空字串），導致背景 AI 解析的
    // fallback（parseTextToJsonViaLLM 在 LLM 回應非乾淨 JSON 時呼叫本
    // 函式解析「LLM 回應文字」）在數十秒後把已填好的表單整片清空。
    // 填表一律由呼叫端的 fillFormFromObj / mergeBackOCRWithLLM 決定
    // （兩者都只寫非空值）。
    return {
        familyNameCn, givenNameCn, givenNameEn, familyNameEn,
        company, department, title, taxId: '',
        workPhone, ext, mobile, email, website,
        addressCn, postalCodeCn, addressEn,
    };
}


// ═══════════════════════════════════════════════════════
//  Photo Resizing
// ═══════════════════════════════════════════════════════
// ─── 等比例縮圖（只縮不放、保留長寬比） ───
function resizeCardImage(dataUrl, maxDim = 640, quality = 0.8) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            let w = img.width, h = img.height;
            if (Math.max(w, h) > maxDim) {
                const s = maxDim / Math.max(w, h);
                w = Math.round(w * s);
                h = Math.round(h * s);
            }
            const canvas = document.createElement('canvas');
            canvas.width = w;
            canvas.height = h;
            canvas.getContext('2d').drawImage(img, 0, 0, w, h);
            resolve(canvas.toDataURL('image/jpeg', quality));
        };
        img.onerror = () => reject(new Error('影像載入失敗'));
        img.src = dataUrl;
    });
}


// ═══════════════════════════════════════════════════════
//  vCard Line Folding
// ═══════════════════════════════════════════════════════
// ─── vCard 2.1 值跳脫：compound 欄位（N/ORG/ADR）內的反斜線與分號 ───
function escape21(value) {
    if (!value) return '';
    return String(value).replace(/\\/g, '\\\\').replace(/;/g, '\\;');
}

// ─── vCard 2.1 Quoted-Printable：UTF-8 → =XX，過長以 =CRLF 軟斷行 ───
//     整檔因此為純 ASCII，Windows 聯絡人/Outlook 不會判錯字碼
//     （規則移植自 vcf_to_win.py 的 qp_encode）
function qpEncode(s, softWrap = 75) {
    let out = '';
    let lineLen = 0;
    for (const byte of new TextEncoder().encode(s)) {
        let tok;
        if (byte === 0x3D) tok = '=3D';                    // '='
        else if (byte === 0x0D) tok = '=0D';               // CR
        else if (byte === 0x0A) tok = '=0A';               // LF
        else if (byte === 0x20) tok = ' ';
        else if (byte >= 33 && byte <= 126) tok = String.fromCharCode(byte);
        else tok = '=' + byte.toString(16).toUpperCase().padStart(2, '0');
        // 軟斷行：不切斷 =XX token
        if (lineLen + tok.length > softWrap) {
            out += '=\r\n';
            lineLen = 0;
        }
        out += tok;
        lineLen += tok.length;
    }
    return out;
}

// ─── 折行：每實體行 ≤ 75 位元組（UTF-8），續行以單一空格開頭 ───
//     以 for...of 逐「完整字元」迭代，避免把中文字/代理對從中切斷
function foldLine(line) {
    const encoder = new TextEncoder();
    if (encoder.encode(line).length <= 75) return line;

    let result = '';
    let current = '';
    for (const ch of line) {
        if (encoder.encode(current + ch).length > 74) {
            result += current + '\r\n ';
            current = ch;
        } else {
            current += ch;
        }
    }
    return result + current;
}

// ─── 產生 2.1 屬性行：純 ASCII 直接輸出（過長折行），含中文改用 QP ───
function vProp21(nameAndParams, value) {
    if (/^[\x20-\x7E]*$/.test(value)) {
        return foldLine(`${nameAndParams}:${value}`);
    }
    return `${nameAndParams};CHARSET=UTF-8;ENCODING=QUOTED-PRINTABLE:${qpEncode(value)}`;
}

// ─── 照片：vCard 2.1 BASE64 區塊（每行 74 字元、空白開頭、空行結尾） ───
//     （格式移植自 vcf_to_win.py 的 photo_block）
function photoBlock21(nameAndParams, b64) {
    const lines = [`${nameAndParams}:`];
    for (let i = 0; i < b64.length; i += 74) {
        lines.push(' ' + b64.slice(i, i + 74));
    }
    lines.push('');   // 2.1 規定 BASE64 屬性以空白行結束
    return lines;
}


// ═══════════════════════════════════════════════════════
//  Generate & Download vCard (.vcf)
// ═══════════════════════════════════════════════════════
async function exportVCF() {
    const familyNameCn = familyNameCnField.value.trim();
    const givenNameCn = givenNameCnField.value.trim();
    const givenNameEn = givenNameEnField.value.trim();
    const familyNameEn = familyNameEnField.value.trim();
    const titleVal = titleField.value.trim();
    const taxIdVal = taxIdField.value.trim();
    const company = companyField.value.trim();
    const department = departmentField.value.trim();
    const workPhone = workPhoneField.value.trim();
    const ext = extField.value.trim();
    const mobile = mobileField.value.trim();
    const email = emailField.value.trim();
    const website = websiteField.value.trim();
    const addrCn = addressCnField.value.trim();
    const zipShared = postalCodeCnField.value.trim() || postalCodeEnField.value.trim();
    const addrEn = addressEnField.value.trim();
    const note = noteField.value.trim();
    const categories = categoriesField.value.trim();

    // Build composite name for N and FN
    // Prefer Chinese name; fall back to English
    const familyName = familyNameCn || familyNameEn;
    const givenName = givenNameCn || givenNameEn;
    const fullNameCn = familyNameCn + givenNameCn;
    const fullNameEn = [givenNameEn, familyNameEn].filter(Boolean).join(' ');
    const fullName = fullNameCn || fullNameEn || familyName + givenName;

    if (!fullName && !mobile && !workPhone && !email) {
        alert('請至少填寫姓名、電話或 Email 其中一項再匯出。');
        return;
    }

    // vCard 2.1 + Quoted-Printable：repo 內 vcf_to_win.py 驗證過的
    // Windows 聯絡人/Outlook 相容格式；Google 聯絡人同樣可匯入。
    const lines = [];
    lines.push('BEGIN:VCARD');
    lines.push('VERSION:2.1');

    // N：姓;名;中間名;稱謂;後綴（compound 欄位個別跳脫分號）
    lines.push(vProp21('N', `${escape21(familyName)};${escape21(givenName)};;;`));

    // FN: Full formatted name — combine both CN and EN if available
    let displayName = fullNameCn;
    if (fullNameEn) {
        displayName = displayName ? `${displayName} (${fullNameEn})` : fullNameEn;
    }
    lines.push(vProp21('FN', displayName));

    // ORG：公司;部門
    if (company || department) {
        lines.push(vProp21('ORG',
            [company, department].filter(Boolean).map(escape21).join(';')));
    }

    if (titleVal) lines.push(vProp21('TITLE', titleVal));
    if (mobile) lines.push(vProp21('TEL;TYPE=CELL,VOICE', mobile));
    if (workPhone) {
        // Outlook 可辨識「號碼 x分機」的寫法
        const fullWorkPhone = ext ? `${workPhone} x${ext}` : workPhone;
        lines.push(vProp21('TEL;TYPE=WORK,VOICE', fullWorkPhone));
    }
    if (email) lines.push(vProp21('EMAIL;TYPE=INTERNET,WORK', email));
    if (website) lines.push(vProp21('URL;TYPE=WORK', website));

    // ADR：;;街道;市;州;郵遞區號;國家
    if (addrCn) {
        lines.push(vProp21('ADR;TYPE=WORK', `;;${escape21(addrCn)};;;${escape21(zipShared)};`));
    }
    if (addrEn) {
        lines.push(vProp21('ADR;TYPE=WORK', `;;${escape21(addrEn)};;;${escape21(zipShared)};`));
    }

    // NOTE：統編 + 備註（換行由 QP 編成 =0A，不需 3.0 的 \n 跳脫）
    let noteContent = note;
    if (taxIdVal) {
        noteContent = noteContent ? `統編: ${taxIdVal}\n${noteContent}` : `統編: ${taxIdVal}`;
    }
    if (noteContent) lines.push(vProp21('NOTE', noteContent));

    if (categories) lines.push(vProp21('CATEGORIES', categories));

    // ─── 影像：主相片優先用透視校正後的名片圖，以 2.1 BASE64 區塊嵌入 ───
    let photoB64 = null;
    if (window.cardLogoBase64) {
        photoB64 = window.cardLogoBase64;               // 後端已壓到 <100KB
    } else if (previewImage.src && previewImage.src.startsWith('data:image/')) {
        try {
            const resized = await resizeCardImage(previewImage.src, 640, 0.8);
            photoB64 = resized.split(',')[1];
        } catch (e) {
            console.warn('無法縮放正面影像:', e);
        }
    }
    if (photoB64) {
        lines.push(...photoBlock21('PHOTO;ENCODING=BASE64;TYPE=JPEG', photoB64));
    }

    // 背面照：第二個 PHOTO 加 X-ABLABEL 參數（vcf_to_win.py 同款寫法）
    if (backPhotoDataUrl) {
        try {
            const resizedBack = await resizeCardImage(backPhotoDataUrl, 640, 0.8);
            const backB64 = resizedBack.split(',')[1];
            lines.push(...photoBlock21('PHOTO;ENCODING=BASE64;TYPE=JPEG;X-ABLABEL=Back', backB64));
        } catch (e) {
            console.warn('無法縮放背面影像:', e);
        }
    }

    lines.push('END:VCARD');

    // 整檔純 ASCII（中文都在 QP 的 =XX 裡），任何字碼頁都不會誤判，
    // 因此不需要 BOM。結尾補一組 CRLF。
    const vcfString = lines.join('\r\n') + '\r\n';

    const blob = new Blob([vcfString], { type: 'text/vcard;charset=utf-8' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    const filename = fullName ? `${fullName.replace(/\s+/g, '_')}_contact.vcf` : 'contact.vcf';
    link.setAttribute('download', filename);

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showSection(successSection);
}
