// ═══════════════════════════════════════════════════════
//  CardScanner — app.js
//  Camera, Zoom, Flip, Perspective Crop, Docker Model Runner OCR, vCard Export
//  OCR via Docker Model Runner (OpenAI-compatible API)
// ═══════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════
//  Docker Model Runner & PaddleOCR API Helpers
// ═══════════════════════════════════════════════════════
const DMR_BASE = 'http://localhost:12435/v1';  // CORS proxy → Ollama :11434 (OpenAI-compatible API)
const CV_BACKEND_BASE = 'http://localhost:5050';



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
    if (!res.ok) {
        // Capture the response body — Ollama explains *why* (e.g. model has no
        // vision support, unsupported response_format) here, not in statusText.
        let detail = '';
        try { detail = await res.text(); } catch (e) { /* ignore */ }
        try { detail = JSON.parse(detail)?.error?.message || JSON.parse(detail)?.error || detail; } catch (e) { /* keep raw */ }
        throw new Error(`Ollama API Error: ${res.status} ${res.statusText}${detail ? ' — ' + detail : ''}`);
    }
    const data = await res.json();
    return data.choices?.[0]?.message?.content || '';
}

// Robustly parse JSON from an LLM response: strips markdown fences,
// extracts the first {...} block, and removes trailing commas.
function parseLLMJson(raw) {
    if (raw == null) throw new Error('EMPTY_RESPONSE');
    let s = String(raw).trim();
    if (!s) throw new Error('EMPTY_RESPONSE');
    // Strip markdown code fences (```json ... ``` or ``` ... ```)
    s = s.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/i, '').trim();
    // Keep only the outermost {...} block, dropping any prose before/after
    const start = s.indexOf('{');
    const end = s.lastIndexOf('}');
    if (start !== -1 && end !== -1 && end > start) {
        s = s.slice(start, end + 1);
    }
    // Remove trailing commas before } or ] which break JSON.parse
    s = s.replace(/,\s*([}\]])/g, '$1');
    return JSON.parse(s);
}

function imageDataUrlToBase64(dataUrl) {
    return dataUrl.split(',')[1] || '';
}

// ─── Ollama memory management ───
// Ollama keeps every used model resident for ~5 min (keep_alive). Leftover
// models from earlier runs compete for RAM/VRAM with the current model and
// cause heavy swapping. These helpers evict them before a heavy run.
const OLLAMA_BASE = DMR_BASE.replace(/\/v1\/?$/, '');

// List models currently resident in memory (Ollama native /api/ps).
async function listLoadedModels() {
    try {
        const r = await fetch(`${OLLAMA_BASE}/api/ps`, {
            headers: { 'Authorization': 'Bearer local-key' }
        });
        if (!r.ok) return [];
        const data = await r.json();
        return (data.models || []).map(m => m.name || m.model).filter(Boolean);
    } catch (e) {
        console.warn('listLoadedModels failed (non-fatal):', e);
        return [];
    }
}

// Unload one model from memory immediately (keep_alive: 0).
async function unloadModel(model) {
    if (!model) return;
    try {
        await fetch(`${OLLAMA_BASE}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer local-key' },
            body: JSON.stringify({ model, keep_alive: 0 })
        });
        console.log('Unloaded model from memory:', model);
    } catch (e) {
        console.warn('unloadModel failed (non-fatal):', model, e);
    }
}

// Free memory before a heavy run: unload every resident model except the
// one we're about to use. Best-effort; never blocks the scan on failure.
async function unloadModelsExcept(keepModel) {
    const loaded = await listLoadedModels();
    const toUnload = loaded.filter(m => m && m !== keepModel);
    if (toUnload.length === 0) return;
    console.log('Evicting leftover models:', toUnload.join(', '));
    await Promise.all(toUnload.map(unloadModel));
}



// XSS protection for dynamic HTML
function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
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
const ollamaModelSelect = document.getElementById('ollamaModelSelect');

// Live OCR Display & Engine Status
const liveOCRText = document.getElementById('liveOCRText');
const singleOcrStatus = document.getElementById('singleOcrStatus');
const activeModelLabel = document.getElementById('activeModelLabel');
const activeModelStatus = document.getElementById('activeModelStatus');


// Sections
const cameraSection = document.getElementById('cameraSection');
const cropSection = document.getElementById('cropSection');
const processingSection = document.getElementById('processingSection');
const resultSection = document.getElementById('resultSection');
const successSection = document.getElementById('successSection');
const backCameraSection = document.getElementById('backCameraSection');

// Text LLM Dropdown removed
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
// Actions
const retakeBtn = document.getElementById('retakeBtn');
const exportBtn = document.getElementById('exportBtn');
const scanAnotherBtn = document.getElementById('scanAnotherBtn');
const contactForm = document.getElementById('contactForm');


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

    // 記住使用者選的模型,重新整理後由 populateModels 恢復
    if (ollamaModelSelect) {
        ollamaModelSelect.addEventListener('change', () => {
            localStorage.setItem(MODEL_STORE_KEY, ollamaModelSelect.value);
        });
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

    // Re-crop buttons
    if (manualCropBtn) {
        manualCropBtn.addEventListener('click', async () => {
            if (!capturedImageDataUrl) return;
            recropMode = true; // re-crop: update image only, don't re-run OCR
            let corners = null;
            try { corners = await detectCardFetch(capturedImageDataUrl); } catch (e) { /* ignore */ }
            openCropSection(capturedImageDataUrl, corners);
        });
    }
    if (manualBackCropBtn) {
        manualBackCropBtn.addEventListener('click', () => {
            if (backCapturedImageDataUrl) openBackCropSection(backCapturedImageDataUrl);
        });
    }

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

    // Back camera Flip
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
});


// ═══════════════════════════════════════════════════════
//  Camera Enumeration
// ═══════════════════════════════════════════════════════
async function populateCameraList(forceRequest = false) {
    try {
        // If we want to ensure labels are visible, we might need a dummy getUserMedia 
        // if no stream is active, but usually enumerateDevices picks up labels after first permission.
        const currentSelected = cameraSelect.value;
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devices.filter(d => d.kind === 'videoinput');
        
        console.log(`[Camera Debug] Found ${videoInputs.length} video inputs. Permissions: ${videoInputs.some(v => v.label) ? 'Granted' : 'Denied/Wait'}`);

        cameraSelect.innerHTML = '';
        videoInputs.forEach((device, index) => {
            const opt = document.createElement('option');
            opt.value = device.deviceId;
            // Use label if available, otherwise generic
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

// 記住使用者選過的視覺模型;無記錄時的預設優先序。
// qwen3.5 視覺推論在本機 Ollama 0.30.7 會崩(第二次請求 runner 終止),
// 故預設鎖定已驗證穩定的 qwen2.5vl 系列,不能拿清單第一個當預設。
const MODEL_STORE_KEY = 'scannerVisionModel';
const PREFERRED_VISION_MODELS = ['qwen2.5vl:7b', 'qwen2.5vl:3b', 'llama3.2-vision'];

async function populateModels() {
    try {
        const res = await fetch(`${DMR_BASE}/models`, {
            headers: { 'Authorization': 'Bearer local-key' }
        });
        if (res.ok) {
            const data = await res.json();
            // Docker Model Runner returns { data: [{id, ...}, ...] }
            // Strip 'docker.io/' prefix for cleaner display while keeping full id for API calls
            const models = (data.data || []).map(m => {
                const cleanId = m.id.replace(/^docker\.io\//, '');
                return { name: cleanId };
            });
            window.availableModels = models;

            if (ollamaModelSelect) {
                const filteredModels = models.filter(m => !m.name.includes('embed'));
                ollamaModelSelect.innerHTML = '';
                const defaultOpt = document.createElement('option');
                defaultOpt.value = '';
                defaultOpt.textContent = '不使用模型 (手動輸入)';
                ollamaModelSelect.appendChild(defaultOpt);

                filteredModels.forEach(model => {
                    const opt = document.createElement('option');
                    opt.value = model.name;
                    opt.textContent = model.name;
                    ollamaModelSelect.appendChild(opt);
                });

                if (filteredModels.length > 0) {
                    // 依序:使用者上次的選擇(含「不使用模型」)→ 已知穩定的
                    // 視覺模型 → 清單第一個。Ollama 清單「最新下載排最前」,
                    // 若無條件用第一個,新 pull 的模型會靜默蓋掉使用者的選擇。
                    const stored = localStorage.getItem(MODEL_STORE_KEY);
                    if (stored !== null && (stored === '' || filteredModels.some(m => m.name === stored))) {
                        ollamaModelSelect.value = stored;
                    } else {
                        ollamaModelSelect.value =
                            PREFERRED_VISION_MODELS.find(p => filteredModels.some(m => m.name === p)) ||
                            filteredModels[0].name;
                    }
                }
            }
        }
    } catch (err) {
        console.error('Failed to fetch Ollama models:', err);
        if (ollamaModelSelect) ollamaModelSelect.innerHTML = '<option value="">Ollama 或 Proxy 未連線</option>';
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
        const constraints = {
            video: {
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            }
        };

        if (selectedDeviceId) {
            constraints.video.deviceId = { ideal: selectedDeviceId };
        } else {
            // Prefer back camera if none selected
            constraints.video.facingMode = "environment";
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
async function detectCardFetch(imageDataUrl) {
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
        }
    } catch (err) {
        console.error('CV Backend Error:', err);
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

    if (!imageData) return;

    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> 偵測中...';

    const corners = await detectCardFetch(imageData);

    if (corners) {
        if (isBack) {
            backCropCorners = corners;
            updateBackCropHandles();
        } else {
            cropCorners = corners;
            updateCropHandles();
        }
    } else {
        alert('未偵測到明顯邊緣，請手動調整。');
    }

    btn.disabled = false;
    btn.innerHTML = originalText;
}

// When true, confirmCrop() updates the card image and returns to the result
// page WITHOUT re-running OCR (used by the "重新裁切" button on the result page).
let recropMode = false;

/**
 * After capture: open the same crop screen the result page uses, pre-filled
 * with auto-detected corners (or a 10% default the user can drag). This mirrors
 * the result-page re-crop, which produces reliable crops. The user confirms
 * with 確認裁切, then OCR runs.
 */
async function autoCycleCrop(imageDataUrl, isBack) {
    console.log(`[Auto-Crop] Detecting card edges for ${isBack ? 'Back' : 'Front'}...`);
    recropMode = false; // first-time capture → OCR should run after the crop

    let corners = null;
    try {
        corners = await detectCardFetch(imageDataUrl);
    } catch (err) {
        console.error('[Auto-Crop] Detection error:', err);
    }
    // Use whatever the detector returns (even a frame-filling card is a valid
    // result). Only fall back to the default box when detection returns nothing.
    if (corners) {
        console.log('[Auto-Crop] Using detected corners as crop start.');
    } else {
        console.warn('[Auto-Crop] No corners (detector returned null or backend down) — using default box.');
    }

    if (isBack) {
        openBackCropSection(imageDataUrl, corners);
    } else {
        openCropSection(imageDataUrl, corners);
    }
}

// ═══════════════════════════════════════════════════════
//  Manual Crop & Perspective Correction
// ═══════════════════════════════════════════════════════
async function openCropSection(imageDataUrl, initialCorners) {
    showSection(cropSection);

    // Use auto-detected corners when provided; otherwise a 10% margin default.
    cropCorners = initialCorners || {
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

                // Re-crop from the result page: just update the card image and
                // go back — do NOT re-run OCR (keep the recognized fields).
                if (recropMode) {
                    recropMode = false;
                    progressFill.style.width = '100%';
                    progressText.textContent = '裁切已更新';
                    showSection(resultSection);
                    return;
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
        // Re-crop mode: don't fall through to OCR; keep existing fields.
        if (recropMode) {
            recropMode = false;
            alert('透視校正失敗，保留原本的辨識結果。');
            showSection(resultSection);
            return;
        }
        // First-time capture: fall back to OCR on the original image.
        alert('透視校正失敗，將使用原始影像進行辨識。');
        skipCrop();
    }
}



function skipCrop() {
    if (!capturedImageDataUrl) return;
    // Re-crop mode: skipping should return to the result page without re-OCR.
    if (recropMode) {
        recropMode = false;
        showSection(resultSection);
        return;
    }
    showSection(processingSection);
    performOCR(capturedImageDataUrl);
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
async function openBackCropSection(imageDataUrl, initialCorners) {
    showSection(backCropSection);

    // Use auto-detected corners when provided; otherwise a 10% margin default.
    backCropCorners = initialCorners || {
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

// 比照正面 confirmCrop()：把影像與四角座標送後端 /api/morph-card 做
// OpenCV 透視校正；失敗時退回原圖辨識（與正面的 fallback 行為一致）。
async function confirmBackCrop() {
    if (!backCapturedImageDataUrl) return;

    try {
        showSection(backProcessingSection);
        backProgressText.textContent = '正在進行背面透視校正與影像優化...';

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
                performBackOCR(correctedUrl);
                return;
            }
        }
        throw new Error('Morphing failed');
    } catch (err) {
        console.error('Back Perspective Transform Error:', err);
        alert('背面透視校正失敗，將使用原始影像進行辨識。');
        skipBackCrop();
    }
}

function skipBackCrop() {
    if (!backCapturedImageDataUrl) return;
    showSection(backProcessingSection);
    backPhotoDataUrl = backCapturedImageDataUrl;
    showBackPhotoPreview(backCapturedImageDataUrl);
    performBackOCR(backCapturedImageDataUrl);
}

// ─── Back OCR ───
async function performBackOCR(imageDataUrl) {
    showSection(backProcessingSection);
    try {
        const b64 = imageDataUrlToBase64(imageDataUrl);
        const visionModel = (ollamaModelSelect && ollamaModelSelect.value) || 'llama3.2-vision';
        
        const backStartTime = Date.now();
        const backTimer = setInterval(() => {
            const elapsed = ((Date.now() - backStartTime) / 1000).toFixed(0);
            backProgressText.textContent = `使用 ${visionModel} 讀取背面資料中... (${elapsed}s)`;
        }, 1000);

        backProgressFill.style.width = '50%';

        const systemPrompt = `You are a strict data extraction assistant. Extract all contact information from the provided business card image.
Return ONLY a valid JSON object with EXACTLY these flat matching fields. Use empty string "" if not found.

{
  "familyNameCn": "",
  "givenNameCn": "",
  "givenNameEn": "",
  "familyNameEn": "",
  "company": "",
  "department": "",
  "title": "",
  "taxId": "",
  "workPhone": "",
  "ext": "",
  "mobile": "",
  "email": "",
  "website": "",
  "addressCn": "",
  "postalCodeCn": "",
  "addressEn": ""
}`;

        const userPrompt = `Extract contact details from this business card back into JSON.`;

        let jsonResult = {};
        let llmResp = '';
        // Free RAM/VRAM: evict any other models left resident from earlier runs
        await unloadModelsExcept(visionModel);
        // Step 1: call the model (network / model-load / OOM errors surface here)
        try {
            llmResp = await dmrChat(visionModel, userPrompt, b64, {
                systemPrompt: systemPrompt,
                temperature: 0.2,
                format: 'json'
            });
        } catch (apiErr) {
            console.error('Vision API call failed:', apiErr);
            throw new Error(`呼叫模型 ${visionModel} 失敗:${apiErr.message}`);
        }

        // Step 2: parse the JSON
        try {
            jsonResult = parseLLMJson(llmResp);
        } catch (jsonErr) {
            console.warn('LLM JSON parsing failed. Raw output:', llmResp, jsonErr);
            const preview = (llmResp || '').trim().slice(0, 80) || '(空白)';
            throw new Error(`模型有回應但不是有效的 JSON,可能多了說明文字或被截斷。原始輸出開頭:「${preview}」`);
        }

        clearInterval(backTimer);
        backProgressFill.style.width = '100%';
        backProgressText.textContent = '背面掃描完成';

        mergeBackOCRWithLLM(jsonResult);

        setTimeout(() => {
            showSection(resultSection);
        }, 500);
    } catch (err) {
        console.error('Back OCR Error:', err);
        alert(`背面辨識發生錯誤。\n${err.message || ''}`);
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
    backPhotoPreview.classList.add('hidden');
    backPreviewImg.src = '';
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
    removeBackPhoto();
}


// ═══════════════════════════════════════════════════════
//  OCR — Dual Engine via Docker Model Runner
// ═══════════════════════════════════════════════════════

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
        const b64 = imageDataUrlToBase64(imageDataUrl);
        const visionModel = (ollamaModelSelect && ollamaModelSelect.value) || 'llama3.2-vision';

        progressFill.style.width = '30%';
        progressText.textContent = `傳送圖片至 ${visionModel} 中...`;

        // Show engine status
        if (singleOcrStatus) {
            singleOcrStatus.classList.remove('hidden');
            if (activeModelLabel) activeModelLabel.textContent = `${visionModel}:`;
            if (activeModelStatus) {
                activeModelStatus.textContent = '辨識中...';
                activeModelStatus.style.color = '';
            }
        }

        const startTime = Date.now();
        const timer = setInterval(() => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);
            progressText.textContent = `分析影像內容中 (${visionModel})... (${elapsed}s)`;
        }, 1000);

        const systemPrompt = `You are a strict data extraction assistant. Extract all contact information from the provided business card image.
Return ONLY a valid JSON object with EXACTLY these flat matching fields. Use empty string "" if not found.

{
  "familyNameCn": "",
  "givenNameCn": "",
  "givenNameEn": "",
  "familyNameEn": "",
  "company": "",
  "department": "",
  "title": "",
  "taxId": "",
  "workPhone": "",
  "ext": "",
  "mobile": "",
  "email": "",
  "website": "",
  "addressCn": "",
  "postalCodeCn": "",
  "addressEn": ""
}`;

        const userPrompt = `Extract contact details from this business card into JSON.`;
        
        let visionJson = {};
        let resp = '';
        // Free RAM/VRAM: evict any other models left resident from earlier runs
        progressText.textContent = `釋放其他模型記憶體中...`;
        await unloadModelsExcept(visionModel);
        // Step 1: call the model (network / model-load / OOM errors surface here)
        try {
            resp = await dmrChat(visionModel, userPrompt, b64, {
                systemPrompt: systemPrompt,
                temperature: 0.2,
                format: 'json'
            });
        } catch (apiErr) {
            console.error('Vision API call failed:', apiErr);
            throw new Error(`呼叫模型 ${visionModel} 失敗:${apiErr.message}`);
        }

        // Show raw LLM output in live text area
        if (liveOCRText) liveOCRText.value = resp;

        // Step 2: parse the JSON (formatting issues surface here, not "not a vision model")
        try {
            visionJson = parseLLMJson(resp);
        } catch (jsonErr) {
            console.warn('LLM JSON parsing failed. Raw output:', resp, jsonErr);
            const preview = (resp || '').trim().slice(0, 80) || '(空白)';
            throw new Error(`模型有回應但不是有效的 JSON,可能多了說明文字或被截斷。原始輸出開頭:「${preview}」`);
        }

        clearInterval(timer);

        if (activeModelStatus) {
            activeModelStatus.textContent = '✓ 完成';
            activeModelStatus.style.color = 'var(--success, #10b981)';
        }

        progressFill.style.width = '100%';
        progressText.textContent = '處理完成';

        fillFormFromObj(visionJson);
        showSection(resultSection);
        if (singleOcrStatus) singleOcrStatus.classList.add('hidden');

    } catch (err) {
        console.error('OCR Error:', err);
        alert(`發生錯誤：\n${err.message || ''}`);
        resetToCamera();
    }
}



function mergeResults(primary, secondary) {
    const merged = {};
    for (const f of OCR_FIELDS) {
        merged[f.key] = (primary[f.key] || '').trim() || (secondary[f.key] || '').trim();
    }
    return merged;
}

function fillFormFromObj(obj) {
    for (const f of OCR_FIELDS) {
        const el = getFieldElement(f.key);
        if (el && obj[f.key]) el.value = obj[f.key];
    }
    if (!postalCodeEnField.value && postalCodeCnField.value) {
        postalCodeEnField.value = postalCodeCnField.value;
    }
}

// Merge the back-card LLM JSON into the form, filling ONLY empty fields so the
// front-side data captured earlier is never overwritten.
function mergeBackOCRWithLLM(obj) {
    if (!obj || typeof obj !== 'object') return;
    for (const f of OCR_FIELDS) {
        const el = getFieldElement(f.key);
        const val = obj[f.key];
        if (el && !el.value && val) {
            el.value = String(val).trim();
        }
    }
    if (!postalCodeEnField.value && postalCodeCnField.value) {
        postalCodeEnField.value = postalCodeCnField.value;
    }
}

// Deprecated table functions removed


// ═══════════════════════════════════════════════════════
//  Parsing
// ═══════════════════════════════════════════════════════

// Deprecated legacy parser functions removed.


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
