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

    const stream = typeof opts.onToken === 'function';
    const payload = {
        model,
        messages,
        stream,
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
    if (!stream) {
        const data = await res.json();
        return data.choices?.[0]?.message?.content || '';
    }

    // 串流:解析 SSE(data: {...}),把生成中的文字即時交給 opts.onToken。
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let full = '';
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buffer.indexOf('\n')) !== -1) {
            const line = buffer.slice(0, nl).trim();
            buffer = buffer.slice(nl + 1);
            if (!line.startsWith('data:')) continue;
            const payloadStr = line.slice(5).trim();
            if (payloadStr === '[DONE]') continue;
            try {
                const delta = JSON.parse(payloadStr).choices?.[0]?.delta?.content || '';
                if (delta) { full += delta; opts.onToken(full); }
            } catch (e) { /* 忽略 keep-alive / 不完整的行 */ }
        }
    }
    return full;
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

// Preload the vision model into memory so the real OCR call doesn't pay the
// cold-start weight load (~4s on a 7B). Fire-and-forget: triggered when the
// crop UI opens, so loading overlaps with the seconds the user spends
// adjusting corners. keep_alive 30m keeps it resident between scans.
async function warmUpModel(model) {
    if (!model) return;
    try {
        await fetch(`${OLLAMA_BASE}/api/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer local-key' },
            body: JSON.stringify({ model, prompt: 'hi', stream: false, keep_alive: '30m' })
        });
        console.log('Warmed up model:', model);
    } catch (e) {
        console.warn('warmUpModel failed (non-fatal):', model, e);
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

// ──────────────────────────────────────────────────────────────
//  簡體→繁體單字轉換（由 OpenCC s2t 字典產生，共 2604 組相異字）
//  視覺模型常「照抄」名片上的簡體字形，提示詞要求轉繁不可靠，
//  故對解析出的 JSON 值做確定性後處理。
// ──────────────────────────────────────────────────────────────
const S2T_FROM = '万与丑专业丛东丝丢两严丧个丰临为丽举么义乌乐乔习乡书买乱争于亏云亘亚产亩亲亵亸亿仅仆从仑仓仪们价众优伙会伛伞伟传伡伣伤伥伦伧伪伫体余佣佥侠侣侥侦侧侨侩侪侬侭俣俦俨俩俪俫俭债倾偬偻偾偿傤傥傧储傩儿兑兖党兰关兴兹养兽冁内冈册写军农冯冲决况冻净凄准凉减凑凛几凤凫凭凯凶击凿刍划刘则刚创删别刬刭刹刽刾刿剀剂剐剑剥剧劝办务劢动励劲劳势勋勚匀匦匮区医华协单卖占卢卤卧卫却卺厂厅历厉压厌厍厐厕厘厢厣厦厨厩厮县叁参叆叇双发变叙叠台叶号叹叽吁吃后吓吕吗吨听启吴呐呒呓呕呖呗员呙呛呜咏咙咛咝咤咨咸响哑哒哓哔哕哗哙哜哝哟唇唛唝唠唡唢唤啧啬啭啮啯啰啴啸喷喽喾嗫嗳嘘嘤嘱噜嚣团园囱围囵国图圆圣圹场坏块坚坛坜坝坞坟坠垄垅垆垒垦垩垫垭垯垱垲垴埘埙埚堑堕塆墙壮声壳壶壸处备复够头夸夹夺奁奂奋奖奥妆妇妈妩妪妫姗姹娄娅娆娇娈娱娲娴婳婴婵婶媪媭嫒嫔嫱嬷孙学孪宁宝实宠审宪宫宽宾寝对寻导寿将尔尘尝尧尴尸尽层屃屉届属屡屦屿岁岂岖岗岘岚岛岩岭岳岽岿峃峄峡峣峤峥峦峰崂崃崄崭嵘嵚嵝巅巩巯币帅师帏帐帘帜带帧帮帱帻帼幂干并广庄庆床庐庑库应庙庞废庼廪开异弃弑张弥弪弯弹强归当录彟彦彨彻征径徕忆忏忧忾怀态怂怃怄怅怆怜总怼怿恋恒恳恶恸恹恺恻恼恽悦悫悬悭悮悯惊惧惨惩惫惬惭惮惯愠愤愦愿慑慭懑懒懔戆戋戏戗战戬戯户扑托执扩扪扫扬扰抚抛抟抠抡抢护报担拟拢拣拥拦拧拨择挂挚挛挜挝挞挟挠挡挢挣挤挥挦捝捞损捡换捣据掳掴掷掸掺掼揽揾揿搀搁搂搄搅携摄摅摆摇摈摊撄撑撵撷撸撺擜擞攒敌敚敛敩数斋斓斗斩断无旧时旷旸昙昵昼昽显晋晒晓晔晕晖暂暧术朴机杀杂权杠条来杨杩杰极构枞枢枣枥枧枨枪枫枭柜柠柽栀栅标栈栉栊栋栌栎栏树栖栗样栾桠桡桢档桤桥桦桧桨桩桪梦梼梾梿检棁棂椁椝椟椠椢椤椫椭椮楼榄榅榇榈榉榝槚槛槟槠横樯樱橥橱橹橼檩欢欤欧歼殁殇残殒殓殚殡殴毁毂毕毙毡毵氇气氢氩氲汇汉汤汹沄沟没沣沤沥沦沧沨沩沪泞泪泶泷泸泺泻泼泽泾洁洒洼浃浅浆浇浈浉浊测浍济浏浐浑浒浓浔浕涂涌涚涛涝涞涟涠涡涢涣涤润涧涨涩淀渊渌渍渎渐渑渔渖渗温游湾湿溁溃溅溆溇滗滚滞滟滠满滢滤滥滦滨滩滪潆潇潋潍潜潴澛澜濑濒灏灭灯灵灶灾灿炀炉炖炜炝点炼炽烁烂烃烛烟烦烧烨烩烫烬热焕焖焘煴熏爱爷牍牦牵牺犊状犷犸犹狈狝狞独狭狮狯狰狱狲猃猎猕猡猪猫猬献獭玑玙玚玛玮环现玱玺珐珑珰珲琎琏琐琼瑶瑷瑸璎瓒瓮瓯电画畅畴疖疗疟疠疡疬疭疮疯疱疴痈痉痒痖痨痪痫痴瘅瘆瘗瘘瘪瘫瘾瘿癞癣癫皂皑皱皲盏盐监盖盗盘眍眦眬睁睐睑瞆瞒瞩矫矶矾矿砀码砖砗砚砜砺砻砾础硁硕硖硗硙硚确硵硷碍碛碜碱礼祃祎祢祯祷祸禀禄禅离秃秆种秘积称秽秾稆税稣稳穑穞穷窃窍窎窑窜窝窥窦窭竖竞笃笋笔笕笺笼笾筑筚筛筜筝筹筼签筿简箓箦箧箨箩箪箫篑篓篮篯篱簖籁籴类籼粜粝粤粪粮粽糁糇糍紧絷緼縆纟纠纡红纣纤纥约级纨纩纪纫纬纭纮纯纰纱纲纳纴纵纶纷纸纹纺纻纼纽纾线绀绁绂练组绅细织终绉绊绋绌绍绎经绐绑绒结绔绕绖绗绘给绚绛络绝绞统绠绡绢绣绤绥绦继绨绩绪绫绬续绮绯绰绱绲绳维绵绶绷绸绹绺绻综绽绾绿缀缁缂缃缄缅缆缇缈缉缊缋缌缍缎缏缐缑缒缓缔缕编缗缘缙缚缛缜缝缞缟缠缡缢缣缤缥缦缧缨缩缪缫缬缭缮缯缰缱缲缳缴缵罂网罗罚罢罴羁羟羡群翘翙翚耢耧耸耻聂聋职聍联聩聪肃肠肤肮肴肾肿胀胁胆胜胧胨胪胫胶脉脍脏脐脑脓脔脚脱脶脸腊腌腘腭腻腼腽腾膑膻臜舆舣舰舱舻艰艳艺节芈芗芜芦苁苇苈苋苌苍苎苏苧苹范茎茏茑茔茕茧荆荐荙荚荛荜荝荞荟荠荡荣荤荥荦荧荨荩荪荫荬荭荮药莅莱莲莳莴莶获莸莹莺莼萚萝萤营萦萧萨葱蒀蒇蒉蒋蒌蒏蓝蓟蓠蓣蓥蓦蔂蔷蔹蔺蔼蕰蕲蕴薮藓藴蘖虏虑虚虫虬虮虱虽虾虿蚀蚁蚂蚃蚕蚝蚬蛊蛎蛏蛮蛰蛱蛲蛳蛴蜕蜗蜡蝇蝈蝉蝎蝼蝾螀螨蟏衅衔补衬衮袄袅袆袜袭袯装裆裈裢裣裤裥褛褴襕见观觃规觅视觇览觉觊觋觌觍觎觏觐觑觞触觯訚詟誉誊讠计订讣认讥讦讧讨让讪讫讬训议讯记讱讲讳讴讵讶讷许讹论讻讼讽设访诀证诂诃评诅识诇诈诉诊诋诌词诎诏诐译诒诓诔试诖诗诘诙诚诛诜话诞诟诠诡询诣诤该详诧诨诩诪诫诬语诮误诰诱诲诳说诵诶请诸诹诺读诼诽课诿谀谁谂调谄谅谆谇谈谉谊谋谌谍谎谏谐谑谒谓谔谕谖谗谘谙谚谛谜谝谞谟谠谡谢谣谤谥谦谧谨谩谪谫谬谭谮谯谰谱谲谳谴谵谶豮贝贞负贠贡财责贤败账货质贩贪贫贬购贮贯贰贱贲贳贴贵贶贷贸费贺贻贼贽贾贿赀赁赂赃资赅赆赇赈赉赊赋赌赍赎赏赐赑赒赓赔赕赖赗赘赙赚赛赜赝赞赟赠赡赢赣赪赵赶趋趱趸跃跄跖跞践跶跷跸跹跻踌踪踬踯蹑蹒蹰蹿躏躜躯輼车轧轨轩轪轫转轭轮软轰轱轲轳轴轵轶轷轸轹轺轻轼载轾轿辀辁辂较辄辅辆辇辈辉辊辋辌辍辎辏辐辑辒输辔辕辖辗辘辙辚辞辟辩辫边辽达迁过迈运还这进远违连迟迩迳迹适选逊递逦逻遗遥邓邝邬邮邹邺邻郁郏郐郑郓郦郧郸酂酝酦酱酽酾酿醖采释里鉴銮錾钅钆钇针钉钊钋钌钍钎钏钐钑钒钓钔钕钖钗钘钙钚钛钜钝钞钟钠钡钢钣钤钥钦钧钨钩钪钫钬钭钮钯钰钱钲钳钴钵钶钷钸钹钺钻钼钽钾钿铀铁铂铃铄铅铆铇铈铉铊铋铌铍铎铏铐铑铒铓铔铕铖铗铘铙铚铛铜铝铞铟铠铡铢铣铤铥铦铧铨铩铪铫铬铭铮铯铰铱铲铳铴铵银铷铸铹铺铻铼铽链铿销锁锂锃锄锅锆锇锈锉锊锋锌锍锎锏锐锑锒锓锔锕锖锗锘错锚锛锜锝锞锟锠锡锢锣锤锥锦锧锨锩锪锫锬锭键锯锰锱锲锳锴锵锶锷锸锹锺锻锼锽锾锿镀镁镂镃镄镅镆镇镈镉镊镋镌镍镎镏镐镑镒镓镔镕镖镗镘镙镚镛镜镝镞镟镠镡镢镣镤镥镦镧镨镩镪镫镬镭镮镯镰镱镲镳镴镵镶长门闩闪闫闬闭问闯闰闱闲闳间闵闶闷闸闹闺闻闼闽闾闿阀阁阂阃阄阅阆阇阈阉阊阋阌阍阎阏阐阑阒阓阔阕阖阗阘阙阚阛队阳阴阵阶际陆陇陈陉陕陦陧陨险随隐隶隽难雇雏雠雳雾霁霉霡霭靓靔静靥鞑鞒鞯鞲韦韧韨韩韪韫韬韵页顶顷顸项顺须顼顽顾顿颀颁颂颃预颅领颇颈颉颊颋颌颍颎颏颐频颒颓颔颕颖颗题颙颚颛颜额颞颟颠颡颢颣颤颥颦颧风飏飐飑飒飓飔飕飖飗飘飙飚飞飨餍饣饤饥饦饧饨饩饪饫饬饭饮饯饰饱饲饳饴饵饶饷饸饹饺饻饼饽饾饿馀馁馂馃馄馅馆馇馈馉馊馋馌馍馎馏馐馑馒馓馔馕马驭驮驯驰驱驲驳驴驵驶驷驸驹驺驻驼驽驾驿骀骁骂骃骄骅骆骇骈骉骊骋验骍骎骏骐骑骒骓骔骕骖骗骘骙骚骛骜骝骞骟骠骡骢骣骤骥骦骧髅髋髌鬓鬶魇魉鱼鱽鱾鱿鲀鲁鲂鲃鲄鲅鲆鲇鲈鲉鲊鲋鲌鲍鲎鲏鲐鲑鲒鲓鲔鲕鲖鲗鲘鲙鲚鲛鲜鲝鲞鲟鲠鲡鲢鲣鲤鲥鲦鲧鲨鲩鲪鲫鲬鲭鲮鲯鲰鲱鲲鲳鲴鲵鲶鲷鲸鲹鲺鲻鲼鲽鲾鲿鳀鳁鳂鳃鳄鳅鳆鳇鳈鳉鳊鳋鳌鳍鳎鳏鳐鳑鳒鳓鳔鳕鳖鳗鳘鳙鳚鳛鳜鳝鳞鳟鳠鳡鳢鳣鳤鸟鸠鸡鸢鸣鸤鸥鸦鸧鸨鸩鸪鸫鸬鸭鸮鸯鸰鸱鸲鸳鸴鸵鸶鸷鸸鸹鸺鸻鸼鸽鸾鸿鹀鹁鹂鹃鹄鹅鹆鹇鹈鹉鹊鹋鹌鹍鹎鹏鹐鹑鹒鹓鹔鹕鹖鹗鹘鹙鹚鹛鹜鹝鹞鹟鹠鹡鹢鹣鹤鹥鹦鹧鹨鹩鹪鹫鹬鹭鹮鹯鹰鹱鹲鹳鹴鹾麦麸麹麺麽黄黉黡黩黪黾鼋鼌鼍鼹齐齑齿龀龁龂龃龄龅龆龇龈龉龊龋龌龙龚龛龟鿎鿏鿒鿔';
const S2T_TO = '萬與醜專業叢東絲丟兩嚴喪個豐臨爲麗舉麼義烏樂喬習鄉書買亂爭於虧雲亙亞產畝親褻嚲億僅僕從侖倉儀們價衆優夥會傴傘偉傳俥俔傷倀倫傖僞佇體餘傭僉俠侶僥偵側僑儈儕儂儘俁儔儼倆儷倈儉債傾傯僂僨償儎儻儐儲儺兒兌兗黨蘭關興茲養獸囅內岡冊寫軍農馮衝決況凍淨悽準涼減湊凜幾鳳鳧憑凱兇擊鑿芻劃劉則剛創刪別剗剄剎劊㓨劌剴劑剮劍剝劇勸辦務勱動勵勁勞勢勳勩勻匭匱區醫華協單賣佔盧滷臥衛卻巹廠廳歷厲壓厭厙龎廁釐廂厴廈廚廄廝縣叄參靉靆雙發變敘疊臺葉號嘆嘰籲喫後嚇呂嗎噸聽啓吳吶嘸囈嘔嚦唄員咼嗆嗚詠嚨嚀噝吒諮鹹響啞噠嘵嗶噦譁噲嚌噥喲脣嘜嗊嘮啢嗩喚嘖嗇囀齧嘓囉嘽嘯噴嘍嚳囁噯噓嚶囑嚕囂團園囪圍圇國圖圓聖壙場壞塊堅壇壢壩塢墳墜壟壠壚壘墾堊墊埡墶壋塏堖塒壎堝塹墮壪牆壯聲殼壺壼處備復夠頭誇夾奪奩奐奮獎奧妝婦媽嫵嫗嬀姍奼婁婭嬈嬌孌娛媧嫺嫿嬰嬋嬸媼嬃嬡嬪嬙嬤孫學孿寧寶實寵審憲宮寬賓寢對尋導壽將爾塵嘗堯尷屍盡層屓屜屆屬屢屨嶼歲豈嶇崗峴嵐島巖嶺嶽崬巋嶨嶧峽嶢嶠崢巒峯嶗崍嶮嶄嶸嶔嶁巔鞏巰幣帥師幃帳簾幟帶幀幫幬幘幗冪幹並廣莊慶牀廬廡庫應廟龐廢廎廩開異棄弒張彌弳彎彈強歸當錄彠彥彲徹徵徑徠憶懺憂愾懷態慫憮慪悵愴憐總懟懌戀恆懇惡慟懨愷惻惱惲悅愨懸慳悞憫驚懼慘懲憊愜慚憚慣慍憤憒願懾憖懣懶懍戇戔戲戧戰戩戱戶撲託執擴捫掃揚擾撫拋摶摳掄搶護報擔擬攏揀擁攔擰撥擇掛摯攣掗撾撻挾撓擋撟掙擠揮撏挩撈損撿換搗據擄摑擲撣摻摜攬搵撳攙擱摟揯攪攜攝攄擺搖擯攤攖撐攆擷擼攛㩵擻攢敵敓斂斆數齋斕鬥斬斷無舊時曠暘曇暱晝曨顯晉曬曉曄暈暉暫曖術樸機殺雜權槓條來楊榪傑極構樅樞棗櫪梘棖槍楓梟櫃檸檉梔柵標棧櫛櫳棟櫨櫟欄樹棲慄樣欒椏橈楨檔榿橋樺檜槳樁樳夢檮棶槤檢梲欞槨槼櫝槧槶欏樿橢槮樓欖榲櫬櫚櫸樧檟檻檳櫧橫檣櫻櫫櫥櫓櫞檁歡歟歐殲歿殤殘殞殮殫殯毆毀轂畢斃氈毿氌氣氫氬氳匯漢湯洶澐溝沒灃漚瀝淪滄渢潙滬濘淚澩瀧瀘濼瀉潑澤涇潔灑窪浹淺漿澆湞溮濁測澮濟瀏滻渾滸濃潯濜塗湧涗濤澇淶漣潿渦溳渙滌潤澗漲澀澱淵淥漬瀆漸澠漁瀋滲溫遊灣溼濚潰濺漵漊潷滾滯灩灄滿瀅濾濫灤濱灘澦瀠瀟瀲濰潛瀦瀂瀾瀨瀕灝滅燈靈竈災燦煬爐燉煒熗點煉熾爍爛烴燭煙煩燒燁燴燙燼熱煥燜燾熅燻愛爺牘犛牽犧犢狀獷獁猶狽獮獰獨狹獅獪猙獄猻獫獵獼玀豬貓蝟獻獺璣璵瑒瑪瑋環現瑲璽琺瓏璫琿璡璉瑣瓊瑤璦璸瓔瓚甕甌電畫暢疇癤療瘧癘瘍癧瘲瘡瘋皰痾癰痙癢瘂癆瘓癇癡癉瘮瘞瘻癟癱癮癭癩癬癲皁皚皺皸盞鹽監蓋盜盤瞘眥矓睜睞瞼瞶瞞矚矯磯礬礦碭碼磚硨硯碸礪礱礫礎硜碩硤磽磑礄確磠礆礙磧磣鹼禮禡禕禰禎禱禍稟祿禪離禿稈種祕積稱穢穠穭稅穌穩穡穭窮竊竅窵窯竄窩窺竇窶豎競篤筍筆筧箋籠籩築篳篩簹箏籌篔籤篠簡籙簀篋籜籮簞簫簣簍籃籛籬籪籟糴類秈糶糲粵糞糧糉糝餱餈緊縶縕緪糹糾紆紅紂纖紇約級紈纊紀紉緯紜紘純紕紗綱納紝縱綸紛紙紋紡紵紖紐紓線紺紲紱練組紳細織終縐絆紼絀紹繹經紿綁絨結絝繞絰絎繪給絢絳絡絕絞統綆綃絹繡綌綏絛繼綈績緒綾緓續綺緋綽鞝緄繩維綿綬繃綢綯綹綣綜綻綰綠綴緇緙緗緘緬纜緹緲緝縕繢緦綞緞緶線緱縋緩締縷編緡緣縉縛縟縝縫縗縞纏縭縊縑繽縹縵縲纓縮繆繅纈繚繕繒繮繾繰繯繳纘罌網羅罰罷羆羈羥羨羣翹翽翬耮耬聳恥聶聾職聹聯聵聰肅腸膚骯餚腎腫脹脅膽勝朧腖臚脛膠脈膾髒臍腦膿臠腳脫腡臉臘醃膕齶膩靦膃騰臏羶臢輿艤艦艙艫艱豔藝節羋薌蕪蘆蓯葦藶莧萇蒼苧蘇薴蘋範莖蘢蔦塋煢繭荊薦薘莢蕘蓽萴蕎薈薺蕩榮葷滎犖熒蕁藎蓀蔭蕒葒葤藥蒞萊蓮蒔萵薟獲蕕瑩鶯蓴蘀蘿螢營縈蕭薩蔥蒕蕆蕢蔣蔞醟藍薊蘺蕷鎣驀虆薔蘞藺藹薀蘄蘊藪蘚蘊櫱虜慮虛蟲虯蟣蝨雖蝦蠆蝕蟻螞蠁蠶蠔蜆蠱蠣蟶蠻蟄蛺蟯螄蠐蛻蝸蠟蠅蟈蟬蠍螻蠑螿蟎蠨釁銜補襯袞襖嫋褘襪襲襏裝襠褌褳襝褲襉褸襤襴見觀覎規覓視覘覽覺覬覡覿覥覦覯覲覷觴觸觶誾讋譽謄訁計訂訃認譏訐訌討讓訕訖託訓議訊記訒講諱謳詎訝訥許訛論訩訟諷設訪訣證詁訶評詛識詗詐訴診詆謅詞詘詔詖譯詒誆誄試詿詩詰詼誠誅詵話誕詬詮詭詢詣諍該詳詫諢詡譸誡誣語誚誤誥誘誨誑說誦誒請諸諏諾讀諑誹課諉諛誰諗調諂諒諄誶談讅誼謀諶諜謊諫諧謔謁謂諤諭諼讒諮諳諺諦謎諞諝謨讜謖謝謠謗諡謙謐謹謾謫譾謬譚譖譙讕譜譎讞譴譫讖豶貝貞負貟貢財責賢敗賬貨質販貪貧貶購貯貫貳賤賁貰貼貴貺貸貿費賀貽賊贄賈賄貲賃賂贓資賅贐賕賑賚賒賦賭齎贖賞賜贔賙賡賠賧賴賵贅賻賺賽賾贗贊贇贈贍贏贛赬趙趕趨趲躉躍蹌蹠躒踐躂蹺蹕躚躋躊蹤躓躑躡蹣躕躥躪躦軀轀車軋軌軒軑軔轉軛輪軟轟軲軻轤軸軹軼軤軫轢軺輕軾載輊轎輈輇輅較輒輔輛輦輩輝輥輞輬輟輜輳輻輯轀輸轡轅轄輾轆轍轔辭闢辯辮邊遼達遷過邁運還這進遠違連遲邇逕跡適選遜遞邐邏遺遙鄧鄺鄔郵鄒鄴鄰鬱郟鄶鄭鄆酈鄖鄲酇醞醱醬釅釃釀醞採釋裏鑑鑾鏨釒釓釔針釘釗釙釕釷釺釧釤鈒釩釣鍆釹鍚釵鈃鈣鈈鈦鉅鈍鈔鍾鈉鋇鋼鈑鈐鑰欽鈞鎢鉤鈧鈁鈥鈄鈕鈀鈺錢鉦鉗鈷鉢鈳鉕鈽鈸鉞鑽鉬鉭鉀鈿鈾鐵鉑鈴鑠鉛鉚鉋鈰鉉鉈鉍鈮鈹鐸鉶銬銠鉺鋩錏銪鋮鋏鋣鐃銍鐺銅鋁銱銦鎧鍘銖銑鋌銩銛鏵銓鎩鉿銚鉻銘錚銫鉸銥鏟銃鐋銨銀銣鑄鐒鋪鋙錸鋱鏈鏗銷鎖鋰鋥鋤鍋鋯鋨鏽銼鋝鋒鋅鋶鐦鐧銳銻鋃鋟鋦錒錆鍺鍩錯錨錛錡鍀錁錕錩錫錮鑼錘錐錦鑕鍁錈鍃錇錟錠鍵鋸錳錙鍥鍈鍇鏘鍶鍔鍤鍬鍾鍛鎪鍠鍰鎄鍍鎂鏤鎡鐨鎇鏌鎮鎛鎘鑷钂鐫鎳鎿鎦鎬鎊鎰鎵鑌鎔鏢鏜鏝鏍鏰鏞鏡鏑鏃鏇鏐鐔钁鐐鏷鑥鐓鑭鐠鑹鏹鐙鑊鐳鐶鐲鐮鐿鑔鑣鑞鑱鑲長門閂閃閆閈閉問闖閏闈閒閎間閔閌悶閘鬧閨聞闥閩閭闓閥閣閡閫鬮閱閬闍閾閹閶鬩閿閽閻閼闡闌闃闠闊闋闔闐闒闕闞闤隊陽陰陣階際陸隴陳陘陝隯隉隕險隨隱隸雋難僱雛讎靂霧霽黴霢靄靚靝靜靨韃鞽韉韝韋韌韍韓韙韞韜韻頁頂頃頇項順須頊頑顧頓頎頒頌頏預顱領頗頸頡頰頲頜潁熲頦頤頻頮頹頷頴穎顆題顒顎顓顏額顳顢顛顙顥纇顫顬顰顴風颺颭颮颯颶颸颼颻飀飄飆飈飛饗饜飠飣飢飥餳飩餼飪飫飭飯飲餞飾飽飼飿飴餌饒餉餄餎餃餏餅餑餖餓餘餒餕餜餛餡館餷饋餶餿饞饁饃餺餾饈饉饅饊饌饢馬馭馱馴馳驅馹駁驢駔駛駟駙駒騶駐駝駑駕驛駘驍罵駰驕驊駱駭駢驫驪騁驗騂駸駿騏騎騍騅騌驌驂騙騭騤騷騖驁騮騫騸驃騾驄驏驟驥驦驤髏髖髕鬢鬹魘魎魚魛魢魷魨魯魴䰾魺鮁鮃鮎鱸鮋鮓鮒鮊鮑鱟鮍鮐鮭鮚鮳鮪鮞鮦鰂鮜鱠鱭鮫鮮鮺鯗鱘鯁鱺鰱鰹鯉鰣鰷鯀鯊鯇鮶鯽鯒鯖鯪鯕鯫鯡鯤鯧鯝鯢鯰鯛鯨鰺鯴鯔鱝鰈鰏鱨鯷鰮鰃鰓鱷鰍鰒鰉鰁鱂鯿鰠鰲鰭鰨鰥鰩鰟鰜鰳鰾鱈鱉鰻鰵鱅䲁鰼鱖鱔鱗鱒鱯鱤鱧鱣䲘鳥鳩雞鳶鳴鳲鷗鴉鶬鴇鴆鴣鶇鸕鴨鴞鴦鴒鴟鴝鴛鷽鴕鷥鷙鴯鴰鵂鴴鵃鴿鸞鴻鵐鵓鸝鵑鵠鵝鵒鷳鵜鵡鵲鶓鵪鵾鵯鵬鵮鶉鶊鵷鷫鶘鶡鶚鶻鶖鷀鶥鶩鷊鷂鶲鶹鶺鷁鶼鶴鷖鸚鷓鷚鷯鷦鷲鷸鷺䴉鸇鷹鸌鸏鸛鸘鹺麥麩麴麪麼黃黌黶黷黲黽黿鼂鼉鼴齊齏齒齔齕齗齟齡齙齠齜齦齬齪齲齷龍龔龕龜䃮䥑鿓鎶';
const S2T_MAP = (() => { const m = {}; for (let i = 0; i < S2T_FROM.length; i++) m[S2T_FROM[i]] = S2T_TO[i]; return m; })();

function toTraditional(str) {
    if (!str) return str;
    let out = '';
    for (const ch of str) out += S2T_MAP[ch] || ch;
    return out;
}

// 把解析結果中的字串值全部轉繁體（非字串原樣保留）
function jsonValuesToTraditional(obj) {
    if (!obj || typeof obj !== 'object') return obj;
    const out = {};
    for (const k in obj) out[k] = (typeof obj[k] === 'string') ? toTraditional(obj[k]) : obj[k];
    return out;
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

    // 預熱:趁使用者調整裁切邊角的幾秒,先把模型載入記憶體,
    // 讓稍後的 OCR 不必付冷啟動載入時間。fire-and-forget,不擋 UI。
    warmUpModel((ollamaModelSelect && ollamaModelSelect.value) || '');

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

    // 預熱(同 openCropSection):趁裁切期間先載入模型。
    warmUpModel((ollamaModelSelect && ollamaModelSelect.value) || '');

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
        // 同 performOCR:送模型前縮到 1024px,避免大圖拖慢 vision 編碼。
        const resizedUrl = await resizeCardImage(imageDataUrl, 1024, 0.85);
        const b64 = imageDataUrlToBase64(resizedUrl);
        const visionModel = (ollamaModelSelect && ollamaModelSelect.value) || 'llama3.2-vision';
        
        const backStartTime = Date.now();
        const backTimer = setInterval(() => {
            const elapsed = ((Date.now() - backStartTime) / 1000).toFixed(0);
            backProgressText.textContent = `使用 ${visionModel} 讀取背面資料中... (${elapsed}s)`;
        }, 1000);

        backProgressFill.style.width = '50%';

        const systemPrompt = `You are a strict data extraction assistant. Extract all contact information from the provided business card image.

CRITICAL RULES:
1. NEVER translate between languages. English text must stay in English exactly as printed on the card; do NOT translate English into Chinese, and do NOT translate Chinese into English.
2. Chinese text must be written in Traditional Chinese (繁體中文): if the card contains Simplified Chinese characters, convert them to Traditional (character conversion only, not translation).
   Examples: 财务部→財務部、经理→經理、100号5楼→100號5樓。
3. Put English values in the *En fields and Chinese values in the *Cn fields. If the card has no content in that language, leave the field "".
Keep digits, punctuation, emails and URLs unchanged.

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
            jsonResult = jsonValuesToTraditional(parseLLMJson(llmResp));
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
        // 送模型前先縮到最長邊 1024px:名片文字在 1024px 已清晰,但原始
        // 相機/上傳影像常達 3000~4000px,整張送會讓 vision 編碼暴慢
        // (實測 3000px≈9s vs 1024px≈2s)。resizeCardImage 只縮不放大。
        const resizedUrl = await resizeCardImage(imageDataUrl, 1024, 0.85);
        const b64 = imageDataUrlToBase64(resizedUrl);
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

CRITICAL RULES:
1. NEVER translate between languages. English text must stay in English exactly as printed on the card; do NOT translate English into Chinese, and do NOT translate Chinese into English.
2. Chinese text must be written in Traditional Chinese (繁體中文): if the card contains Simplified Chinese characters, convert them to Traditional (character conversion only, not translation).
   Examples: 财务部→財務部、经理→經理、100号5楼→100號5樓。
3. Put English values in the *En fields and Chinese values in the *Cn fields. If the card has no content in that language, leave the field "".
Keep digits, punctuation, emails and URLs unchanged.

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
        if (liveOCRText) liveOCRText.value = '';
        // Step 1: call the model (network / model-load / OOM errors surface here)
        try {
            resp = await dmrChat(visionModel, userPrompt, b64, {
                systemPrompt: systemPrompt,
                temperature: 0.2,
                format: 'json',
                // 串流:邊生成邊寫進「Live OCR Output」視窗
                onToken: (txt) => { if (liveOCRText) liveOCRText.value = txt; }
            });
        } catch (apiErr) {
            console.error('Vision API call failed:', apiErr);
            throw new Error(`呼叫模型 ${visionModel} 失敗:${apiErr.message}`);
        }

        // Show raw LLM output in live text area
        if (liveOCRText) liveOCRText.value = resp;

        // Step 2: parse the JSON (formatting issues surface here, not "not a vision model")
        try {
            visionJson = jsonValuesToTraditional(parseLLMJson(resp));
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

// ─── vCard 3.0（RFC 2426）文字值跳脫：\ ; , 換行 ───
//     Google 聯絡人、Apple iCloud 聯絡人用 3.0；整檔直接 UTF-8，不用 QP
function escape30(value) {
    if (!value) return '';
    return String(value)
        .replace(/\\/g, '\\\\')
        .replace(/;/g, '\\;')
        .replace(/,/g, '\\,')
        .replace(/\r?\n/g, '\\n');
}

// ─── 產生 3.0 屬性行（值須已跳脫），超過 75 位元組依 RFC 2426 折行 ───
function vProp30(nameAndParams, value) {
    return foldLine(`${nameAndParams}:${value}`);
}

// ─── 照片：vCard 3.0 用 ENCODING=b，整行交給 foldLine 折行 ───
function photoLine30(b64) {
    return foldLine(`PHOTO;ENCODING=b;TYPE=JPEG:${b64}`);
}


// ─── vCard 2.1（傳統版 Outlook／Windows 聯絡人）───
//     主要地址 TYPE=WORK,PREF → Outlook「商務地址」並設為郵寄地址；
//     輔助地址 TYPE=POSTAL → Outlook「其他地址」（MS-OXVCARD ADR 對應表）
function buildVCard21(c) {
    const L = ['BEGIN:VCARD', 'VERSION:2.1'];
    L.push(vProp21('N', `${escape21(c.familyName)};${escape21(c.givenName)};;;`));
    L.push(vProp21('FN', c.displayName));
    if (c.orgParts.length) L.push(vProp21('ORG', c.orgParts.map(escape21).join(';')));
    if (c.title) L.push(vProp21('TITLE', c.title));
    if (c.mobile) L.push(vProp21('TEL;TYPE=CELL,VOICE', c.mobile));
    if (c.workPhone) L.push(vProp21('TEL;TYPE=WORK,VOICE', c.workPhone));
    if (c.email) L.push(vProp21('EMAIL;TYPE=INTERNET,WORK', c.email));
    if (c.website) L.push(vProp21('URL;TYPE=WORK', c.website));
    if (c.primaryAddr) L.push(vProp21('ADR;TYPE=WORK,PREF', `;;${escape21(c.primaryAddr)};;;${escape21(c.zip)};`));
    if (c.secondaryAddr) L.push(vProp21('ADR;TYPE=POSTAL', `;;${escape21(c.secondaryAddr)};;;${escape21(c.zip)};`));
    if (c.note) L.push(vProp21('NOTE', c.note));
    if (c.categories) L.push(vProp21('CATEGORIES', c.categories));
    if (c.photoB64) L.push(...photoBlock21('PHOTO;ENCODING=BASE64;TYPE=JPEG', c.photoB64));
    // 背面照：第二個 PHOTO 加 X-ABLABEL 參數（vcf_to_win.py 同款寫法）
    if (c.backB64) L.push(...photoBlock21('PHOTO;ENCODING=BASE64;TYPE=JPEG;X-ABLABEL=Back', c.backB64));
    L.push('END:VCARD');
    return L;
}

// ─── vCard 3.0（Google 聯絡人／Apple iCloud 聯絡人）───
//     輔助地址用 item1. 群組 + X-ABLabel 自訂標籤（Google/Apple 匯出同款寫法）
//     3.0 只放一張照片（正面），避免匯入端誤判
function buildVCard30(c) {
    const L = ['BEGIN:VCARD', 'VERSION:3.0'];
    L.push(vProp30('N', `${escape30(c.familyName)};${escape30(c.givenName)};;;`));
    L.push(vProp30('FN', escape30(c.displayName)));
    if (c.orgParts.length) L.push(vProp30('ORG', c.orgParts.map(escape30).join(';')));
    if (c.title) L.push(vProp30('TITLE', escape30(c.title)));
    if (c.mobile) L.push(vProp30('TEL;TYPE=CELL,VOICE', escape30(c.mobile)));
    if (c.workPhone) L.push(vProp30('TEL;TYPE=WORK,VOICE', escape30(c.workPhone)));
    if (c.email) L.push(vProp30('EMAIL;TYPE=INTERNET,WORK', escape30(c.email)));
    if (c.website) L.push(vProp30('URL;TYPE=WORK', c.website));
    if (c.primaryAddr) L.push(vProp30('ADR;TYPE=WORK,PREF', `;;${escape30(c.primaryAddr)};;;${escape30(c.zip)};`));
    if (c.secondaryAddr) {
        L.push(vProp30('item1.ADR', `;;${escape30(c.secondaryAddr)};;;${escape30(c.zip)};`));
        L.push(vProp30('item1.X-ABLabel', escape30(c.secondaryLabel)));
    }
    if (c.note) L.push(vProp30('NOTE', escape30(c.note)));
    if (c.categories) {
        L.push(vProp30('CATEGORIES', c.categories.split(/[,，]/).map(s => escape30(s.trim())).filter(Boolean).join(',')));
    }
    if (c.photoB64) L.push(photoLine30(c.photoB64));
    L.push('END:VCARD');
    return L;
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

    // 姓名：英文為主要資料，沒有英文才用中文；兩者都有時中文列為輔助資料
    const fullNameCn = familyNameCn + givenNameCn;
    const fullNameEn = [givenNameEn, familyNameEn].filter(Boolean).join(' ');
    const hasEnName = !!(familyNameEn || givenNameEn);
    const familyName = hasEnName ? familyNameEn : familyNameCn;
    const givenName = hasEnName ? givenNameEn : givenNameCn;
    const fullName = fullNameEn || fullNameCn;

    if (!fullName && !mobile && !workPhone && !email) {
        alert('請至少填寫姓名、電話或 Email 其中一項再匯出。');
        return;
    }

    // 顯示名稱保留中文於括號內，方便用中文搜尋
    const displayName = (fullNameEn && fullNameCn) ? `${fullNameEn} (${fullNameCn})` : fullName;

    // 地址：英文為主要地址；兩者都有時中文為輔助地址
    const primaryAddr = addrEn || addrCn;
    const secondaryAddr = (addrEn && addrCn) ? addrCn : '';

    // NOTE：輔助資料（中文姓名、中文地址）+ 統編 + 備註
    const noteLines = [];
    if (hasEnName && fullNameCn) noteLines.push(`中文姓名: ${fullNameCn}`);
    if (secondaryAddr) noteLines.push(`中文地址: ${secondaryAddr}`);
    if (taxIdVal) noteLines.push(`統編: ${taxIdVal}`);
    if (note) noteLines.push(note);

    // 匯出格式：2.1（Outlook）或 3.0（Google／Apple）
    const versionSelect = document.getElementById('vcardVersionSelect');
    const version = (versionSelect && versionSelect.value === '3.0') ? '3.0' : '2.1';

    const contact = {
        familyName, givenName, displayName,
        orgParts: [company, department].filter(Boolean),
        title: titleVal,
        mobile,
        // Outlook 可辨識「號碼 x分機」的寫法
        workPhone: workPhone ? (ext ? `${workPhone} x${ext}` : workPhone) : '',
        email, website,
        primaryAddr, secondaryAddr, secondaryLabel: '中文地址',
        zip: zipShared,
        note: noteLines.join('\n'),
        categories,
        photoB64: null, backB64: null,
    };

    // ─── 影像：主相片優先用透視校正後的名片圖 ───
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
    contact.photoB64 = photoB64;

    // 背面照只放進 2.1（3.0 只保留一張照片）
    if (version === '2.1' && backPhotoDataUrl) {
        try {
            const resizedBack = await resizeCardImage(backPhotoDataUrl, 640, 0.8);
            contact.backB64 = resizedBack.split(',')[1];
        } catch (e) {
            console.warn('無法縮放背面影像:', e);
        }
    }

    const lines = version === '3.0' ? buildVCard30(contact) : buildVCard21(contact);

    // 2.1：整檔純 ASCII（中文在 QP 的 =XX 裡）；3.0：直接 UTF-8。
    // 兩者都不加 BOM，結尾補一組 CRLF。
    const vcfString = lines.join('\r\n') + '\r\n';

    const blob = new Blob([vcfString], { type: 'text/vcard;charset=utf-8' });
    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');
    link.href = url;
    const suffix = version === '3.0' ? 'v30' : 'v21';
    const filename = fullName ? `${fullName.replace(/\s+/g, '_')}_contact_${suffix}.vcf` : `contact_${suffix}.vcf`;
    link.setAttribute('download', filename);

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    showSection(successSection);
}
