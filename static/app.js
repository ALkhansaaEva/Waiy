// Get DOM elements references
const fileInput   = document.getElementById('fileInput');
const dropZone    = document.getElementById('dropZone');

const frameEl     = document.getElementById('frame');
const placeholder = document.getElementById('placeholder');
const canvas      = document.getElementById('previewCanvas');
const ctx         = canvas.getContext('2d');

const changeBar   = document.getElementById('changeBar');
const changeBtn   = document.getElementById('changeBtn');
const resetBtn    = document.getElementById('resetBtn');

const loadHappy   = document.getElementById('loadHappy');
const loadSad     = document.getElementById('loadSad');

const analyzeBtn  = document.getElementById('analyzeBtn');
const loadingEl   = document.getElementById('loading');
const resultEl    = document.getElementById('result');
const statusEl    = document.getElementById('status');

// Configuration constants
const REQ_TIMEOUT_MS = 15000; // Timeout for API requests in milliseconds

// State variables
let isBusy = false;
let abortCtrl = null;
let currentDataUrl = null;

// Utility functions for UI control
const show = el => el.classList.remove('hidden');
const hide = el => el.classList.add('hidden');
const setStatus = msg => statusEl.textContent = msg;

// Enable or disable buttons and inputs during busy state
function setBusy(busy){
  isBusy = busy;
  [analyzeBtn, resetBtn, changeBtn, loadHappy, loadSad, fileInput].forEach(el => {
    if(el) el.disabled = busy;
  });
}

// Safely parse JSON, return null on failure
function parseJsonSafe(text){
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

// Add and remove a temporary "bump" animation class
function bump(el){
  el.classList.add('bump');
  setTimeout(() => el.classList.remove('bump'), 220);
}

// Resize canvas based on container size and device pixel ratio
function resizeCanvasForDPR(){
  const frame = canvas.parentElement;
  const cssWidth = Math.round(frame.clientWidth || 320);
  const size = Math.max(220, Math.min(cssWidth, 420));
  const dpr = window.devicePixelRatio || 1;

  canvas.style.width = size + 'px';
  canvas.style.height = size + 'px';
  canvas.width = Math.round(size * dpr);
  canvas.height = Math.round(size * dpr);

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0e1420'; // Background color for canvas
  ctx.fillRect(0, 0, size, size);

  return size;
}

// Draw image on canvas centered and scaled to fit
function drawImageToCanvas(dataUrl){
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const size = resizeCanvasForDPR();
      const iw = img.naturalWidth, ih = img.naturalHeight;
      const scale = Math.min(size / iw, size / ih);
      const w = Math.round(iw * scale);
      const h = Math.round(ih * scale);
      const x = Math.round((size - w) / 2);
      const y = Math.round((size - h) / 2);

      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, x, y, w, h);
      resolve();
    };
    img.onerror = () => reject(new Error('فشل في تحميل الصورة'));
    img.src = dataUrl;
  });
}

// Clear canvas content and repaint background
function clearCanvas(){
  resizeCanvasForDPR();
}

// Generate sample face images (happy or sad) using canvas drawing
function makeSample(kind = 'happy'){
  const c = document.createElement('canvas');
  const s = 256;
  c.width = s;
  c.height = s;
  const g = c.getContext('2d');

  // Face circle
  g.fillStyle = '#FFD54F';
  g.beginPath();
  g.arc(s / 2, s / 2, s * 0.42, 0, Math.PI * 2);
  g.fill();

  // Eyes
  g.fillStyle = '#1e1e1e';
  g.beginPath();
  g.arc(s * 0.38, s * 0.42, s * 0.045, 0, Math.PI * 2);
  g.fill();
  g.beginPath();
  g.arc(s * 0.62, s * 0.42, s * 0.045, 0, Math.PI * 2);
  g.fill();

  // Mouth with cubic Bézier curve
  const lineWidth = Math.max(8, Math.round(s * 0.08));
  g.lineWidth = lineWidth;
  g.lineCap = 'round';
  g.strokeStyle = '#1e1e1e';
  g.beginPath();

  const x0 = s * 0.30;
  const x1 = s * 0.70;
  const y = kind === 'happy' ? s * 0.64 : s * 0.70;
  const dx = s * 0.18;
  const dY = s * 0.18;

  g.moveTo(x0, y);
  if(kind === 'happy'){
    // Smile curve (control points below the line)
    g.bezierCurveTo(x0 + dx, y + dY, x1 - dx, y + dY, x1, y);
  } else {
    // Frown curve (control points above the line)
    g.bezierCurveTo(x0 + dx, y - dY, x1 - dx, y - dY, x1, y);
  }
  g.stroke();

  return c.toDataURL('image/png');
}

// Show or hide uploader area (drop zone)
function showUploader(showIt){
  if(showIt){
    dropZone.classList.remove('hidden');
    dropZone.style.display = '';
  } else {
    dropZone.classList.add('hidden');
    dropZone.style.display = 'none';
  }
}

// Set selected image data and update UI accordingly
function setPickedData(dataUrl){
  currentDataUrl = dataUrl;
  showUploader(false);
  hide(placeholder);
  show(changeBar);
  clearCanvas();
  drawImageToCanvas(dataUrl)
    .then(() => {
      show(analyzeBtn);
      bump(frameEl);
      bump(analyzeBtn);
      setStatus('الصورة جاهزة — اضغط "تحليل".');
    })
    .catch(() => {
      setStatus('فشل في عرض الصورة.');
    });
}

// Reset UI to initial state
function clearUI(){
  currentDataUrl = null;
  showUploader(true);
  hide(analyzeBtn);
  hide(changeBar);
  show(placeholder);
  clearCanvas();
  resultEl.innerHTML = '';
  hide(resultEl);
  hide(loadingEl);
  setStatus('جاهز');
}

// Send image to backend for prediction with hard timeout
async function predictBase64(dataUrl){
  if(!dataUrl){
    setStatus('لا توجد صورة لتحليلها.');
    return;
  }
  hide(resultEl);
  show(loadingEl);
  setBusy(true);

  if(abortCtrl) abortCtrl.abort();
  abortCtrl = new AbortController();
  const timeoutId = setTimeout(() => {
    try { abortCtrl.abort('timeout'); } catch {}
  }, REQ_TIMEOUT_MS);

  try {
    const res = await fetch('/predict_base64', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ image_base64: dataUrl }),
      signal: abortCtrl.signal
    });

    const text = await res.text();
    const j = parseJsonSafe(text) || {};

    if(!res.ok) throw new Error(j.detail || j.message || `HTTP ${res.status}`);

    const sad   = j.probs?.Sad   ?? 0;
    const happy = j.probs?.Happy ?? 0;

    resultEl.innerHTML = `
      <div><strong>الحالة:</strong> ${j.label}</div>
      <div><strong>النسبة:</strong> ${(j.confidence*100).toFixed(1)}%</div>
      <div><strong>سعيد:</strong> ${(happy*100).toFixed(1)}% · <strong>حزين:</strong> ${(sad*100).toFixed(1)}%</div>
      <div><small>زمن الاستدلال:</small> ${j.inference_ms} ms</div>
    `;
    show(resultEl);
    setStatus('تم ✓');

  } catch(err) {
    const msg = (err && err.name === 'AbortError')
      ? 'تم إلغاء الطلب (انتهى الوقت). تحقق من الخادم أو الشبكة.'
      : (err?.message || 'خطأ غير معروف');
    resultEl.innerHTML = `<span class="danger">خطأ: ${msg}</span>`;
    show(resultEl);
    setStatus('خطأ — تحقّق من الخادم/API');
  } finally {
    hide(loadingEl);
    setBusy(false);
    clearTimeout(timeoutId);
    abortCtrl = null;
  }
}

// Handle file input or dropped file
function handleFile(file){
  if (!(file?.type?.startsWith('image/'))) {
    setStatus('اختر ملف صورة صالح.');
    return;
  }

  const MAX_MB = 8;
  if(file.size > MAX_MB * 1024 * 1024){
    setStatus(`الصورة كبيرة (> ${MAX_MB} MB).`);
    return;
  }

  const reader = new FileReader();
  reader.onload = e => setPickedData(e.target.result);
  reader.onerror = () => setStatus('تعذّر قراءة الملف.');
  reader.readAsDataURL(file);
}

// Event Listeners

// Clicking change button triggers file picker
changeBtn.addEventListener('click', () => fileInput.click());

// File input change
fileInput.addEventListener('change', e => {
  const file = e.target.files?.[0];
  if(file) handleFile(file);
  else setStatus('جاهز');
});

// Drag and drop events
['dragenter', 'dragover'].forEach(evt =>
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('is-dragover');
  })
);

['dragleave', 'drop'].forEach(evt =>
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('is-dragover');
  })
);

dropZone.addEventListener('drop', e => {
  const file = e.dataTransfer.files?.[0];
  if(file) handleFile(file);
});

dropZone.addEventListener('click', () => fileInput.click());

// Buttons actions
resetBtn.addEventListener('click', clearUI);
analyzeBtn.addEventListener('click', () => {
  if(!isBusy) predictBase64(currentDataUrl);
});

// Load sample images buttons
loadHappy.addEventListener('click', () => {
  if(!isBusy) setPickedData(makeSample('happy'));
});
loadSad.addEventListener('click', () => {
  if(!isBusy) setPickedData(makeSample('sad'));
});

// Repaint canvas on window resize
window.addEventListener('resize', () => {
  if(currentDataUrl) {
    drawImageToCanvas(currentDataUrl).catch(() => clearCanvas());
  } else {
    clearCanvas();
  }
});

// Initialization
(function init(){
  clearUI();
})();
