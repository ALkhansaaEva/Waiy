// static/app.js

// ======= DOM refs =======
const fileInput   = document.getElementById('fileInput');
const dropZone    = document.getElementById('dropZone');

const frameEl     = document.getElementById('frame');
const placeholder = document.getElementById('placeholder');
const canvas      = document.getElementById('previewCanvas');
const ctx         = canvas.getContext('2d');

const changeBar   = document.getElementById('changeBar');
const changeBtn   = document.getElementById('changeBtn');
const resetBtn    = document.getElementById('resetBtn');

const analyzeBtn  = document.getElementById('analyzeBtn');
const loadingEl   = document.getElementById('loading');
const resultEl    = document.getElementById('result');
const statusEl    = document.getElementById('status');

// ======= Config =======
const REQ_TIMEOUT_MS = 15000;

// ======= State =======
let isBusy = false;
let abortCtrl = null;
let currentDataUrl = null;

// ======= Emotion meta (emoji, color) =======
const EMOJI = {
  "Anger":"😠", "Disgust":"🤢", "Fear":"😨",
  "Happy":"😊", "Neutral":"😐", "Sad":"☹️", "Surprise":"😲"
};
const COLORS = {
  "Anger":"#ff6b6b",
  "Disgust":"#50c878",
  "Fear":"#8a6cff",
  "Happy":"#ffb020",
  "Neutral":"#7b8da5",
  "Sad":"#5da8ff",
  "Surprise":"#eb6fff"
};
const ORDER = ["Anger","Disgust","Fear","Happy","Neutral","Sad","Surprise"];

// ======= UI helpers =======
const show = el => el.classList.remove('hidden');
const hide = el => el.classList.add('hidden');
const setStatus = msg => statusEl.textContent = msg;

function setBusy(busy){
  isBusy = busy;
  [analyzeBtn, resetBtn, changeBtn, fileInput].forEach(el => { if (el) el.disabled = busy; });
}

function parseJsonSafe(text){
  try { return JSON.parse(text); } catch { return null; }
}

function bump(el){
  el.classList.add('bump'); setTimeout(() => el.classList.remove('bump'), 220);
}

// ======= Canvas =======
function resizeCanvasForDPR(){
  const frame = canvas.parentElement;
  const cssWidth = Math.round(frame.clientWidth || 320);
  const size = Math.max(260, Math.min(cssWidth, 520));
  const dpr = window.devicePixelRatio || 1;

  canvas.style.width = size + 'px';
  canvas.style.height = size + 'px';
  canvas.width = Math.round(size * dpr);
  canvas.height = Math.round(size * dpr);

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#0e1420';
  ctx.fillRect(0, 0, size, size);
  return size;
}

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
      ctx.clearRect(0, 0, size, size);
      ctx.drawImage(img, x, y, w, h);
      resolve();
    };
    img.onerror = () => reject(new Error('فشل في تحميل الصورة'));
    img.src = dataUrl;
  });
}

function clearCanvas(){ resizeCanvasForDPR(); }

// ======= Uploader =======
function showUploader(showIt){
  if(showIt){ dropZone.classList.remove('hidden'); dropZone.style.display = ''; }
  else { dropZone.classList.add('hidden'); dropZone.style.display = 'none'; }
}

function setPickedData(dataUrl){
  currentDataUrl = dataUrl;
  showUploader(false);
  hide(placeholder);
  show(changeBar);
  hide(resultEl);
  clearCanvas();
  drawImageToCanvas(dataUrl)
    .then(() => { show(analyzeBtn); bump(frameEl); bump(analyzeBtn); setStatus('الصورة جاهزة — اضغط "تحليل".'); })
    .catch(() => { setStatus('فشل في عرض الصورة.'); });
}

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

// ======= Rendering results (modern) =======
function fmtPct(x){ return (x*100).toFixed(1) + '%'; }

function normalizeProbs(probsDict){
  // Ensure all 7 emotions exist; fill missing with 0.
  const probs = {};
  let s = 0;
  ORDER.forEach(k => { const v = Number(probsDict?.[k] ?? 0); probs[k] = v; s += v; });
  // If not summing ~1, renormalize for display only
  if (s > 0 && Math.abs(1 - s) > 1e-3){
    ORDER.forEach(k => probs[k] = probs[k] / s);
  }
  return probs;
}

function renderBars(probs){
  const wrap = document.createElement('div');
  wrap.className = 'bars';
  ORDER.forEach(k => {
    const row = document.createElement('div');
    row.className = 'bar-row';
    const cap = document.createElement('div');
    cap.className = 'bar-cap';
    cap.innerHTML = `<span class="emo">${EMOJI[k]||''}</span> <span>${k}</span>`;
    const track = document.createElement('div');
    track.className = 'bar-track';
    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    fill.style.background = COLORS[k] || '#4da3ff';
    fill.style.width = Math.round((probs[k] || 0) * 100) + '%';
    const pct = document.createElement('div');
    pct.className = 'bar-pct';
    pct.textContent = fmtPct(probs[k] || 0);
    track.appendChild(fill);
    row.appendChild(cap);
    row.appendChild(track);
    row.appendChild(pct);
    wrap.appendChild(row);
  });
  return wrap;
}

function renderTop3(probs){
  // Build array and sort desc
  const arr = ORDER.map(k => ({k, v: probs[k]||0})).sort((a,b)=>b.v-a.v).slice(0,3);
  const wrap = document.createElement('div');
  wrap.className = 'top3';
  arr.forEach(({k, v}) => {
    const chip = document.createElement('span');
    chip.className = 'top-chip';
    chip.style.borderColor = COLORS[k] || '#4da3ff';
    chip.style.background = 'linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,0))';
    chip.innerHTML = `<span class="emo">${EMOJI[k]||''}</span><strong>${k}</strong><em>${fmtPct(v)}</em>`;
    wrap.appendChild(chip);
  });
  return wrap;
}

function renderResultCard(j){
  const probs = normalizeProbs(j.probs || {});
  const label = j.label || '—';
  const conf  = Number(j.confidence || 0);

  const card = document.createElement('div');
  card.className = 'result-card';

  const header = document.createElement('div');
  header.className = 'result-head';
  const lh = document.createElement('div');
  lh.className = 'result-title';
  lh.innerHTML = `<span class="emo big">${EMOJI[label]||'🧠'}</span>
                  <span class="lbl" style="color:${COLORS[label]||'var(--accent)'}">${label}</span>
                  <span class="conf">${fmtPct(conf)}</span>`;
  const rt = document.createElement('div');
  rt.className = 'result-meta';
  rt.innerHTML = `<span>زمن الاستدلال: ${j.inference_ms} ms</span>`;
  header.appendChild(lh); header.appendChild(rt);

  const sep = document.createElement('div'); sep.className = 'sep';

  const top = renderTop3(probs);
  const bars = renderBars(probs);

  card.appendChild(header);
  card.appendChild(sep);
  card.appendChild(top);
  card.appendChild(bars);
  return card;
}

// ======= Networking =======
async function predictBase64(dataUrl){
  if(!dataUrl){ setStatus('لا توجد صورة لتحليلها.'); return; }
  hide(resultEl); show(loadingEl); setBusy(true);

  if(abortCtrl) abortCtrl.abort();
  abortCtrl = new AbortController();
  const timeoutId = setTimeout(() => { try{ abortCtrl.abort('timeout'); }catch{} }, REQ_TIMEOUT_MS);

  try{
    const res = await fetch('/predict_base64', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ image_base64: dataUrl }),
      signal: abortCtrl.signal
    });

    const text = await res.text();
    const j = parseJsonSafe(text) || {};
    if(!res.ok) throw new Error(j.detail || j.message || `HTTP ${res.status}`);

    resultEl.innerHTML = '';
    resultEl.appendChild(renderResultCard(j));
    show(resultEl);
    setStatus('تم ✓');
  }catch(err){
    const msg = (err && err.name === 'AbortError')
      ? 'تم إلغاء الطلب (انتهى الوقت).'
      : (err?.message || 'خطأ غير معروف');
    resultEl.innerHTML = `<div class="result-card"><span class="danger">خطأ: ${msg}</span></div>`;
    show(resultEl);
    setStatus('خطأ — تحقّق من الخادم/API');
  }finally{
    hide(loadingEl); setBusy(false); clearTimeout(timeoutId); abortCtrl = null;
  }
}

// ======= File input / drag&drop =======
function handleFile(file){
  if (!(file?.type?.startsWith('image/'))){ setStatus('اختر ملف صورة صالح.'); return; }
  const MAX_MB = 8;
  if(file.size > MAX_MB * 1024 * 1024){ setStatus(`الصورة كبيرة (> ${MAX_MB} MB).`); return; }
  const reader = new FileReader();
  reader.onload = e => setPickedData(e.target.result);
  reader.onerror = () => setStatus('تعذّر قراءة الملف.');
  reader.readAsDataURL(file);
}

// Events
changeBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => { const f = e.target.files?.[0]; if(f) handleFile(f); else setStatus('جاهز'); });

['dragenter','dragover'].forEach(evt => dropZone.addEventListener(evt, e => {
  e.preventDefault(); e.stopPropagation(); dropZone.classList.add('is-dragover');
}));
['dragleave','drop'].forEach(evt => dropZone.addEventListener(evt, e => {
  e.preventDefault(); e.stopPropagation(); dropZone.classList.remove('is-dragover');
}));
dropZone.addEventListener('drop', e => { const f = e.dataTransfer.files?.[0]; if(f) handleFile(f); });
dropZone.addEventListener('click', () => fileInput.click());

resetBtn.addEventListener('click', clearUI);
analyzeBtn.addEventListener('click', () => { if(!isBusy) predictBase64(currentDataUrl); });

window.addEventListener('resize', () => {
  if(currentDataUrl){ drawImageToCanvas(currentDataUrl).catch(() => clearCanvas()); }
  else { clearCanvas(); }
});

// Init
(function init(){ clearUI(); })();
