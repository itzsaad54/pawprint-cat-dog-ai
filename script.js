/* PAWPRINT — Cat vs Dog classifier (ONNX Runtime Web)
 * Put your model file at ./model/model.onnx
 */
const CONFIG = {
  MODEL_URL: 'model/model.onnx',
  SIZE: 224,
  // If predictions look wrong, try 'mobilenet' (x/127.5-1) or 'unit' (x/255).
  PREPROCESS: 'raw',      // 'raw' = 0..255 (notebook default, preprocessing is inside the model)
  DOG_INDEX: 1,           // classes = ['cat','dog'] -> dog is 1
};

const $ = (id) => document.getElementById(id);
const el = {
  drop: $('drop'), file: $('file'), preview: $('preview'), cam: $('cam'),
  pick: $('pickBtn'), camBtn: $('camBtn'), snap: $('snapBtn'), reset: $('resetBtn'),
  verdict: $('verdict'), confBar: $('confBar'), confVal: $('confVal'), confLbl: $('confLbl'),
  pCat: $('pCat'), pDog: $('pDog'), hist: $('hist'), clear: $('clearHist'),
  status: $('status'), statusTxt: $('statusTxt'), toast: $('toast'),
};

let session = null, layout = 'NHWC', mode = 'none', stream = null;

/* ---------- helpers ---------- */
function toast(msg) {
  el.toast.textContent = msg; el.toast.classList.add('show');
  clearTimeout(toast.t); toast.t = setTimeout(() => el.toast.classList.remove('show'), 3200);
}
function setStatus(kind, text) {
  el.status.className = 'status mono ' + kind; el.statusTxt.textContent = text;
}
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

/* ---------- theme ---------- */
(function theme() {
  const t = store.get('pp-theme', 'light');
  document.documentElement.dataset.theme = t;
  $('themeBtn').onclick = () => {
    const n = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = n; store.set('pp-theme', n);
  };
})();

/* ---------- marquee + reveal ---------- */
$('marq').innerHTML = Array(16).fill('<span style="margin:0 28px">CAT</span><span>/</span><span style="margin:0 28px">DOG</span><span>/</span>').join('');
const io = new IntersectionObserver((es) => es.forEach(e => e.isIntersecting && e.target.classList.add('in')), { threshold: .12 });
document.querySelectorAll('.reveal').forEach(n => io.observe(n));

/* ---------- model loading ---------- */
async function loadModel() {
  if (typeof ort === 'undefined') {
    setStatus('err', 'ONNX Runtime failed to load');
    toast('onnxruntime-web CDN did not load — check your internet connection');
    return;
  }
  try {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
    session = await ort.InferenceSession.create(CONFIG.MODEL_URL, { executionProviders: ['wasm'] });
    console.log('ONNX inputs:', session.inputNames, 'outputs:', session.outputNames);
    // find which layout the model wants (NHWC for Keras exports, NCHW for PyTorch)
    try { await run(new Float32Array(CONFIG.SIZE * CONFIG.SIZE * 3)); }
    catch (e) {
      console.warn('NHWC failed, trying NCHW:', e.message);
      layout = 'NCHW'; await run(new Float32Array(CONFIG.SIZE * CONFIG.SIZE * 3));
    }
    console.log('Layout:', layout);
    mode = 'browser'; setStatus('ok', 'Model ready · browser');
  } catch (e) {
    console.error('Model load error:', e);
    mode = 'none';
    const m = String(e.message || e);
    setStatus('err', 'Error: ' + m.slice(0, 60));
    toast('Model error: ' + m.slice(0, 140));
  }
}

/* ---------- prediction ---------- */
async function run(chw_or_hwc) {
  const S = CONFIG.SIZE;
  const dims = layout === 'NHWC' ? [1, S, S, 3] : [1, 3, S, S];
  const feeds = { [session.inputNames[0]]: new ort.Tensor('float32', chw_or_hwc, dims) };
  const out = await session.run(feeds);
  return out[session.outputNames[0]].data;
}
function pixelsFrom(src) {
  const S = CONFIG.SIZE;
  const c = document.createElement('canvas'); c.width = c.height = S;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(src, 0, 0, S, S);
  const d = ctx.getImageData(0, 0, S, S).data;
  const f = new Float32Array(S * S * 3);
  const norm = CONFIG.PREPROCESS === 'mobilenet' ? v => v / 127.5 - 1
             : CONFIG.PREPROCESS === 'unit' ? v => v / 255 : v => v;
  for (let i = 0; i < S * S; i++) {
    const r = norm(d[i * 4]), g = norm(d[i * 4 + 1]), b = norm(d[i * 4 + 2]);
    if (layout === 'NHWC') { f[i * 3] = r; f[i * 3 + 1] = g; f[i * 3 + 2] = b; }
    else { f[i] = r; f[S * S + i] = g; f[2 * S * S + i] = b; }
  }
  return f;
}
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
async function predictDog(imgEl) {
  if (mode !== 'browser') throw new Error('Model not loaded');
  const o = await run(pixelsFrom(imgEl));
  console.log('raw output:', Array.from(o));
  if (o.length === 1) return (o[0] >= 0 && o[0] <= 1) ? o[0] : sigmoid(o[0]);
  const a = Array.from(o), sum = a.reduce((x, y) => x + y, 0);
  if (Math.abs(sum - 1) < 1e-3 && a.every(v => v >= 0)) return a[CONFIG.DOG_INDEX];
  const e = a.map(v => Math.exp(v - Math.max(...a))), z = e.reduce((x, y) => x + y, 0);
  return e[CONFIG.DOG_INDEX] / z;
}

/* ---------- UI render ---------- */
function showResult(pDog) {
  const isDog = pDog > 0.5, conf = isDog ? pDog : 1 - pDog;
  el.verdict.textContent = isDog ? 'Dog' : 'Cat';
  el.verdict.classList.remove('empty');
  el.confBar.style.width = (conf * 100) + '%';
  el.confVal.textContent = (conf * 100).toFixed(1) + '%';
  el.confLbl.textContent = conf > .9 ? 'Very confident' : conf > .7 ? 'Fairly confident' : 'Unsure';
  el.pCat.textContent = ((1 - pDog) * 100).toFixed(1) + '%';
  el.pDog.textContent = (pDog * 100).toFixed(1) + '%';
  return { label: isDog ? 'dog' : 'cat', conf };
}
function resetResult() {
  el.verdict.textContent = '—'; el.verdict.classList.add('empty');
  el.confBar.style.width = '0'; el.confVal.textContent = '0.0%'; el.confLbl.textContent = 'Confidence';
  el.pCat.textContent = el.pDog.textContent = '–';
}
function thumb(imgEl) {
  const c = document.createElement('canvas'); c.width = c.height = 80;
  const s = Math.min(imgEl.naturalWidth || imgEl.videoWidth, imgEl.naturalHeight || imgEl.videoHeight);
  const w = imgEl.naturalWidth || imgEl.videoWidth, h = imgEl.naturalHeight || imgEl.videoHeight;
  c.getContext('2d').drawImage(imgEl, (w - s) / 2, (h - s) / 2, s, s, 0, 0, 80, 80);
  return c.toDataURL('image/jpeg', .6);
}
function renderHistory() {
  const h = store.get('pp-hist', []);
  el.hist.innerHTML = h.length ? '' : '<span class="mono" style="color:var(--mute)">Nothing yet.</span>';
  h.forEach(i => {
    const f = document.createElement('figure');
    f.innerHTML = `<img src="${i.img}" alt="${i.label}"><figcaption>${i.label} ${Math.round(i.conf * 100)}</figcaption>`;
    el.hist.appendChild(f);
  });
}
el.clear.onclick = (e) => { e.preventDefault(); store.set('pp-hist', []); renderHistory(); };

/* ---------- flow ---------- */
async function analyse(imgEl) {
  if (mode === 'none') { toast('Model not loaded yet'); return; }
  el.drop.classList.add('scanning');
  try {
    await new Promise(r => setTimeout(r, 450)); // let the scan animation breathe
    const p = await predictDog(imgEl);
    const r = showResult(p);
    const h = store.get('pp-hist', []);
    h.unshift({ img: thumb(imgEl), label: r.label, conf: r.conf });
    store.set('pp-hist', h.slice(0, 12)); renderHistory();
  } catch (e) { console.error(e); toast('Prediction failed: ' + e.message); }
  el.drop.classList.remove('scanning');
}
function loadFile(file) {
  if (!file || !file.type.startsWith('image/')) { toast('Please choose an image file'); return; }
  stopCam();
  const url = URL.createObjectURL(file);
  el.preview.onload = () => analyse(el.preview);
  el.preview.src = url; el.preview.hidden = false; el.drop.classList.add('has');
  resetResult();
}

/* ---------- camera ---------- */
async function startCam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    el.cam.srcObject = stream; el.cam.hidden = false; el.preview.hidden = true;
    el.drop.classList.add('has'); el.snap.hidden = false; el.camBtn.textContent = 'Stop camera';
  } catch { toast('Camera unavailable or permission denied'); }
}
function stopCam() {
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null; el.cam.hidden = true; el.snap.hidden = true; el.camBtn.textContent = 'Use camera';
}
el.camBtn.onclick = () => stream ? (stopCam(), el.drop.classList.toggle('has', !el.preview.hidden)) : startCam();
el.snap.onclick = async () => {
  const c = document.createElement('canvas');
  c.width = el.cam.videoWidth; c.height = el.cam.videoHeight;
  c.getContext('2d').drawImage(el.cam, 0, 0);
  const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', .95));
  loadFile(new File([blob], 'snap.jpg', { type: 'image/jpeg' }));
};

/* ---------- events ---------- */
el.pick.onclick = () => el.file.click();
el.drop.onclick = () => { if (!stream) el.file.click(); };
el.drop.onkeydown = (e) => { if (e.key === 'Enter') el.file.click(); };
el.file.onchange = () => loadFile(el.file.files[0]);
['dragenter', 'dragover'].forEach(t => el.drop.addEventListener(t, e => { e.preventDefault(); el.drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(t => el.drop.addEventListener(t, e => { e.preventDefault(); el.drop.classList.remove('over'); }));
el.drop.addEventListener('drop', e => loadFile(e.dataTransfer.files[0]));
window.addEventListener('paste', e => {
  const f = [...(e.clipboardData?.files || [])].find(f => f.type.startsWith('image/'));
  if (f) loadFile(f);
});
el.reset.onclick = () => {
  stopCam(); el.preview.hidden = true; el.preview.removeAttribute('src');
  el.drop.classList.remove('has'); el.file.value = ''; resetResult();
};

renderHistory();
loadModel();
