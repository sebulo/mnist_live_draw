const MODEL_URLS = [
  // Publicly hosted MNIST TFJS model (10-class).
  "https://iwatake2222.github.io/tfjs_study/mnist/conv_mnist_tfjs/model.json",
];

const drawCanvas = document.getElementById("drawCanvas");
const transformCanvas = document.getElementById("transformCanvas");
const mnistCanvas = document.getElementById("mnistCanvas");
const modelStatus = document.getElementById("modelStatus");
const topPrediction = document.getElementById("topPrediction");
const latencyEl = document.getElementById("latency");
const barsEl = document.getElementById("bars");
const clearBtn = document.getElementById("clearBtn");
const fullscreenBtn = document.getElementById("fullscreenBtn");
const brushSize = document.getElementById("brushSize");
const brushSizeValue = document.getElementById("brushSizeValue");
const eraser = document.getElementById("eraser");
const rotation = document.getElementById("rotation");
const rotationValue = document.getElementById("rotationValue");
const scale = document.getElementById("scale");
const scaleValue = document.getElementById("scaleValue");
const translateX = document.getElementById("translateX");
const translateXValue = document.getElementById("translateXValue");
const translateY = document.getElementById("translateY");
const translateYValue = document.getElementById("translateYValue");
const blur = document.getElementById("blur");
const blurValue = document.getElementById("blurValue");
const invert = document.getElementById("invert");
const realtime = document.getElementById("realtime");
const classifyBtn = document.getElementById("classifyBtn");

const drawCtx = drawCanvas.getContext("2d", { willReadFrequently: true });
const transformCtx = transformCanvas.getContext("2d", { willReadFrequently: true });
const mnistCtx = mnistCanvas.getContext("2d", { willReadFrequently: true });
const mnistSmall = document.createElement("canvas");
mnistSmall.width = 28;
mnistSmall.height = 28;
const mnistSmallCtx = mnistSmall.getContext("2d");

let model = null;
let isDrawing = false;
let lastPoint = null;
let predictTimer = null;

const DIGITS = Array.from({ length: 10 }, (_, i) => i);

function initBars() {
  barsEl.innerHTML = "";
  DIGITS.forEach((digit) => {
    const row = document.createElement("div");
    row.className = "bar";
    row.innerHTML = `
      <div>${digit}</div>
      <div class="bar-fill"><span style="width: 0%"></span></div>
      <div class="percent">0%</div>
    `;
    barsEl.appendChild(row);
  });
}

function setCanvasDefaults() {
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.strokeStyle = "#ffffff";
  drawCtx.fillStyle = "#000000";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
}

function clearCanvas() {
  drawCtx.fillStyle = "#000000";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  lastPoint = null;
  updatePipeline();
  resetPredictions();
}

function getPointerPos(event) {
  const rect = drawCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (drawCanvas.width / rect.width);
  const y = (event.clientY - rect.top) * (drawCanvas.height / rect.height);
  return { x, y };
}

function startDraw(event) {
  isDrawing = true;
  lastPoint = getPointerPos(event);
}

function draw(event) {
  if (!isDrawing) return;
  const pos = getPointerPos(event);
  drawCtx.lineWidth = Number(brushSize.value);
  drawCtx.strokeStyle = eraser.checked ? "#000000" : "#ffffff";
  drawCtx.beginPath();
  drawCtx.moveTo(lastPoint.x, lastPoint.y);
  drawCtx.lineTo(pos.x, pos.y);
  drawCtx.stroke();
  lastPoint = pos;
  updatePipeline();
  schedulePrediction();
}

function endDraw() {
  isDrawing = false;
  lastPoint = null;
}

function updatePipeline() {
  renderTransformed();
  preprocessTo28();
}

function renderTransformed() {
  const w = transformCanvas.width;
  const h = transformCanvas.height;
  const rot = (Number(rotation.value) * Math.PI) / 180;
  const sc = Number(scale.value) / 100;
  const tx = Number(translateX.value);
  const ty = Number(translateY.value);
  const blurPx = Number(blur.value);

  transformCtx.save();
  transformCtx.clearRect(0, 0, w, h);
  transformCtx.fillStyle = "#000000";
  transformCtx.fillRect(0, 0, w, h);
  transformCtx.filter = blurPx > 0 ? `blur(${blurPx}px)` : "none";
  transformCtx.translate(w / 2 + tx, h / 2 + ty);
  transformCtx.rotate(rot);
  transformCtx.scale(sc, sc);
  transformCtx.drawImage(drawCanvas, -w / 2, -h / 2);
  transformCtx.restore();
  transformCtx.filter = "none";

  if (invert.checked) {
    const img = transformCtx.getImageData(0, 0, w, h);
    const data = img.data;
    for (let i = 0; i < data.length; i += 4) {
      data[i] = 255 - data[i];
      data[i + 1] = 255 - data[i + 1];
      data[i + 2] = 255 - data[i + 2];
    }
    transformCtx.putImageData(img, 0, 0);
  }
}

function preprocessTo28() {
  const w = transformCanvas.width;
  const h = transformCanvas.height;
  const img = transformCtx.getImageData(0, 0, w, h);
  const data = img.data;

  let minX = w;
  let minY = h;
  let maxX = 0;
  let maxY = 0;
  let found = false;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const value = data[idx];
      const intensity = invert.checked ? 255 - value : value;
      if (intensity > 10) {
        found = true;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  mnistSmallCtx.clearRect(0, 0, 28, 28);
  mnistSmallCtx.fillStyle = invert.checked ? "#ffffff" : "#000000";
  mnistSmallCtx.fillRect(0, 0, 28, 28);

  if (!found) {
    drawMnistPreview();
    return null;
  }

  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;
  const scaleFactor =
    (20 * (Number(scale.value) / 100)) / Math.max(boxW, boxH);
  const targetW = boxW * scaleFactor;
  const targetH = boxH * scaleFactor;
  // Center into 28x28, then apply translation in MNIST space.
  const baseDx = (28 - targetW) / 2;
  const baseDy = (28 - targetH) / 2;
  const tx28 = (Number(translateX.value) / transformCanvas.width) * 28;
  const ty28 = (Number(translateY.value) / transformCanvas.height) * 28;
  const dx = baseDx + tx28;
  const dy = baseDy + ty28;

  mnistSmallCtx.imageSmoothingEnabled = true;
  mnistSmallCtx.drawImage(
    transformCanvas,
    minX,
    minY,
    boxW,
    boxH,
    dx,
    dy,
    targetW,
    targetH
  );

  drawMnistPreview();
  return mnistSmallCtx.getImageData(0, 0, 28, 28);
}

function drawMnistPreview() {
  mnistCtx.imageSmoothingEnabled = false;
  mnistCtx.clearRect(0, 0, mnistCanvas.width, mnistCanvas.height);
  mnistCtx.drawImage(mnistSmall, 0, 0, mnistCanvas.width, mnistCanvas.height);
}

function getInputTensor() {
  const imageData = preprocessTo28();
  if (!imageData) return null;
  const { data } = imageData;
  const floats = new Float32Array(28 * 28);
  for (let i = 0; i < data.length; i += 4) {
    floats[i / 4] = data[i] / 255;
  }
  return tf.tensor4d(floats, [1, 28, 28, 1]);
}

async function runPrediction() {
  if (!model) return;
  const input = getInputTensor();
  if (!input) {
    resetPredictions();
    return;
  }

  const start = performance.now();
  const preds = model.predict(input);
  const values = await preds.data();
  const elapsed = performance.now() - start;
  latencyEl.textContent = `${elapsed.toFixed(1)} ms`;
  updateBars(values);

  tf.dispose([preds, input]);
}

function updateBars(values) {
  let bestIndex = 0;
  let bestVal = values[0];
  const rows = barsEl.querySelectorAll(".bar");
  rows.forEach((row, idx) => {
    const percent = Math.round(values[idx] * 100);
    row.querySelector(".bar-fill span").style.width = `${percent}%`;
    row.querySelector(".percent").textContent = `${percent}%`;
    if (values[idx] > bestVal) {
      bestVal = values[idx];
      bestIndex = idx;
    }
  });
  topPrediction.textContent = `${bestIndex} (${Math.round(bestVal * 100)}%)`;
}

function resetPredictions() {
  latencyEl.textContent = "—";
  topPrediction.textContent = "—";
  const rows = barsEl.querySelectorAll(".bar");
  rows.forEach((row) => {
    row.querySelector(".bar-fill span").style.width = "0%";
    row.querySelector(".percent").textContent = "0%";
  });
}

function schedulePrediction() {
  if (!realtime.checked) return;
  clearTimeout(predictTimer);
  predictTimer = setTimeout(runPrediction, 80);
}

function updateUIValues() {
  brushSizeValue.textContent = brushSize.value;
  rotationValue.textContent = rotation.value;
  scaleValue.textContent = (Number(scale.value) / 100).toFixed(2);
  translateXValue.textContent = translateX.value;
  translateYValue.textContent = translateY.value;
  blurValue.textContent = blur.value;
}

function bindEvents() {
  drawCanvas.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    drawCanvas.setPointerCapture(event.pointerId);
    startDraw(event);
  });

  drawCanvas.addEventListener("pointermove", draw);
  drawCanvas.addEventListener("pointerup", endDraw);
  drawCanvas.addEventListener("pointerleave", endDraw);
  drawCanvas.addEventListener("pointercancel", endDraw);

  clearBtn.addEventListener("click", clearCanvas);
  fullscreenBtn.addEventListener("click", toggleFullscreen);

  [brushSize, eraser, rotation, scale, translateX, translateY, blur, invert].forEach(
    (control) => {
      control.addEventListener("input", () => {
        updateUIValues();
        updatePipeline();
        schedulePrediction();
      });
    }
  );

  realtime.addEventListener("change", () => {
    classifyBtn.disabled = realtime.checked;
    if (realtime.checked) schedulePrediction();
  });

  classifyBtn.addEventListener("click", runPrediction);
}

function toggleFullscreen() {
  const wrap = drawCanvas.closest(".canvas-wrap");
  if (!document.fullscreenElement) {
    if (wrap.requestFullscreen) {
      wrap.requestFullscreen();
    }
  } else {
    document.exitFullscreen();
  }
}

document.addEventListener("fullscreenchange", () => {
  const wrap = drawCanvas.closest(".canvas-wrap");
  if (!wrap) return;
  if (document.fullscreenElement) {
    wrap.classList.add("fullscreen");
    fullscreenBtn.textContent = "Exit Fullscreen";
  } else {
    wrap.classList.remove("fullscreen");
    fullscreenBtn.textContent = "Fullscreen";
  }
});

function isTenClassModel(loadedModel) {
  const outputShape = loadedModel.outputs?.[0]?.shape;
  return Array.isArray(outputShape) && outputShape[outputShape.length - 1] === 10;
}

async function loadModel() {
  modelStatus.textContent = "Loading model…";
  for (const url of MODEL_URLS) {
    try {
      const loaded = await tf.loadLayersModel(url);
      if (!isTenClassModel(loaded)) {
        console.warn("Model output shape not 10-class", url, loaded.outputs);
        tf.dispose(loaded);
        continue;
      }
      model = loaded;
      modelStatus.textContent = "Model ready";
      classifyBtn.disabled = realtime.checked;
      return;
    } catch (err) {
      console.warn("Model load failed", url, err);
    }
  }
  modelStatus.textContent = "Model failed to load (check console)";
}

function init() {
  initBars();
  setCanvasDefaults();
  updateUIValues();
  updatePipeline();
  bindEvents();
  loadModel();
}

init();
