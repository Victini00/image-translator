// 이미지 번역기 웹 UI 클라이언트 로직
//
// 좌표계 주의:
//   서버가 주는 box/폰트 크기/줄 위치는 전부 "원본 이미지 픽셀 좌표" 기준이다.
//   캔버스도 원본 해상도로 잡고 CSS로만 축소해서 보여주므로, 그리는 코드에서는
//   배율을 신경 쓸 필요가 없다. 마우스 좌표만 scale로 나눠 원본 좌표로 바꾼다.
//
// 미리보기 = 최종 결과:
//   폰트 크기·줄바꿈·각 줄의 위치를 전부 서버(/api/fit)가 실제 렌더링과 똑같은
//   함수로 계산해서 내려준다. 브라우저는 그 좌표 그대로 canvas에 fillText 할 뿐이라
//   PIL이 그린 결과와 사실상 같은 그림이 된다.

const state = {
  uploadedName: null,
  base: null,
  outKey: null,
  imageWidth: 0,
  imageHeight: 0,
  paragraphs: [],     // {id, text, box, is_interior, font_size, fit}
  downloadName: null,
  config: null,
  pollTimer: null,
  tool: "select",     // "select" | "paint"
  selectedId: null,
  paintHistory: [],   // 되돌리기용 스냅샷
};

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------ 공통 유틸

function setStatus(el, message, kind = "") {
  el.textContent = message;
  el.className = "status" + (kind ? " " + kind : "");
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || `요청 실패 (${res.status})`);
    err.log = data.log;
    throw err;
  }
  return data;
}

function showLog(text) {
  if (!text) return;
  $("log-block").hidden = false;
  $("log-output").textContent = text;
  $("log-output").scrollTop = $("log-output").scrollHeight;
}

function selectedModel() {
  return document.querySelector('input[name="model"]:checked').value;
}

/** 화면 표시 배율 (표시 크기 / 원본 크기) */
function displayScale() {
  const img = $("stage-img");
  return img.naturalWidth ? img.clientWidth / img.naturalWidth : 1;
}

// ------------------------------------------- 렌더링 설정 / 폰트 로딩

async function loadRenderConfig() {
  state.config = await (await fetch("/api/render-config")).json();
}

/** 실제 렌더링에 쓰는 폰트를 브라우저에 등록한다(캔버스에서 같은 글꼴로 그리려고). */
async function loadFonts() {
  const defs = [
    ["ITInterior", "/api/font/interior"],
    ["ITExterior", "/api/font/exterior"],
  ];
  await Promise.all(defs.map(async ([family, url]) => {
    try {
      const face = new FontFace(family, `url(${url})`);
      await face.load();
      document.fonts.add(face);
    } catch (e) {
      console.warn(`폰트 로드 실패: ${family}`, e);
    }
  }));
}

// ------------------------------------------------------------ API 키

async function loadApiKeyState() {
  try {
    const data = await (await fetch("/api/apikey")).json();
    setStatus($("apikey-current"),
      data.has_key ? `저장된 키: ${data.masked}` : "저장된 키 없음",
      data.has_key ? "ok" : "");
  } catch {
    setStatus($("apikey-current"), "키 상태를 확인하지 못했습니다.", "err");
  }
}

$("apikey-save").addEventListener("click", async () => {
  const key = $("apikey-input").value.trim();
  if (!key) {
    setStatus($("apikey-current"), "키를 입력해 주세요.", "err");
    return;
  }
  try {
    const data = await postJSON("/api/apikey", { key });
    $("apikey-input").value = "";
    setStatus($("apikey-current"), `저장됨: ${data.masked}`, "ok");
  } catch (e) {
    setStatus($("apikey-current"), e.message, "err");
  }
});

// ------------------------------------------------------------ 업로드

$("file-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  setStatus($("upload-status"), "업로드 중...", "busy");
  const form = new FormData();
  form.append("image", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "업로드 실패");
    state.uploadedName = data.name;
    state.base = data.base;
    setStatus($("upload-status"), `선택됨: ${data.name}`, "ok");
    $("run-btn").disabled = false;
  } catch (err) {
    setStatus($("upload-status"), err.message, "err");
    $("run-btn").disabled = true;
  }
});

// ------------------------------------------------- 진행률 / 폴백 안내

async function showStagePreview() {
  try {
    const data = await (await fetch("/api/models")).json();
    renderProgress(data.models[selectedModel()].stages, 0, null, false);
  } catch (e) {
    console.warn("단계 목록을 불러오지 못했습니다", e);
  }
}

function renderProgress(stages, currentIndex, elapsed, finished) {
  const box = $("progress-box");
  box.hidden = false;
  box.innerHTML = "";

  stages.forEach((name, i) => {
    const step = i + 1;
    const row = document.createElement("div");
    let cls = "pending", icon = "○";
    if (finished || step < currentIndex) { cls = "done"; icon = "●"; }
    else if (step === currentIndex) { cls = "active"; icon = "◐"; }
    row.className = `progress-step ${cls}`;
    row.innerHTML = `<span class="dot">${icon}</span><span>${step}. ${name}</span>`;
    box.appendChild(row);
  });

  if (elapsed !== null && elapsed !== undefined) {
    const foot = document.createElement("div");
    foot.className = "progress-elapsed";
    foot.textContent = finished ? `완료 (${elapsed}초)` : `경과 ${elapsed}초`;
    box.appendChild(foot);
  }
}

function showFallbackNotice(fallback) {
  const box = $("fallback-notice");
  if (!fallback) {
    box.hidden = true;
    return;
  }
  box.className = "notice-box" + (fallback.kind === "all" ? " all" : "");
  box.textContent = fallback.message;
  box.hidden = false;
}

// ------------------------------------------------------------ 파이프라인 실행

$("run-btn").addEventListener("click", async () => {
  if (!state.uploadedName) return;

  $("run-btn").disabled = true;
  $("fallback-notice").hidden = true;
  setStatus($("run-status"), "실행 시작...", "busy");

  try {
    const started = await postJSON("/api/run", {
      name: state.uploadedName, model: selectedModel(),
    });
    pollProgress(started.job_id, started.stages);
  } catch (err) {
    setStatus($("run-status"), err.message, "err");
    if (err.log) showLog(err.log);
    $("run-btn").disabled = false;
  }
});

function pollProgress(jobId, stages) {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    let job;
    try {
      const res = await fetch(`/api/progress/${jobId}`);
      job = await res.json();
      if (!res.ok) throw new Error(job.error || "진행 상황 조회 실패");
    } catch (e) {
      clearInterval(state.pollTimer);
      setStatus($("run-status"), e.message, "err");
      $("run-btn").disabled = false;
      return;
    }

    renderProgress(job.stages || stages, job.stage_index || 0, job.elapsed,
                   job.status === "done");

    if (job.status === "running") {
      setStatus($("run-status"),
        `${job.stage_index}/${job.stage_total} ${job.stage_name} 진행 중...`, "busy");
      return;
    }

    clearInterval(state.pollTimer);
    $("run-btn").disabled = false;
    showLog(job.log);

    if (job.status === "error") {
      setStatus($("run-status"), job.error, "err");
      return;
    }

    const r = job.result;
    state.outKey = r.out_key;
    state.base = r.base;
    state.imageWidth = r.image_width;
    state.imageHeight = r.image_height;
    state.paragraphs = r.paragraphs.map((p) => ({ ...p, font_size: null, fit: null }));
    state.downloadName = r.rendered_name;

    setStatus($("run-status"), `완료 — 문단 ${r.paragraphs.length}개`, "ok");
    showFallbackNotice(job.fallback);
    buildEditor(r.cleaned_url);
    showResult(r.rendered_url, `${r.out_key}/${r.rendered_name}`);
    switchView("editor");
  }, 700);
}

// ------------------------------------------------------------ 편집기 구성

function buildEditor(cleanedUrl) {
  const stage = $("stage");
  const img = $("stage-img");

  [...stage.querySelectorAll(".para-box")].forEach((el) => el.remove());
  $("editor-empty").hidden = true;
  $("stage-wrap").hidden = false;
  state.selectedId = null;
  state.paintHistory = [];

  img.onload = () => {
    // 캔버스는 원본 해상도로 잡고 CSS로만 축소한다 (좌표 = 원본 픽셀)
    [$("paint-canvas"), $("text-canvas")].forEach((c) => {
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
    });
    $("paint-canvas").getContext("2d").clearRect(0, 0, img.naturalWidth, img.naturalHeight);
    layoutBoxes();
    $("rerender-btn").disabled = false;
    updateFontSizeControls();
  };
  img.src = cleanedUrl;

  state.paragraphs.forEach((para) => stage.appendChild(createBox(para)));
}

/** 박스 DOM 위치를 화면 배율에 맞춰 갱신하고, 서버에 조판을 요청한다. */
function layoutBoxes() {
  const scale = displayScale();
  state.paragraphs.forEach((para) => {
    const el = document.querySelector(`.para-box[data-id="${para.id}"]`);
    if (el) placeBox(el, para, scale);
  });
  syncLayoutWithServer();
}

function placeBox(el, para, scale = displayScale()) {
  const [x1, y1, x2, y2] = para.box;
  el.style.left = `${x1 * scale}px`;
  el.style.top = `${y1 * scale}px`;
  el.style.width = `${(x2 - x1) * scale}px`;
  el.style.height = `${(y2 - y1) * scale}px`;
}

/**
 * 서버에 실제 fit_text() 계산을 요청해서 각 문단의 조판 결과를 받아온다.
 * 받은 좌표 그대로 캔버스에 그리므로 최종 렌더링과 같은 그림이 된다.
 */
let syncTimer = null;
function syncLayoutWithServer(delay = 120, onDone = null) {
  clearTimeout(syncTimer);
  syncTimer = setTimeout(async () => {
    if (!state.paragraphs.length) return;
    try {
      const data = await postJSON("/api/fit", {
        items: state.paragraphs.map((p) => ({
          id: p.id, text: p.text, box: p.box,
          is_interior: p.is_interior, font_size: p.font_size,
        })),
      });
      const byId = new Map(data.items.map((f) => [f.id, f]));
      state.paragraphs.forEach((p) => { p.fit = byId.get(p.id) || p.fit; });
    } catch (e) {
      console.warn("서버 배치 계산 실패", e);
    }
    drawTextLayer();
    updateFontSizeControls();
    if (onDone) onDone();
  }, delay);
}

/** 문단 텍스트를 캔버스에 그린다 (PIL draw_paragraph_text와 동일한 좌표 계산). */
function drawTextLayer() {
  const canvas = $("text-canvas");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";

  state.paragraphs.forEach((para) => {
    const fit = para.fit;
    if (!fit) return;
    const [bx1, by1] = para.box;
    const family = para.is_interior ? "ITInterior" : "ITExterior";
    ctx.font = `${fit.size}px "${family}"`;

    fit.lines.forEach((line, i) => {
      if (!line) return;
      // PIL은 (x, y)를 글자 윗선 기준으로 받고, canvas는 베이스라인 기준이라 ascent를 더한다
      const x = bx1 + fit.line_x[i];
      const y = by1 + fit.block_y + i * fit.line_h + fit.ascent;
      if (fit.stroke > 0) {
        // PIL stroke_width는 글자 바깥으로 그만큼 번지므로 선 굵기는 2배로 잡는다
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = fit.stroke * 2;
        ctx.lineJoin = "round";
        ctx.strokeText(line, x, y);
      }
      ctx.fillStyle = "#141414";
      ctx.fillText(line, x, y);
    });
  });
}

// ------------------------------------------------------------ 박스 조작

function createBox(para) {
  const el = document.createElement("div");
  el.className = "para-box" + (para.is_interior ? "" : " exterior");
  el.dataset.id = para.id;
  el.title = `원문: ${para.source_text || "(없음)"}`;

  const handle = document.createElement("div");
  handle.className = "resize-handle";
  el.appendChild(handle);

  el.addEventListener("dblclick", (e) => {
    e.stopPropagation();
    startEditing(el, para);
  });

  el.addEventListener("mousedown", (e) => {
    selectBox(para.id);
    if (e.target === handle) return;
    startDrag(e, el, para, "move");
  });
  handle.addEventListener("mousedown", (e) => {
    e.stopPropagation();
    startDrag(e, el, para, "resize");
  });

  return el;
}

function selectBox(id) {
  state.selectedId = id;
  document.querySelectorAll(".para-box").forEach((n) => {
    n.classList.toggle("selected", Number(n.dataset.id) === id);
  });
  updateFontSizeControls();
}

function currentPara() {
  return state.paragraphs.find((p) => p.id === state.selectedId) || null;
}

function updateFontSizeControls() {
  const para = currentPara();
  const input = $("font-size-input");
  const auto = $("font-size-auto");
  input.disabled = !para;
  auto.disabled = !para || !para.font_size;
  if (!para) {
    input.value = "";
    input.placeholder = "자동";
    return;
  }
  input.value = para.font_size || "";
  input.placeholder = para.fit ? `자동 (${para.fit.size})` : "자동";
}

$("font-size-input").addEventListener("input", () => {
  const para = currentPara();
  if (!para) return;
  const value = parseInt($("font-size-input").value, 10);
  para.font_size = Number.isFinite(value) && value > 0 ? value : null;
  $("font-size-auto").disabled = !para.font_size;
  syncLayoutWithServer();
});

$("font-size-auto").addEventListener("click", () => {
  const para = currentPara();
  if (!para) return;
  para.font_size = null;
  $("font-size-input").value = "";
  $("font-size-auto").disabled = true;
  syncLayoutWithServer();
});

/**
 * 편집 패널을 박스와 겹치지 않는 자리에 놓는다.
 * 아래 -> 위 -> 오른쪽 -> 왼쪽 순으로 시도하고, 다 안 되면 겹침이 가장 적은 자리를 쓴다.
 */
function placeEditorPanel(panel, para, scale) {
  const stage = $("stage");
  const SW = stage.clientWidth, SH = stage.clientHeight;
  const gap = 8;

  const width = Math.min(260, Math.max(180, (para.box[2] - para.box[0]) * scale));
  panel.style.width = `${width}px`;
  const h = panel.offsetHeight;   // 내용이 다 들어간 뒤의 실제 높이

  const bx1 = para.box[0] * scale, by1 = para.box[1] * scale;
  const bx2 = para.box[2] * scale, by2 = para.box[3] * scale;
  const clamp = (v, max) => Math.max(0, Math.min(v, Math.max(0, max)));

  const candidates = [
    { left: clamp(bx1, SW - width), top: by2 + gap },                 // 아래
    { left: clamp(bx1, SW - width), top: by1 - h - gap },             // 위
    { left: bx2 + gap, top: clamp(by1, SH - h) },                     // 오른쪽
    { left: bx1 - width - gap, top: clamp(by1, SH - h) },             // 왼쪽
  ];

  const overlapArea = (c) => {
    const ox = Math.max(0, Math.min(c.left + width, bx2) - Math.max(c.left, bx1));
    const oy = Math.max(0, Math.min(c.top + h, by2) - Math.max(c.top, by1));
    return ox * oy;
  };
  const inside = (c) => c.left >= 0 && c.top >= 0 && c.left + width <= SW && c.top + h <= SH;

  let best = candidates.find((c) => inside(c) && overlapArea(c) === 0);
  if (!best) {
    // 완벽한 자리가 없으면 화면 안에 들어오면서 겹침이 최소인 곳으로
    best = candidates
      .map((c) => ({ left: clamp(c.left, SW - width), top: clamp(c.top, SH - h) }))
      .sort((a, b) => overlapArea(a) - overlapArea(b))[0];
  }

  panel.style.left = `${best.left}px`;
  panel.style.top = `${best.top}px`;
}

/**
 * 더블클릭 편집: 박스를 가리지 않도록 편집창을 박스 바깥(아래/위/옆)에 띄우고,
 * 타이핑하는 동안 캔버스의 실제 조판을 실시간으로 갱신한다.
 * 그래서 "이렇게 치면 어디서 줄이 바뀌는지"를 편집하면서 바로 눈으로 볼 수 있다.
 */
function startEditing(el, para) {
  if (document.querySelector(".para-editor")) return;
  const stage = $("stage");
  const scale = displayScale();
  const original = para.text;

  const panel = document.createElement("div");
  panel.className = "para-editor";

  const editor = document.createElement("textarea");
  editor.value = para.text;
  editor.spellcheck = false;
  panel.appendChild(editor);

  const info = document.createElement("div");
  info.className = "editor-info";
  panel.appendChild(info);

  const keys = document.createElement("div");
  keys.className = "editor-keys";
  keys.innerHTML = "<b>Enter</b> 줄바꿈 · <b>Ctrl+Enter</b> 완료 · <b>Esc</b> 취소";
  panel.appendChild(keys);

  const refreshInfo = () => {
    const fit = para.fit;
    info.textContent = fit
      ? `적용 결과: ${fit.lines.length}줄 · 글자 크기 ${fit.size}px`
      : "계산 중...";
  };
  refreshInfo();
  stage.appendChild(panel);
  el.classList.add("editing-target");

  // 내용을 다 채운 뒤에 실제 크기를 재서, 박스를 가리지 않는 자리에 놓는다
  placeEditorPanel(panel, para, scale);

  // 타이핑할 때마다 서버 조판을 다시 받아 캔버스를 갱신 (실시간 미리보기)
  editor.addEventListener("input", () => {
    para.text = editor.value.replace(/\r/g, "");
    syncLayoutWithServer(220, refreshInfo);
  });

  const finish = (save) => {
    if (!save) {
      para.text = original;
      syncLayoutWithServer(0);
    }
    el.classList.remove("editing-target");
    panel.remove();
  };

  editor.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "Escape") { e.preventDefault(); finish(false); }
    // Enter는 줄바꿈으로 그대로 쓰고, 편집 종료는 Ctrl+Enter로
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); finish(true); }
  });
  // 편집창 안을 클릭하는 동안에는 닫히지 않게 한다
  panel.addEventListener("mousedown", (e) => e.stopPropagation());
  editor.addEventListener("blur", () => setTimeout(() => {
    if (document.activeElement !== editor) finish(true);
  }, 120));

  editor.focus();
  editor.select();
}

function startDrag(e, el, para, mode) {
  e.preventDefault();
  const scale = displayScale();
  const startX = e.clientX;
  const startY = e.clientY;
  const [ox1, oy1, ox2, oy2] = para.box;

  const onMove = (ev) => {
    const dx = (ev.clientX - startX) / scale;
    const dy = (ev.clientY - startY) / scale;

    let box;
    if (mode === "move") {
      box = [ox1 + dx, oy1 + dy, ox2 + dx, oy2 + dy];
    } else {
      box = [ox1, oy1, Math.max(ox1 + 12, ox2 + dx), Math.max(oy1 + 12, oy2 + dy)];
    }

    const w = box[2] - box[0];
    const h = box[3] - box[1];
    box[0] = Math.max(0, Math.min(box[0], state.imageWidth - w));
    box[1] = Math.max(0, Math.min(box[1], state.imageHeight - h));
    box[2] = Math.min(state.imageWidth, box[0] + w);
    box[3] = Math.min(state.imageHeight, box[1] + h);

    para.box = box;
    placeBox(el, para, scale);
  };

  const onUp = () => {
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
    syncLayoutWithServer(0);
  };

  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
}

window.addEventListener("resize", () => {
  const scale = displayScale();
  state.paragraphs.forEach((para) => {
    const el = document.querySelector(`.para-box[data-id="${para.id}"]`);
    if (el) placeBox(el, para, scale);
  });
});

// ------------------------------------------------------------ 덧칠(브러시) 도구

document.querySelectorAll(".tool-btn").forEach((btn) => {
  btn.addEventListener("click", () => setTool(btn.dataset.tool));
});

function setTool(tool) {
  state.tool = tool;
  document.querySelectorAll(".tool-btn").forEach((b) => {
    b.classList.toggle("active", b.dataset.tool === tool);
  });
  document.querySelectorAll(".paint-only").forEach((n) => { n.hidden = tool !== "paint"; });
  document.querySelectorAll(".select-only").forEach((n) => { n.hidden = tool === "paint"; });
  $("stage").classList.toggle("paint-mode", tool === "paint");
  $("brush-cursor").hidden = true;
  if (tool === "paint") { updateBrushCursor(); peekBrushCursor(); }
  $("editor-hint").textContent = tool === "paint"
    ? "드래그해서 색을 칠하면 그 부분이 덮입니다. detection이 놓쳐서 원본 글자가 남은 곳을 지울 때 쓰세요. 칠한 내용은 '편집 반영해서 다시 렌더링'을 눌러야 결과에 적용됩니다."
    : "박스를 더블클릭하면 글자 수정(Enter로 줄바꿈, Ctrl+Enter로 완료), 드래그하면 위치 이동, 오른쪽 아래 모서리를 끌면 박스 크기 조절. 박스를 선택하면 글자 크기를 직접 지정할 수 있습니다.";
}

/**
 * 브러시 크기/색을 실제 칠하기 전에 눈으로 가늠할 수 있게 커서 원을 띄운다.
 * 밝은 색이면 어두운 테두리, 어두운 색이면 밝은 테두리를 써서 어떤 배경 위에서도
 * 원이 보이게 한다.
 */
function updateBrushCursor() {
  const cursor = $("brush-cursor");
  const color = $("paint-color").value;
  const size = Number($("paint-size").value) * displayScale();

  cursor.style.width = `${size}px`;
  cursor.style.height = `${size}px`;
  cursor.style.background = color + "59";       // 살짝 비치게(35%)

  // 색의 밝기를 보고 테두리 색을 반대로 잡는다
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(color.substr(i, 2), 16));
  const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  const isLight = luminance > 0.55;
  cursor.style.borderColor = isLight ? "#111" : "#fff";
  // 반대쪽 색 링을 하나 더 둘러서 비슷한 밝기의 배경에서도 보이게
  cursor.style.boxShadow = isLight
    ? "0 0 0 1px rgba(255,255,255,0.9)"
    : "0 0 0 1px rgba(0,0,0,0.75)";
}

function showBrushCursorAt(clientX, clientY) {
  const cursor = $("brush-cursor");
  const rect = $("stage").getBoundingClientRect();
  cursor.style.left = `${clientX - rect.left}px`;
  cursor.style.top = `${clientY - rect.top}px`;
  cursor.hidden = false;
}

/** 슬라이더를 움직일 때는 마우스가 이미지 밖이어도 가운데에 잠깐 보여준다. */
let brushPeekTimer = null;
function peekBrushCursor() {
  const stage = $("stage");
  if (stage.hidden || !stage.clientWidth) return;
  const cursor = $("brush-cursor");
  cursor.style.left = `${stage.clientWidth / 2}px`;
  cursor.style.top = `${stage.clientHeight / 2}px`;
  cursor.hidden = false;
  clearTimeout(brushPeekTimer);
  brushPeekTimer = setTimeout(() => { cursor.hidden = true; }, 1100);
}

$("paint-size").addEventListener("input", () => {
  $("paint-size-label").textContent = $("paint-size").value;
  updateBrushCursor();
  if (state.tool === "paint") peekBrushCursor();
});

$("paint-color").addEventListener("input", () => {
  updateBrushCursor();
  if (state.tool === "paint") peekBrushCursor();
});

document.querySelectorAll(".swatch").forEach((sw) => {
  sw.addEventListener("click", () => {
    $("paint-color").value = sw.dataset.color;
    updateBrushCursor();
    if (state.tool === "paint") peekBrushCursor();
  });
});

/** 스포이드: 이미지에서 색을 집어 브러시 색으로 쓴다(주변 배경색 맞추기용). */
let pickingColor = false;
$("paint-pick").addEventListener("click", () => {
  pickingColor = !pickingColor;
  $("stage").classList.toggle("picking", pickingColor);
  $("paint-pick").classList.toggle("active", pickingColor);
});

function imagePointFromEvent(e) {
  const rect = $("stage").getBoundingClientRect();
  const scale = displayScale();
  return {
    x: (e.clientX - rect.left) / scale,
    y: (e.clientY - rect.top) / scale,
  };
}

function pickColorAt(pt) {
  // 배경 이미지 + 덧칠 레이어를 합쳐서 실제 보이는 색을 집는다
  const tmp = document.createElement("canvas");
  tmp.width = 1; tmp.height = 1;
  const tctx = tmp.getContext("2d");
  tctx.drawImage($("stage-img"), pt.x, pt.y, 1, 1, 0, 0, 1, 1);
  tctx.drawImage($("paint-canvas"), pt.x, pt.y, 1, 1, 0, 0, 1, 1);
  const [r, g, b] = tctx.getImageData(0, 0, 1, 1).data;
  return "#" + [r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("");
}

(function setupPainting() {
  const stage = $("stage");
  let painting = false;

  const pushHistory = () => {
    const canvas = $("paint-canvas");
    state.paintHistory.push(canvas.getContext("2d")
      .getImageData(0, 0, canvas.width, canvas.height));
    if (state.paintHistory.length > 20) state.paintHistory.shift();
  };

  stage.addEventListener("mousedown", (e) => {
    if (pickingColor) {
      $("paint-color").value = pickColorAt(imagePointFromEvent(e));
      pickingColor = false;
      stage.classList.remove("picking");
      $("paint-pick").classList.remove("active");
      return;
    }
    if (state.tool !== "paint") return;
    e.preventDefault();
    pushHistory();
    painting = true;

    const ctx = $("paint-canvas").getContext("2d");
    const pt = imagePointFromEvent(e);
    ctx.strokeStyle = $("paint-color").value;
    ctx.fillStyle = $("paint-color").value;
    ctx.lineWidth = Number($("paint-size").value);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(pt.x, pt.y);
    // 점 하나만 찍는 경우도 보이도록
    ctx.arc(pt.x, pt.y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(pt.x, pt.y);
  });

  stage.addEventListener("mousemove", (e) => {
    // 덧칠 모드에서는 브러시 크기만 한 원을 커서처럼 따라다니게 한다
    if (state.tool === "paint" && !pickingColor) showBrushCursorAt(e.clientX, e.clientY);

    if (!painting) return;
    const ctx = $("paint-canvas").getContext("2d");
    const pt = imagePointFromEvent(e);
    ctx.lineTo(pt.x, pt.y);
    ctx.stroke();
  });

  stage.addEventListener("mouseleave", () => { $("brush-cursor").hidden = true; });

  ["mouseup", "mouseleave"].forEach((ev) => {
    stage.addEventListener(ev, () => { painting = false; });
  });
})();

$("paint-undo").addEventListener("click", () => {
  const snapshot = state.paintHistory.pop();
  if (!snapshot) return;
  $("paint-canvas").getContext("2d").putImageData(snapshot, 0, 0);
});

$("paint-clear").addEventListener("click", () => {
  const canvas = $("paint-canvas");
  const ctx = canvas.getContext("2d");
  state.paintHistory.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});

/** 덧칠 레이어가 비어 있지 않으면 PNG data URL로 넘긴다. */
function paintLayerDataUrl() {
  const canvas = $("paint-canvas");
  if (!canvas.width || !canvas.height) return null;
  const data = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] !== 0) return canvas.toDataURL("image/png");
  }
  return null;   // 아무것도 안 칠했으면 보내지 않는다
}

// ------------------------------------------------------------ 재렌더링 / 결과

$("rerender-btn").addEventListener("click", async () => {
  if (!state.outKey) return;
  $("rerender-btn").disabled = true;
  setStatus($("run-status"), "편집 내용 반영해서 다시 렌더링 중...", "busy");

  try {
    const data = await postJSON("/api/rerender", {
      out_key: state.outKey,
      base: state.base,
      paint: paintLayerDataUrl(),
      paragraphs: state.paragraphs.map((p) => ({
        id: p.id, text: p.text, box: p.box, font_size: p.font_size,
      })),
    });
    state.downloadName = data.rendered_name;
    showResult(data.rendered_url, `${state.outKey}/${data.rendered_name}`);
    showLog(data.log);
    setStatus($("run-status"), "다시 렌더링 완료", "ok");
    switchView("result");
  } catch (err) {
    setStatus($("run-status"), err.message, "err");
  } finally {
    $("rerender-btn").disabled = false;
  }
});

function showResult(url, downloadPath) {
  $("result-empty").hidden = true;
  const img = $("result-img");
  img.hidden = false;
  img.src = url;

  const btn = $("download-btn");
  btn.href = `/api/download/${downloadPath}`;
  btn.setAttribute("download", state.downloadName || "rendered.png");
  btn.classList.remove("disabled");
}

// ------------------------------------------------------------ 탭 전환

function switchView(view) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  $("editor-view").hidden = view !== "editor";
  $("result-view").hidden = view !== "result";
  if (view === "editor") layoutBoxes();
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchView(tab.dataset.view));
});

// ------------------------------------------------------------ 초기화

document.querySelectorAll('input[name="model"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    $("fallback-notice").hidden = true;
    showStagePreview();
  });
});

(async function init() {
  await loadRenderConfig();
  await loadFonts();
  loadApiKeyState();
  showStagePreview();
  setTool("select");
})();
