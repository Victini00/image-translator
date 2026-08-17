import sys
import os
import re
import json
import time
import uuid
import shutil
import threading
import subprocess
from flask import Flask, request, jsonify, render_template, send_from_directory

# Windows 콘솔 기본 인코딩(cp949)이 일본어/한국어 혼용 출력 시 깨지는 것 방지
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

"""
<사용법>

이미지 번역 파이프라인 웹 UI.

    python app.py            (기본 http://127.0.0.1:5000)
    python app.py --port 8080

기존 파이프라인 스크립트(image_translator_v1.py / image_translator_v1_using_gemini.py)를
그대로 subprocess로 호출하고, 그 산출물(cleaned 이미지 + 번역 포함 JSON)을 브라우저에
넘겨서 사용자가 번역문과 위치를 직접 고칠 수 있게 한다. 고친 내용은 JSON에
반영해서 Rendering 단계만 다시 돌려(재번역 없음) 최종 이미지를 만든다.

동작 흐름:
  1. 이미지 업로드            -> POST /api/upload
  2. 모델 선택 후 실행         -> POST /api/run       (백그라운드 실행, job_id 반환)
     진행 상황 폴링            -> GET  /api/progress/<job_id>
  3. 브라우저에서 편집          -> 클라이언트 측에서만 진행
  4. 편집 결과로 다시 렌더링    -> POST /api/rerender  (Rendering 단계만 재실행)
  5. 결과 저장                 -> GET  /api/download/<파일명>

편집 화면 미리보기는 최종 렌더링과 똑같이 보이도록, 실제 렌더링에 쓰는 폰트 파일을
/api/font/<종류>로 서빙하고(브라우저가 @font-face로 로드), 렌더링 파라미터
(최대/최소 폰트 크기, 줄간격, 테두리 두께)도 /api/render-config로 넘겨준다.
줄바꿈/폰트 크기 결정 로직은 inpainting_rendering.fit_text()와 같은 규칙으로
클라이언트(app.js)에서 다시 계산한다.

Gemini API 키는 UI에서 입력하면 프로젝트 루트 .env에 GEMINI_API_KEY로 저장되고,
다음 실행부터는 자동으로 불러와진다. .env는 .gitignore에 걸려 있어 git에 올라가지 않는다.
"""

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(WEB_DIR)

# 경로·기본값·산출물 파일명 규칙은 src/config.py가 단일 출처다.
sys.path.insert(0, SRC_DIR)
import config  # noqa: E402

PROJECT_ROOT = config.PROJECT_ROOT
MODELS_DIR = config.CODE_MODELS_DIR

V1_SCRIPT = os.path.join(SRC_DIR, "image_translator_v1.py")
V1_GEMINI_SCRIPT = os.path.join(SRC_DIR, "image_translator_v1_using_gemini.py")
RENDERING_SCRIPT = os.path.join(MODELS_DIR, "inpainting_rendering.py")

ENV_PATH = config.ENV_PATH
WORK_DIR = config.WEB_WORK_DIR
UPLOAD_DIR = config.WEB_UPLOAD_DIR

# 폰트 경로와 박스 배치 규칙은 실제 렌더링과 반드시 같아야 하므로 render_layout에서
# 그대로 가져온다. render_layout은 표준 라이브러리만 쓰는 모듈이라
# (inpainting_rendering과 달리) torch/transformers를 끌고 오지 않는다.
sys.path.insert(0, MODELS_DIR)
import render_layout  # noqa: E402

RENDER_CONFIG = {
    "max_font_size": render_layout.DEFAULT_MAX_FONT_SIZE,
    "min_font_size": render_layout.DEFAULT_MIN_FONT_SIZE,
    "stroke_width": render_layout.DEFAULT_STROKE_WIDTH,
    "line_spacing": render_layout.DEFAULT_LINE_SPACING,
    "bubble_inset_ratio": render_layout.DEFAULT_BUBBLE_INSET_RATIO,
}

FONTS = {
    "interior": render_layout.DEFAULT_INTERIOR_FONT,
    "exterior": render_layout.DEFAULT_EXTERIOR_FONT,
}

# 각 단계 이름은 오케스트레이터가 찍는 "[2/4 Cleaning] 실행" 마커의 순서와 1:1로
# 대응된다. 화면에는 그 스크립트 내부 명칭(Masking/Rendering) 대신 실제로 무슨 일을
# 하는지가 드러나는 이름을 보여준다.
MODELS = {
    "v1": {
        "label": "v1 (로컬 전용: PaddleOCR + hell0ks)",
        "script": V1_SCRIPT,
        "needs_api_key": False,
        "stages": [
            "글자 찾기 + 읽기 (PaddleOCR)",
            "원본 글자 지우기 (LaMa)",
            "번역 (hell0ks) + 글자 그리기",
        ],
    },
    "v1_gemini": {
        "label": "v1 + Gemini (인식/번역을 Gemini API로)",
        "script": V1_GEMINI_SCRIPT,
        "needs_api_key": True,
        "stages": [
            "글자 위치 찾기 (PaddleOCR)",
            "글자 읽기 + 번역 (Gemini)",
            "원본 글자 지우기 (LaMa)",
            "번역문 그리기",
        ],
    },
}

# 백그라운드 작업 상태 저장소 (단일 사용자용 로컬 도구라 메모리에만 둔다)
JOBS = {}
JOBS_LOCK = threading.Lock()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 업로드 이미지 32MB 제한


# ---------------------------------------------------------------- .env 관리

def read_api_key():
    """.env에서 GEMINI_API_KEY 값을 읽는다. 없으면 None."""
    if not os.path.exists(ENV_PATH):
        return None
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                value = line.split("=", 1)[1].strip()
                return value or None
    return None


def write_api_key(key):
    """.env의 GEMINI_API_KEY만 갱신한다(다른 줄은 그대로 보존).
    파일이 없으면 새로 만든다."""
    lines = []
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith("GEMINI_API_KEY="):
            lines[i] = f"GEMINI_API_KEY={key}"
            replaced = True
            break
    if not replaced:
        lines.append(f"GEMINI_API_KEY={key}")

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def mask_key(key):
    """UI에 보여줄 용도로 키를 가린다 (앞 4자 + **** + 뒤 4자)."""
    if not key:
        return None
    if len(key) <= 12:
        return key[:2] + "*" * 6
    return f"{key[:4]}{'*' * 8}{key[-4:]}"


# ------------------------------------------------------- 파이프라인 실행 도우미

UPLOAD_ALLOWED_EXTS = config.UPLOAD_ALLOWED_EXTS


def safe_base_name(filename):
    """업로드 파일명에서 경로 조작/특수문자를 제거하고 확장자 없는 이름만 남긴다."""
    name = os.path.basename(filename or "")
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"[^\w\-.]", "_", stem, flags=re.UNICODE).strip("._") or "image"
    ext = ext.lower() if ext.lower() in UPLOAD_ALLOWED_EXTS else ".png"
    return stem, ext


def child_env():
    """자식 프로세스용 환경변수.
    - PYTHONNOUSERSITE: 유저 site-packages가 섞이면 numpy/numba 충돌로 깨짐
    - PYTHONUNBUFFERED: 버퍼링되면 단계 진행 표시가 실시간으로 안 올라옴"""
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_script(script_path, args, cwd):
    """스크립트를 subprocess로 돌리고 (성공여부, 출력로그)를 반환한다."""
    result = subprocess.run(
        [sys.executable, script_path] + args, cwd=cwd,
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=child_env(),
    )
    return result.returncode == 0, (result.stdout or "") + (result.stderr or "")


# 오케스트레이터가 각 단계 시작 때 찍는 "[2/4 Cleaning] 실행" 형태를 찾아낸다
STAGE_RE = re.compile(r"\[(\d+)\s*/\s*(\d+)\s+(.+?)\]\s*실행")


def run_script_streaming(script_path, args, cwd, on_stage):
    """스크립트를 돌리면서 출력을 한 줄씩 읽어, 단계 표시가 나올 때마다
    on_stage(현재단계번호, 전체단계수, 단계이름)을 호출한다."""
    proc = subprocess.Popen(
        [sys.executable, script_path] + args, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, env=child_env(),
    )
    lines = []
    for line in proc.stdout:
        lines.append(line)
        match = STAGE_RE.search(line)
        if match:
            on_stage(int(match.group(1)), int(match.group(2)), match.group(3).strip())
    proc.wait()
    return proc.returncode == 0, "".join(lines)


FALLBACK_SUMMARY_RE = re.compile(r"\[요약\] Gemini 성공 (\d+)개 / 로컬 폴백 (\d+)개 \(전체 (\d+)개\)")


def detect_fallback(log, model_key):
    """실행 로그를 보고 Gemini 대신 로컬 모델로 넘어간 정황을 찾아낸다.

    반환: None(해당 없음) 또는 {"kind", "message"} - UI에 그대로 띄운다."""
    if model_key != "v1_gemini":
        return None

    match = FALLBACK_SUMMARY_RE.search(log or "")
    if not match:
        # 요약 줄 자체가 없다 = Gemini 단계가 비정상 종료했을 가능성
        return None

    ok_count, fallback_count, total = (int(g) for g in match.groups())
    if fallback_count == 0:
        return None

    if fallback_count >= total:
        kind = "all"
        head = "Gemini 호출이 실패해서 전부 로컬 모델(PaddleOCR + hell0ks)로 처리했습니다."
    else:
        kind = "partial"
        head = (f"문단 {total}개 중 {fallback_count}개가 Gemini 처리에 실패해서 "
                f"로컬 모델(hell0ks)로 번역했습니다.")

    # 원인에 따라 사용자가 할 수 있는 조치가 달라서 구분해 안내한다
    if "API 오류(429)" in log or "RESOURCE_EXHAUSTED" in log or "quota" in log.lower():
        reason = (" API 요청 한도를 초과했습니다 — 잠시 후 다시 시도하거나 "
                  "ai.dev/rate-limit 에서 남은 한도를 확인해 보세요.")
    elif any(f"API 오류({code})" in log for code in (500, 502, 503, 504)):
        reason = " Gemini 서버가 일시적으로 혼잡한 상태였습니다 — 잠시 후 다시 시도하면 대개 성공합니다."
    elif "API 키" in log or "API_KEY" in log or "401" in log or "403" in log:
        reason = " API 키가 잘못되었거나 권한이 없습니다 — 키를 다시 확인해 주세요."
    else:
        reason = ""

    return {"kind": kind, "message": head + reason, "ok": ok_count,
            "fallback": fallback_count, "total": total}


def effective_render_box(para, inset_ratio=0.15):
    """실제 렌더링이 쓸 박스를 계산한다(편집 UI 초기 위치용).
    inpainting_rendering.get_box_for_paragraph()를 그대로 호출해서
    규칙이 어긋날 여지를 없앤다."""
    (x1, y1, x2, y2), _ = render_layout.get_box_for_paragraph(para, inset_ratio)
    return [int(x1), int(y1), int(x2), int(y2)]


def build_editor_payload(translated_json_path):
    """렌더링 결과 JSON을 편집 UI가 쓰기 좋은 형태로 정리한다."""
    with open(translated_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for para in data["paragraphs"]:
        text = (para.get("translated_text") or "").strip()
        if not text:
            continue
        items.append({
            "id": para["id"],
            "text": text,
            "source_text": para.get("gemini_text") or para.get("merged_text") or "",
            "box": effective_render_box(para, RENDER_CONFIG["bubble_inset_ratio"]),
            "is_interior": bool(para.get("bubble_bbox")),
        })
    return data, items


# ------------------------------------------------------------- 백그라운드 작업

def set_job(job_id, **fields):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(fields)


def pipeline_worker(job_id, model_key, img_path, out_dir, base):
    """파이프라인을 백그라운드에서 돌리면서 진행 상황을 JOBS에 기록한다."""
    model = MODELS[model_key]

    def on_stage(index, total, name):
        set_job(job_id, stage_index=index, stage_total=total, stage_name=name)

    try:
        ok, log = run_script_streaming(
            model["script"], ["--img", img_path, "--out-dir", out_dir], SRC_DIR, on_stage,
        )
        if not ok:
            set_job(job_id, status="error", error="파이프라인 실행 실패", log=log[-4000:])
            return

        translated_json = os.path.join(out_dir, config.translated_json(base))
        cleaned_png = os.path.join(out_dir, config.cleaned_image(base))
        rendered_png = os.path.join(out_dir, config.rendered_image(base))
        for path in (translated_json, cleaned_png, rendered_png):
            if not os.path.exists(path):
                set_job(job_id, status="error",
                        error=f"결과 파일이 생성되지 않았습니다: {os.path.basename(path)}",
                        log=log[-4000:])
                return

        data, items = build_editor_payload(translated_json)
        rel = os.path.basename(out_dir)
        set_job(job_id, status="done", log=log[-4000:],
                fallback=detect_fallback(log, model_key), result={
            "base": base,
            "out_key": rel,
            "image_width": data["image_width"],
            "image_height": data["image_height"],
            "cleaned_url": f"/api/file/{rel}/{config.cleaned_image(base)}",
            "rendered_url": f"/api/file/{rel}/{config.rendered_image(base)}",
            "rendered_name": config.rendered_image(base),
            "paragraphs": items,
        })
    except Exception as e:  # 예상 못 한 오류도 UI에 그대로 보여준다
        set_job(job_id, status="error", error=f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ 라우트

@app.route("/")
def index():
    return render_template("index.html", models=MODELS)


@app.route("/api/models")
def get_models():
    """모델별 라벨과 실제 처리 단계 목록 (실행 전 안내용)."""
    return jsonify({"models": {
        key: {"label": m["label"], "stages": m["stages"], "needs_api_key": m["needs_api_key"]}
        for key, m in MODELS.items()
    }})


@app.route("/api/render-config")
def get_render_config():
    """편집 미리보기가 최종 렌더링과 같은 규칙으로 그리도록 파라미터를 넘긴다."""
    return jsonify(RENDER_CONFIG)


_font_cache = {}


def sanitize_font(path):
    """폰트를 브라우저가 받아들이는 형태로 정리해서 바이트로 돌려준다.

    HY피오피M.TTF처럼 cmap 테이블이 실제 글리프 수를 넘는 인덱스를 가리키는
    폰트가 있다. PIL은 그냥 읽지만 브라우저는 OTS 검증에서 거부한다
    ("Range glyph reference too high"). 실제로 존재하는 글리프를 가리키는
    매핑만 남겨 cmap을 새로 만들어 준다."""
    from fontTools.ttLib import TTFont
    from fontTools.ttLib.tables._c_m_a_p import CmapSubtable
    import io

    font = TTFont(path, fontNumber=0)
    glyphs = set(font.getGlyphOrder())

    mapping = {}
    for table in font["cmap"].tables:
        for codepoint, name in table.cmap.items():
            if name in glyphs:
                mapping[codepoint] = name

    subtable = CmapSubtable.newSubtable(4)
    subtable.platformID, subtable.platEncID, subtable.language = 3, 1, 0
    subtable.cmap = {cp: n for cp, n in mapping.items() if cp <= 0xFFFF}
    font["cmap"].tables = [subtable]

    buf = io.BytesIO()
    font.save(buf)
    return buf.getvalue()


@app.route("/api/font/<kind>")
def get_font(kind):
    """실제 렌더링에 쓰는 폰트 파일을 서빙한다(@font-face용).

    일부 폰트(HY피오피M 등)는 cmap 테이블이 규격에 안 맞아서 PIL은 읽지만
    브라우저(OTS 검증)는 거부한다. 그런 경우 fontTools로 한 번 다시 저장해서
    정상 테이블로 만들어 넘긴다. 결과는 메모리에 캐시한다."""
    path = FONTS.get(kind)
    if not path or not os.path.exists(path):
        return jsonify({"error": f"폰트 파일을 찾을 수 없습니다: {path}"}), 404

    if kind not in _font_cache:
        try:
            _font_cache[kind] = sanitize_font(path)
        except Exception as e:
            print(f"[경고] {kind} 폰트 변환 실패, 원본 그대로 전송: {e}")
            with open(path, "rb") as f:
                _font_cache[kind] = f.read()

    from flask import Response
    return Response(_font_cache[kind], mimetype="font/ttf")


@app.route("/api/fit", methods=["POST"])
def fit_layout():
    """편집 미리보기용 - 실제 렌더링과 같은 fit_text()로 폰트 크기/줄바꿈을 계산해 준다.

    브라우저에서 같은 알고리즘을 다시 구현해도 글자 폭 측정이 PIL(FreeType 힌팅)과
    미묘하게 달라 폰트 크기가 1px씩 어긋난다. 그래서 최종 표시는 서버가 계산한
    값을 그대로 쓴다(정의상 결과와 100% 일치)."""
    items = (request.json or {}).get("items") or []
    results = []
    for item in items:
        box = item.get("box") or [0, 0, 1, 1]
        w = max(1, int(round(box[2] - box[0])))
        h = max(1, int(round(box[3] - box[1])))
        try:
            fit = render_layout.layout_text(
                (item.get("text") or "").strip(), w, h, bool(item.get("is_interior")),
                RENDER_CONFIG["max_font_size"], RENDER_CONFIG["min_font_size"],
                RENDER_CONFIG["stroke_width"], RENDER_CONFIG["line_spacing"],
                force_size=item.get("font_size"),
            )
        except Exception as e:
            print(f"[경고] 배치 계산 실패(id={item.get('id')}): {e}")
            continue
        results.append({"id": item.get("id"), **fit})
    return jsonify({"items": results})


@app.route("/api/apikey", methods=["GET"])
def get_apikey():
    key = read_api_key()
    return jsonify({"has_key": key is not None, "masked": mask_key(key)})


@app.route("/api/apikey", methods=["POST"])
def set_apikey():
    key = (request.json or {}).get("key", "").strip()
    if not key:
        return jsonify({"error": "키가 비어 있습니다."}), 400
    write_api_key(key)
    return jsonify({"ok": True, "masked": mask_key(key)})


@app.route("/api/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify({"error": "이미지 파일이 없습니다."}), 400

    stem, ext = safe_base_name(file.filename)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_name = f"{stem}{ext}"
    file.save(os.path.join(UPLOAD_DIR, saved_name))

    return jsonify({"ok": True, "name": saved_name, "base": stem,
                    "url": f"/api/file/uploads/{saved_name}"})


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    """파이프라인을 백그라운드로 시작하고 job_id를 즉시 돌려준다."""
    body = request.json or {}
    name = body.get("name")
    model_key = body.get("model", "v1")

    if model_key not in MODELS:
        return jsonify({"error": f"알 수 없는 모델: {model_key}"}), 400
    model = MODELS[model_key]

    img_path = os.path.join(UPLOAD_DIR, os.path.basename(name or ""))
    if not os.path.exists(img_path):
        return jsonify({"error": "업로드된 이미지를 찾을 수 없습니다."}), 400

    if model["needs_api_key"] and not read_api_key():
        return jsonify({"error": "Gemini API 키가 설정되지 않았습니다. 먼저 키를 저장해 주세요."}), 400

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_dir = os.path.join(WORK_DIR, f"{base}__{model_key}")
    # 같은 이미지+모델로 다시 돌리면 이전 산출물이 섞이지 않게 비우고 시작
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    job_id = uuid.uuid4().hex
    set_job(job_id, status="running", stage_index=0, stage_total=len(model["stages"]),
            stage_name="준비 중", stages=model["stages"], started_at=time.time())

    threading.Thread(target=pipeline_worker,
                     args=(job_id, model_key, img_path, out_dir, base),
                     daemon=True).start()

    return jsonify({"ok": True, "job_id": job_id, "stages": model["stages"]})


@app.route("/api/progress/<job_id>")
def progress(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        job = dict(job) if job else None
    if job is None:
        return jsonify({"error": "알 수 없는 작업입니다."}), 404
    job["elapsed"] = round(time.time() - job.get("started_at", time.time()), 1)
    return jsonify(job)


def composite_paint_layer(cleaned_png, paint_data_url, out_dir, base):
    """브라우저에서 브러시로 칠한 투명 레이어(PNG data URL)를 지운 이미지 위에 합성한다.

    detection이 놓쳐서 원본 글자가 남은 부분을 사용자가 직접 색으로 덮을 수 있게
    하는 기능. 합성 결과를 새 파일로 저장하고 그 경로를 돌려준다."""
    import base64
    from io import BytesIO
    from PIL import Image

    header, _, encoded = paint_data_url.partition(",")
    if "base64" not in header:
        raise ValueError("지원하지 않는 이미지 형식입니다.")

    overlay = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGBA")
    background = Image.open(cleaned_png).convert("RGBA")
    if overlay.size != background.size:
        overlay = overlay.resize(background.size, Image.LANCZOS)

    background.alpha_composite(overlay)
    painted_path = os.path.join(out_dir, config.painted_image(base))
    background.convert("RGB").save(painted_path)
    return painted_path


@app.route("/api/rerender", methods=["POST"])
def rerender():
    """편집된 텍스트/위치를 JSON에 반영하고 Rendering 단계만 다시 실행한다.
    번역은 다시 하지 않는다(--use-existing-translation)."""
    body = request.json or {}
    out_key = os.path.basename(body.get("out_key") or "")
    base = os.path.basename(body.get("base") or "")
    edits = body.get("paragraphs") or []

    out_dir = os.path.join(WORK_DIR, out_key)
    translated_json = os.path.join(out_dir, config.translated_json(base))
    cleaned_png = os.path.join(out_dir, config.cleaned_image(base))
    if not (os.path.exists(translated_json) and os.path.exists(cleaned_png)):
        return jsonify({"error": "이전 실행 결과를 찾을 수 없습니다. 먼저 실행해 주세요."}), 400

    with open(translated_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_id = {p["id"]: p for p in data["paragraphs"]}
    for edit in edits:
        para = by_id.get(edit.get("id"))
        if para is None:
            continue
        # 사용자가 넣은 줄바꿈은 그대로 살린다(양 끝 공백만 정리)
        para["translated_text"] = (edit.get("text") or "").strip()
        box = edit.get("box")
        if box and len(box) == 4:
            para["render_bbox"] = [int(round(v)) for v in box]
        size = edit.get("font_size")
        if size:
            para["font_size"] = int(size)
        else:
            para.pop("font_size", None)  # 자동 크기로 되돌림

    edited_json = os.path.join(out_dir, config.edited_json(base))
    with open(edited_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 사용자가 브러시로 덧칠한 레이어가 있으면 지운 이미지 위에 먼저 합성한다
    base_image = cleaned_png
    paint = body.get("paint")
    if paint:
        try:
            base_image = composite_paint_layer(cleaned_png, paint, out_dir, base)
        except Exception as e:
            print(f"[경고] 덧칠 레이어 합성 실패, 원본으로 진행: {e}")

    rendered_png = os.path.join(out_dir, config.rendered_edited_image(base))
    ok, log = run_script(RENDERING_SCRIPT, [
        "--json", edited_json,
        "--cleaned-image", base_image,
        "--out", rendered_png,
        "--use-existing-translation",
    ], cwd=MODELS_DIR)

    if not ok or not os.path.exists(rendered_png):
        return jsonify({"error": "렌더링 실패", "log": log[-4000:]}), 500

    # 브라우저 캐시 때문에 이전 이미지가 그대로 보이는 것을 막으려고 쿼리스트링을 붙임
    return jsonify({
        "ok": True,
        "rendered_url": f"/api/file/{out_key}/{config.rendered_edited_image(base)}?t={int(time.time())}",
        "rendered_name": config.rendered_edited_image(base),
        "log": log[-4000:],
    })


def _safe_work_path(subpath):
    full = os.path.normpath(os.path.join(WORK_DIR, subpath))
    if not full.startswith(os.path.normpath(WORK_DIR)) or not os.path.exists(full):
        return None
    return full


@app.route("/api/file/<path:subpath>")
def serve_file(subpath):
    """작업 폴더(output/web) 안의 파일만 서빙한다."""
    full = _safe_work_path(subpath)
    if full is None:
        return jsonify({"error": "파일을 찾을 수 없습니다."}), 404
    return send_from_directory(os.path.dirname(full), os.path.basename(full))


@app.route("/api/download/<path:subpath>")
def download_file(subpath):
    full = _safe_work_path(subpath)
    if full is None:
        return jsonify({"error": "파일을 찾을 수 없습니다."}), 404
    return send_from_directory(os.path.dirname(full), os.path.basename(full), as_attachment=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="이미지 번역 파이프라인 웹 UI")
    parser.add_argument("--host", type=str, default=config.WEB_HOST)
    parser.add_argument("--port", type=int, default=config.WEB_PORT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    for kind, path in FONTS.items():
        if not os.path.exists(path):
            print(f"[경고] {kind} 폰트를 찾을 수 없습니다: {path}")
    print(f"웹 UI 시작: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
