"""
프로젝트 전역 설정
"""

import os

# ------------------------------------------------------------------ 디렉터리

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

CODE_MODELS_DIR = os.path.join(SRC_DIR, "models")          # 모델 관련 .py 파일
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")          # 모델 가중치 파일
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

RAW_IMAGES_DIR = os.path.join(DATA_DIR, "raw", "images")

PIPELINE_V1_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pipeline_v1")
PIPELINE_V1_GEMINI_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pipeline_v1_gemini")
OCR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "ocr", "v4")
CLEANED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "inpainting", "cleaned")
RENDERED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "translation", "rendering")
TRANSLATION_TEXT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "translation", "text", "hell0ks")

# 단계별 스크립트를 단독 실행할 때 쓰는 샘플 이미지 이름. 실제 작업은 --img /
SAMPLE_IMAGE_NAME = "Fillme"

def native_path(path):
    """
    C++ 추론 엔진(Paddle / torch.jit)에 파일 경로를 넘길 때 쓴다.
    """
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


# ------------------------------------------------------------------ OCR (Detection / Recognition)

# detection: mobile과 실측 비교 후 server 채택(검출 누락이 더 적음).
# 단, 아주 작은 반복 의성어는 mobile이 더 잘 잡는 경우도 있었다.
DETECTION_MODEL = "PP-OCRv5_server_det"

# recognition: 일본어 만화 텍스트로 직접 파인튜닝한 v4를 사용
# PaddleOCR에는 원본 모델 이름을 알려주고, 가중치는 아래 디렉터리에서 읽는다.
RECOGNITION_MODEL_NAME = "PP-OCRv5_mobile_rec"
RECOGNITION_MODEL_DIR = os.path.join(
    MODELS_DIR, "ocr", "PP-OCRv5_mobile_rec_jp_fine_tuned_v4", "complete_model"
)

OCR_LANG = "japan"

# 인식 신뢰도 임계값. 
REC_SCORE_THRESH = 0.4

# mask_bbox에 추가할 여유 픽셀
MASK_PAD = 10

# PaddleOCR(PaddleX)이 받아주는 확장자.
PADDLE_SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# ------------------------------------------------------------------ 말풍선 검출

BUBBLE_DETECTOR_REPO = "ogkalu/comic-speech-bubble-detector-yolov8m"
BUBBLE_DETECTOR_FILE = "comic-speech-bubble-detector.pt"
BUBBLE_CONF_THRESH = 0.5

# ------------------------------------------------------------------ 인페인팅 

LAMA_DIR = os.path.join(MODELS_DIR, "inpainting", "LaMa")
LAMA_FILENAME = "big-lama.pt"

LAMA_DOWNLOAD_URL = (
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
)

# 말풍선 밖 텍스트(LaMa) / 안 텍스트(단색 채우기)에 각각 다른 팽창 픽셀을 쓴다.
LAMA_CONTEXT_PAD = 30
LAMA_DILATE_PX = 4
FILL_DILATE_PX = 1

# ------------------------------------------------------------------ 번역

TRANSLATION_MODEL = "hell0ks/ja-ko-vn-7b-v1"
TRANSLATION_LORA_DIR = os.path.join(
    MODELS_DIR, "translation", "hell0ks_ja-ko-vn-7b-v1", "lora", "v1"
)
# LoRA를 다시 학습할 때 쓰는 학습 데이터(translation_hell0ks_lora_train.py).
TRANSLATION_LORA_TRAIN_DATA = os.path.join(
    DATA_DIR, "raw", "texts", "joujiboi", "japanese-anime-speech-v2", "translated",
    "audio_transcription_list_processed_lora_train_set_v1.jsonl",
)

# 번역 전 OCR 오류 교정 단계에서 시도했던 모델. 중국어 간체 혼입 문제로 폐기했고
# 파이프라인에는 들어가지 않는다
# (src/models/experiments/translation_correction.py 참고).
CORRECTION_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Gemini: 특정 버전이 아니라 "최신 안정 Flash"를 가리키는 별칭.
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_CROP_PAD = 20

# ------------------------------------------------------------------ 폰트

def _resolve_font(candidates, env_var):
    """
    폰트 경로를 결정
    """
    override = os.environ.get(env_var)
    if override:
        return override
    for path in candidates:
        expanded = os.path.expandvars(os.path.expanduser(path))
        if os.path.exists(expanded):
            return expanded
    return os.path.expandvars(os.path.expanduser(candidates[0]))


# 말풍선 안: 본문 가독성 기준으로 4종 비교 후 채택
INTERIOR_FONT_CANDIDATES = [
    r"%LOCALAPPDATA%\Microsoft\Windows\Fonts\GmarketSansMedium.otf",
    r"C:\Windows\Fonts\GmarketSansMedium.otf",
]
# 말풍선 밖(효과음): 그림 위에 얹혀도 읽히도록 흰 테두리와 함께 사용
EXTERIOR_FONT_CANDIDATES = [
    r"%LOCALAPPDATA%\Microsoft\Windows\Fonts\HY피오피M.TTF",
    r"C:\Windows\Fonts\HY피오피M.TTF",
]

# 다른 폰트를 쓰려면 환경변수로 지정하거나, 각 스크립트의 --interior-font /
# --exterior-font 옵션을 쓰면 된다.
INTERIOR_FONT = _resolve_font(INTERIOR_FONT_CANDIDATES, "IT_INTERIOR_FONT")
EXTERIOR_FONT = _resolve_font(EXTERIOR_FONT_CANDIDATES, "IT_EXTERIOR_FONT")

# ------------------------------------------------------------------ 글자 배치

MAX_FONT_SIZE = 36
MIN_FONT_SIZE = 12
STROKE_WIDTH = 2          # 말풍선 밖 텍스트 흰 테두리 두께
LINE_SPACING = 1.25
BUBBLE_INSET_RATIO = 0.15  # bubble_bbox를 안쪽으로 줄이는 비율

# 줄을 하나 더 쓰는 것을 허용하는 기준. 줄 수가 늘어난 대신 글자 크기가 이 비율
# 이상 커질 때만 그 조합을 택한다. 낮추면 글자가 커지는 대신 줄이 잘게 나뉜다.
LINE_INCREASE_GAIN = 0.15

# ------------------------------------------------------------------ 웹 UI

WEB_WORK_DIR = os.path.join(OUTPUT_DIR, "web")      # 작업물(업로드/중간/결과)
WEB_UPLOAD_DIR = os.path.join(WEB_WORK_DIR, "uploads")
WEB_HOST = "127.0.0.1"
WEB_PORT = 5000

# 업로드를 받아주는 확장자. PaddleOCR이 직접 못 읽는 형식(webp 등)은 파이프라인
# 앞단에서 PNG로 자동 변환한다(PADDLE_SUPPORTED_EXTS 참고).
UPLOAD_ALLOWED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif")

# ------------------------------------------------------------------ 산출물 파일명

# 파이프라인 각 단계가 주고받는 파일 이름 규칙. 오케스트레이터와 웹 UI가 따로
# 조립하고 있었어서, 한쪽만 바뀌면 다음 단계가 파일을 못 찾는 문제가 있었다.

def paragraphs_json(base):
    """Masking 산출물 - 문단별 좌표/원문"""
    return f"{base}_paragraphs.json"


def corrected_json(base):
    """교정 단계(experiments/translation_correction.py) 산출물 - 파이프라인에는 안 들어감"""
    return f"{base}_paragraphs_corrected.json"


def translated_json(base):
    """Rendering이 남기는 번역문 포함 JSON"""
    return f"{base}_paragraphs_translated.json"


def edited_json(base):
    """웹 UI 편집 내용을 반영한 JSON"""
    return f"{base}_paragraphs_edited.json"


def cleaned_image(base):
    """Cleaning 산출물 - 원본 글자를 지운 이미지"""
    return f"{base}_cleaned.png"


def painted_image(base):
    """웹 UI 덧칠 레이어를 합성한 이미지"""
    return f"{base}_cleaned_painted.png"


def rendered_image(base):
    """최종 결과"""
    return f"{base}_rendered.png"


def rendered_edited_image(base):
    """웹 UI 편집 후 다시 렌더링한 결과"""
    return f"{base}_rendered_edited.png"


def converted_image(base):
    """PaddleOCR이 못 읽는 형식을 PNG로 변환한 파일"""
    return f"{base}_converted.png"
