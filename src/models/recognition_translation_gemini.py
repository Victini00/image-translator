import sys
import os
import json
import re
import time
import argparse
from google import genai
from google.genai import types
from google.genai import errors
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv

# Windows 콘솔 기본 인코딩(cp949)이 일본어/한국어 혼용 출력 시 깨지는 것 방지
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# 프로젝트 루트(이 파일 기준 ./../../)의 .env에서 GEMINI_API_KEY를 읽어와
# 환경변수로 등록한다. find_dotenv()가 현재 작업 디렉터리부터 상위로 올라가며
# .env를 찾으므로, src/models에서 직접 실행하든 subprocess로 실행되든 동작함.
load_dotenv()

"""
<사용법>

Masking(paragraphs.json) 결과를 받아서, 이미지 전체의 문단들을 Google Gemini
API(비전)에 보내 "한 번에 다 읽고, 한 번에 다 번역"시킨다.
(Claude API 버전에서 계정 크레딧 문제로 Gemini 무료 티어로 전환했는데, 문단마다
따로 호출하니 API 요청 수가 너무 많이 나가서 - 문단 20개짜리 이미지 한 장에
20~40회 호출 - 무료 티어 일일 한도(RPD)를 몇 장 테스트하자마자 넘겨버림.
그래서 이미지 1장 = 요청 2회(인식 1회 + 번역 1회)로 줄임.)

기존 파이프라인은 PaddleOCR(detection+recognition)이 읽은 텍스트를
hell0ks+LoRA가 번역했는데, 이 스크립트는 detection(문단 위치, bbox/poly)은
그대로 PaddleOCR+YOLO 결과를 쓰되, recognition(글자 읽기)과 번역은 Gemini가
대신한다 - 지금까지 recognition에서 반복됐던 문제(세로 조판 특수문자
회전(ー/〜), 후리가나 혼입, 특이한 폰트체)를 완화하기 위한 하이브리드 방식.

호출 1회(인식): 문단별 crop 이미지를 전부 한 번의 contents에 담아서(각 이미지
앞에 "문단 id=N" 라벨을 붙여 순서/식별 보장) 한 번에 다 읽게 시킨다.
호출 2회(번역): 인식된 문단별 원문 텍스트 목록을 텍스트만으로(이미지 없이)
한 번에 번역시킨다.

주의: detection(bbox)의 정밀도는 PaddleOCR+YOLO 그대로라 정확하지만,
Gemini가 "읽은" 텍스트가 실제 문단 범위를 벗어나거나 여러 줄을 하나로
합칠 수 있음 - merged_text를 그대로 덮어쓰지 않고 별도 필드(gemini_text)에
저장해서 원본 OCR 결과와 비교 가능하게 해둠.

인식/번역 각각 독립적으로 실패할 수 있고(예: 인식은 성공, 번역만 실패), 그
경우에도 성공한 부분은 살아남는다 - 번역만 실패하면 gemini_text(더 정확한
원문)는 남아서, inpainting_rendering.py의 hell0ks 폴백 번역이 그 원문을 쓰게
됨. id 매칭이 안 된 문단(모델이 응답에서 빠뜨린 경우)만 개별적으로 폴백된다.

paragraphs.json을 읽어서 각 문단에 gemini_text(Gemini가 읽은 원문),
translated_text(한국어 번역)를 추가한 뒤 같은(또는 --out으로 지정한) 경로에
다시 저장한다. inpainting_rendering.py에서 --use-existing-translation
옵션을 주면 이 translated_text를 그대로 써서 hell0ks 번역을 건너뛴다.

사전 준비: 프로젝트 루트에 .env 파일을 만들고 GEMINI_API_KEY=... 한 줄을
넣어두면 자동으로 읽어온다 (키는 https://aistudio.google.com 에서 카드 등록 없이
발급 가능 - 무료 티어는 분당/일당 요청 수 제한만 있음). .env는 .gitignore에
걸려 있어서 git에 올라가지 않는다.

예시:
    python recognition_translation_gemini.py --json ./../../output/pipeline_v1_gemini/test2_paragraphs.json
"""

DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_CROP_PAD = 20
# 인식 호출과 번역 호출 사이 최소 대기(초) - 이제 이미지당 호출이 2회뿐이라
# RPM 걱정은 거의 없지만, 혹시 몰라 아주 짧게 여유를 둠.
DEFAULT_REQUEST_INTERVAL = 2.0

RECOGNITION_SYSTEM_PROMPT = """당신은 일본 만화(manga) 페이지에서 잘라낸(crop) 여러 개의
이미지 조각을 보고, 각 조각 안의 일본어 텍스트를 정확히 읽는 작업을 합니다.

각 이미지 조각 앞에는 "문단 id=N" 라벨이 붙어 있습니다 - 반드시 그 id에 맞춰
결과를 반환하세요 (라벨과 이미지 순서가 항상 1:1로 대응됩니다).

읽을 때 주의할 점:
- 세로 조판(vertical typesetting)인 경우 위에서 아래로, 오른쪽에서 왼쪽 순서로 읽습니다.
- 세로 조판에서는 「ー」(장음 기호)나 「〜」 같은 특수문자가 가로 조판일 때와 달리
  90도 회전되어(세로로 눕혀져) 보이는 경우가 많습니다 - 이를 알파벳(I, S 등)으로
  잘못 읽지 말고, 문맥상 자연스러운 장음/물결표로 인식하세요.
- 후리가나(한자 위/옆에 작게 달린 읽기 보조 가나)는 본문에 포함하지 마세요.
  본문(한자/가나) 자체만 읽으세요.
- 한 조각 안에 텍스트가 여러 줄에 걸쳐 있어도, 하나의 문단(발화)이므로
  자연스럽게 이어서 하나의 문장(들)으로 합쳐서 읽으세요.

반드시 지정된 JSON 스키마 형식으로만 응답하세요. 모든 id에 대해 결과를 빠짐없이 반환하세요."""

TRANSLATION_SYSTEM_PROMPT = """당신은 일본 만화(manga) 대사를 자연스러운 한국어로
번역하는 작업을 합니다. id별로 일본어 원문 목록이 주어지면, 각각을 한국어로
번역해서 같은 id로 반환하세요.

번역할 때 주의할 점:
- 만화 대사/효과음의 어조와 뉘앙스를 살려서 자연스러운 한국어로 번역하세요.
- 직역보다는 상황에 맞는 자연스러운 구어체를 우선하세요.
- 각 id는 독립된 문단(발화)이므로, id끼리 서로 이어붙이거나 순서를 바꾸지 마세요.

반드시 지정된 JSON 스키마 형식으로만 응답하세요. 모든 id에 대해 결과를 빠짐없이 반환하세요."""


class OCRItem(BaseModel):
    id: int
    japanese_text: str


class OCRBatchResult(BaseModel):
    items: list[OCRItem]


class TranslationItem(BaseModel):
    id: int
    korean_translation: str


class TranslationBatchResult(BaseModel):
    items: list[TranslationItem]


def crop_paragraph_image(image, bbox, pad):
    """문단 bbox 주변에 여유(pad)를 두고 원본 이미지에서 잘라낸다."""
    img_w, img_h = image.size
    x1, y1, x2, y2 = bbox
    cx1 = max(0, int(x1 - pad))
    cy1 = max(0, int(y1 - pad))
    cx2 = min(img_w, int(x2 + pad))
    cy2 = min(img_h, int(y2 + pad))
    return image.crop((cx1, cy1, cx2, cy2))


def _parse_retry_delay(message, default=15.0):
    """429 에러 메시지에 담긴 'Please retry in 42.1s' 형태에서 대기 시간을 뽑아낸다.
    못 찾으면 기본값(default)을 씀."""
    if message:
        m = re.search(r"retry in ([\d.]+)s", message)
        if m:
            return float(m.group(1)) + 1.0  # 약간 여유를 더 둠
    return default


def _generate_with_retry(client, model, contents, config, max_retries=1):
    """429(rate limit)는 에러 메시지의 권장 대기 시간만큼 쉬었다가 한 번 재시도한다.
    그 외 에러거나 재시도까지 실패하면 예외를 그대로 올린다(호출부에서 처리)."""
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as e:
            if e.code == 429 and attempt < max_retries:
                delay = _parse_retry_delay(e.message)
                print(f"  [경고] rate limit(429), {delay:.1f}초 대기 후 재시도...")
                time.sleep(delay)
                continue
            raise


def recognize_batch(client, model, id_crop_pairs, fallback_texts):
    """(id, crop 이미지) 목록을 한 번의 요청으로 다 읽는다.
    성공한 id만 담긴 {id: 원문} dict를 반환. 전체 실패 시 빈 dict."""
    contents = []
    for pid, crop in id_crop_pairs:
        hint = fallback_texts.get(pid, "")
        label = f"문단 id={pid}"
        if hint:
            label += f" (참고용 기존 OCR: \"{hint}\")"
        contents.append(f"{label}:")
        contents.append(crop.convert("RGB"))
    contents.append("위 각 이미지 조각의 일본어 텍스트를 읽어서 id와 매칭해 반환해 주세요.")

    try:
        response = _generate_with_retry(
            client, model, contents,
            types.GenerateContentConfig(
                system_instruction=RECOGNITION_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=OCRBatchResult,
            ),
        )
    except errors.APIError as e:
        print(f"  [경고] 인식 호출 API 오류({e.code}), 전체 폴백: {e.message}")
        return {}

    if response.parsed is None:
        print("  [경고] 인식 응답 파싱 실패, 전체 폴백")
        return {}

    return {item.id: item.japanese_text.strip() for item in response.parsed.items}


def translate_batch(client, model, id_to_text):
    """{id: 일본어 원문} dict를 한 번의 요청으로 다 번역한다.
    성공한 id만 담긴 {id: 번역문} dict를 반환. 전체 실패 시 빈 dict."""
    if not id_to_text:
        return {}

    lines = [f"id={pid}: {text}" for pid, text in id_to_text.items()]
    prompt = "다음 일본어 대사들을 각각 한국어로 번역해서 같은 id로 반환해 주세요:\n" + "\n".join(lines)

    try:
        response = _generate_with_retry(
            client, model, [prompt],
            types.GenerateContentConfig(
                system_instruction=TRANSLATION_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=TranslationBatchResult,
            ),
        )
    except errors.APIError as e:
        print(f"  [경고] 번역 호출 API 오류({e.code}), 번역만 전체 폴백: {e.message}")
        return {}

    if response.parsed is None:
        print("  [경고] 번역 응답 파싱 실패, 번역만 전체 폴백")
        return {}

    return {item.id: item.korean_translation.strip() for item in response.parsed.items}


def run(args):
    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_path = args.img or data.get("image_path")
    if not img_path or not os.path.exists(img_path):
        print(f"에러: 원본 이미지를 찾을 수 없음: {img_path}")
        sys.exit(1)

    image = Image.open(img_path).convert("RGB")
    client = genai.Client()

    paragraphs = [p for p in data["paragraphs"] if p["merged_text"].strip()]
    fallback_texts = {p["id"]: p["merged_text"].strip() for p in paragraphs}
    id_crop_pairs = [(p["id"], crop_paragraph_image(image, p["bbox"], args.crop_pad)) for p in paragraphs]

    print(f"{len(paragraphs)}개 문단 인식 요청 중 (모델: {args.model})...")
    id_to_ja = recognize_batch(client, args.model, id_crop_pairs, fallback_texts)
    print(f"  -> {len(id_to_ja)}/{len(paragraphs)}개 문단 인식 성공")

    if args.request_interval > 0 and id_to_ja:
        time.sleep(args.request_interval)

    id_to_ko = {}
    if id_to_ja:
        print(f"{len(id_to_ja)}개 문단 번역 요청 중...")
        id_to_ko = translate_batch(client, args.model, id_to_ja)
        print(f"  -> {len(id_to_ko)}/{len(id_to_ja)}개 문단 번역 성공")

    for para in paragraphs:
        pid = para["id"]
        fallback_text = fallback_texts[pid]
        ja_text = id_to_ja.get(pid)
        ko_text = id_to_ko.get(pid)

        if ja_text is None:
            # 인식 자체가 실패한(id가 응답에 없는) 문단 - 기존 OCR 결과로 완전히
            # 폴백됨(뒤 단계의 hell0ks 번역이 merged_text를 그대로 씀).
            print(f"  [문단 {pid}] Gemini 인식 실패 -> 기존 OCR 결과로 폴백: {fallback_text}")
            continue

        para["gemini_text"] = ja_text
        bubble_str = "내부" if para.get("bubble_bbox") else "외부"
        if ko_text is not None:
            para["translated_text"] = ko_text
            print(f"  [문단 {pid}, 말풍선 {bubble_str}] {ja_text} -> {ko_text}")
        else:
            # 인식은 성공, 번역만 실패 - gemini_text(더 정확한 원문)는 남겨두고
            # 번역은 뒤 단계의 hell0ks 폴백에 맡김(그래도 원문 품질은 개선됨).
            print(f"  [문단 {pid}, 말풍선 {bubble_str}] {ja_text} (번역 실패, hell0ks로 폴백 예정)")
        if ja_text != fallback_text:
            print(f"    (참고: 기존 PaddleOCR 원문과 다름 - PaddleOCR: {fallback_text})")

    out_path = args.out or args.json
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nGemini 인식+번역 결과 포함 JSON 저장: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Gemini API로 문단 전체를 한 번에 읽고(1회) 한 번에 번역(1회) - recognition+translation 대체"
    )
    parser.add_argument("--json", type=str,
                        default="./../../output/ocr/masking_test/shirobako_paragraphs.json",
                        help="paragraphs JSON 경로 (masking 출력)")
    parser.add_argument("--img", type=str, default=None,
                        help="원본 이미지 경로 (기본값: JSON 안의 image_path 사용)")
    parser.add_argument("--out", type=str, default=None,
                        help="결과 JSON 저장 경로 (기본값: --json 경로에 덮어쓰기)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"사용할 Gemini 모델 (기본값: {DEFAULT_MODEL})")
    parser.add_argument("--crop-pad", type=int, default=DEFAULT_CROP_PAD,
                        help=f"문단 bbox 주변 크롭 여유 픽셀 (기본값: {DEFAULT_CROP_PAD})")
    parser.add_argument("--request-interval", type=float, default=DEFAULT_REQUEST_INTERVAL,
                        help=f"인식 호출과 번역 호출 사이 대기 시간(초) (기본값: {DEFAULT_REQUEST_INTERVAL})")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
