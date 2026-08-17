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

# 모델 이름·크롭 여유 픽셀·.env 위치는 src/config.py가 단일 출처다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# 프로젝트 루트의 .env에서 GEMINI_API_KEY를 읽어와 환경변수로 등록한다.
# 경로를 명시하므로 어느 작업 디렉터리에서 실행하든(직접 실행/subprocess) 동작함.
load_dotenv(config.ENV_PATH)

"""
사전 준비: 프로젝트 루트에 .env 파일을 만들고 GEMINI_API_KEY=... 한 줄을
넣어두면 자동으로 읽어옴.
"""

DEFAULT_MODEL = config.GEMINI_MODEL
DEFAULT_CROP_PAD = config.GEMINI_CROP_PAD

SYSTEM_PROMPT = """당신은 일본 만화(manga) 페이지에서 잘라낸(crop) 여러 개의
이미지 조각을 보고, 각 조각의 일본어 텍스트를 정확히 읽어서 한국어로 번역하는 작업을 합니다.

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

번역할 때 주의할 점:
- 만화 대사/효과음의 어조와 뉘앙스를 살려서 자연스러운 한국어로 번역하세요.
- 직역보다는 상황에 맞는 자연스러운 구어체를 우선하세요.
- 각 id는 독립된 문단(발화)이므로, id끼리 서로 이어붙이거나 순서를 바꾸지 마세요.
- 번역문에는 줄바꿈을 넣지 마세요(줄바꿈은 렌더링 단계가 알아서 합니다).

반드시 지정된 JSON 스키마 형식으로만 응답하세요. 모든 id에 대해 결과를 빠짐없이 반환하세요."""


class ParagraphItem(BaseModel):
    id: int
    japanese_text: str
    korean_translation: str


class BatchResult(BaseModel):
    items: list[ParagraphItem]


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


def _generate_with_retry(client, model, contents, config, max_retries=3):
    """
    일시적인 실패는 잠깐 쉬었다가 다시 시도
    """
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as e:
            if attempt >= max_retries:
                raise
            if e.code == 429:
                delay = _parse_retry_delay(e.message)
                print(f"  [경고] 요청 한도(429), {delay:.1f}초 대기 후 재시도 "
                      f"({attempt + 1}/{max_retries})...")
            elif e.code in (500, 502, 503, 504):
                delay = 3.0 * (2 ** attempt)  # 3초 -> 6초 -> 12초
                print(f"  [경고] 서버 일시 오류({e.code}), {delay:.1f}초 대기 후 재시도 "
                      f"({attempt + 1}/{max_retries})...")
            else:
                raise
            time.sleep(delay)


def recognize_and_translate_batch(client, model, id_crop_pairs, fallback_texts):
    """
    성공한 id만 담긴 {id: (원문, 번역문)} dict를 반환. 전체 실패 시 빈 dict.
    """
    contents = []
    for pid, crop in id_crop_pairs:
        hint = fallback_texts.get(pid, "")
        label = f"문단 id={pid}"
        if hint:
            label += f" (참고용 기존 OCR: \"{hint}\")"
        contents.append(f"{label}:")
        contents.append(crop.convert("RGB"))
    contents.append("위 각 이미지 조각의 일본어 텍스트를 읽고 한국어로 번역해서 id와 매칭해 반환해 주세요.")

    try:
        response = _generate_with_retry(
            client, model, contents,
            types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=BatchResult,
            ),
        )
    except errors.APIError as e:
        print(f"  [경고] Gemini 호출 API 오류({e.code}), 전체 폴백: {e.message}")
        return {}

    if response.parsed is None:
        print("  [경고] Gemini 응답 파싱 실패, 전체 폴백")
        return {}

    return {
        item.id: (item.japanese_text.strip(), item.korean_translation.strip())
        for item in response.parsed.items
    }


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

    print(f"{len(paragraphs)}개 문단 인식+번역 요청 중 (모델: {args.model}, API 호출 1회)...")
    results = recognize_and_translate_batch(client, args.model, id_crop_pairs, fallback_texts)
    print(f"  -> {len(results)}/{len(paragraphs)}개 문단 성공")

    fallback_count = 0
    for para in paragraphs:
        pid = para["id"]
        fallback_text = fallback_texts[pid]
        result = results.get(pid)

        if result is None:
            # 응답에 이 id가 없음 - 기존 OCR 결과로 완전히 폴백됨
            # (뒤 단계의 hell0ks 번역이 merged_text를 그대로 씀).
            fallback_count += 1
            print(f"  [문단 {pid}] Gemini 실패 -> 로컬(hell0ks)로 폴백: {fallback_text}")
            continue

        ja_text, ko_text = result
        para["gemini_text"] = ja_text
        para["translated_text"] = ko_text
        bubble_str = "내부" if para.get("bubble_bbox") else "외부"
        print(f"  [문단 {pid}, 말풍선 {bubble_str}] {ja_text} -> {ko_text}")
        if ja_text != fallback_text:
            print(f"    (참고: 기존 PaddleOCR 원문과 다름 - PaddleOCR: {fallback_text})")

    # 웹 UI가 로그에서 폴백 여부를 집계할 수 있도록 요약 한 줄을 남긴다
    print(f"[요약] Gemini 성공 {len(results)}개 / 로컬 폴백 {fallback_count}개 (전체 {len(paragraphs)}개)")

    out_path = args.out or args.json
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nGemini 인식+번역 결과 포함 JSON 저장: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Gemini API로 문단 전체를 한 번의 호출로 읽고 번역 (recognition+translation 대체)"
    )
    parser.add_argument("--json", type=str,
                        default=os.path.join(config.OCR_OUTPUT_DIR,
                                             config.paragraphs_json(config.SAMPLE_IMAGE_NAME)),
                        help="paragraphs JSON 경로 (masking 출력)")
    parser.add_argument("--img", type=str, default=None,
                        help="원본 이미지 경로 (기본값: JSON 안의 image_path 사용)")
    parser.add_argument("--out", type=str, default=None,
                        help="결과 JSON 저장 경로 (기본값: --json 경로에 덮어쓰기)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"사용할 Gemini 모델 (기본값: {DEFAULT_MODEL})")
    parser.add_argument("--crop-pad", type=int, default=DEFAULT_CROP_PAD,
                        help=f"문단 bbox 주변 크롭 여유 픽셀 (기본값: {DEFAULT_CROP_PAD})")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
