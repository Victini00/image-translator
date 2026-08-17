import sys
import os
import json
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Windows 콘솔 기본 인코딩(cp949)이 일본어/특수문자 출력 시 깨지는 것 방지
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

"""
<사용법>

Masking(paragraphs.json) + Cleaning(지운 이미지)을 받아서,
1. 각 문단의 원문(merged_text)을 hell0ks+LoRA로 한국어로 번역하고
2. 말풍선 안/밖 위치에 맞춰 폰트 크기 자동 조절 + 줄바꿈해서 그려 넣는다.

말풍선 안(bubble_bbox 있음): 단색 배경 위에 일반 텍스트
말풍선 밖(bubble_bbox 없음, 효과음 등): 테두리(stroke) 있는 텍스트
"""

# 모델 이름·경로·배치 파라미터는 src/config.py가 단일 출처다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# 번역 모델 (translation_hell0ks.py와 동일한 방식)
DEFAULT_MODEL = config.TRANSLATION_MODEL
DEFAULT_LORA = config.TRANSLATION_LORA_DIR

# 폰트 경로와 배치 규칙(get_box_for_paragraph)은 웹 UI 미리보기와 공유해야 하므로
# 의존성 없는 render_layout 모듈에 두고 가져다 쓴다 (규칙이 갈라지지 않게).
from render_layout import (  # noqa: E402
    DEFAULT_INTERIOR_FONT,
    DEFAULT_EXTERIOR_FONT,
    get_box_for_paragraph,
    text_width,
    wrap_lines,
    fit_text,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_translation_model(model_name, lora_path, device):
    """hell0ks 베이스 모델(+ 있으면 LoRA)을 로드한다. translation_hell0ks.py와 동일한 방식."""
    print(f"번역 모델 로딩: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if device.type == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
        ).to(device)

    if lora_path:
        print(f"LoRA 어댑터 적용: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    print("번역 모델 로딩 완료!")
    return tokenizer, model


def translate_text(tokenizer, model, text):
    messages = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def draw_paragraph_text(image, draw, para, text, interior_font_path, exterior_font_path,
                         max_size, min_size, stroke_width, bubble_inset_ratio):
    (bx1, by1, bx2, by2), is_interior = get_box_for_paragraph(para, bubble_inset_ratio)
    box_w, box_h = bx2 - bx1, by2 - by1

    if is_interior:
        font_path = interior_font_path
        sw = 0
        fill = (20, 20, 20)
        stroke_fill = None
    else:
        font_path = exterior_font_path
        sw = stroke_width
        fill = (20, 20, 20)
        stroke_fill = (255, 255, 255)

    # font_size: 웹 편집 UI에서 사용자가 크기를 직접 정한 경우 자동 조절 대신 그 값을 씀
    font, lines, line_h = fit_text(
        draw, text, font_path, box_w, box_h, max_size, min_size, sw,
        force_size=para.get("font_size"),
    )

    total_h = line_h * len(lines)
    y = by1 + max(0, (box_h - total_h) // 2)
    for line in lines:
        w = text_width(draw, line, font, sw)
        x = bx1 + max(0, (box_w - w) // 2)
        kwargs = {"font": font, "fill": fill}
        if sw > 0:
            kwargs["stroke_width"] = sw
            kwargs["stroke_fill"] = stroke_fill
        draw.text((x, y), line, **kwargs)
        y += line_h


def run_rendering(args):
    device = torch.device(args.device) if args.device else get_device()
    print(f"사용 디바이스: {device}")

    # --use-existing-translation: recognition_translation_gemini.py가 이미
    # translated_text를 채워놨으면 hell0ks 모델을 아예 로드하지 않고 건너뜀
    # (Gemini 하이브리드 파이프라인용 - 번역 이중 작업/불필요한 모델 로딩 방지).
    tokenizer, model = (None, None) if args.use_existing_translation \
        else load_translation_model(args.model, args.lora_path, device)

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    image = Image.open(args.cleaned_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    print(f"{len(data['paragraphs'])}개 문단 처리 시작...")

    for para in data["paragraphs"]:
        # experiments/translation_correction.py(보류된 교정 단계)가 남기는 corrected_text,
        # recognition_translation_gemini.py가 다시 읽은 gemini_text, 둘 다
        # 없으면(구버전 JSON 등) merged_text로 하위 호환 - 표시/로그용 원문.
        ja_text = para.get("gemini_text", para.get("corrected_text", para["merged_text"])).strip()
        if not ja_text:
            continue

        if args.use_existing_translation:
            ko_text = para.get("translated_text")
            if not ko_text:
                # Gemini 단계가 이 문단만 실패했을 수 있음 - 이 경우에만 폴백으로
                # hell0ks를 그때그때 로드해서 씀 (매 문단마다 로드하지 않도록 캐싱).
                if model is None:
                    tokenizer, model = load_translation_model(args.model, args.lora_path, device)
                ko_text = translate_text(tokenizer, model, ja_text)
        else:
            ko_text = translate_text(tokenizer, model, ja_text)
        para["translated_text"] = ko_text

        draw_paragraph_text(
            image, draw, para, ko_text,
            args.interior_font, args.exterior_font,
            args.max_font_size, args.min_font_size,
            args.stroke_width, args.bubble_inset_ratio,
        )
        bubble_str = "내부" if para.get("bubble_bbox") else "외부"
        print(f"  [문단 {para['id']}, 말풍선 {bubble_str}] {ja_text} -> {ko_text}")

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    image.save(args.out)
    print(f"\n렌더링 결과 저장: {args.out}")

    if args.save_json:
        save_dir = os.path.dirname(args.save_json)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"번역문 포함 JSON 저장: {args.save_json}")


def main():
    parser = argparse.ArgumentParser(description="번역 붙이기 + 렌더링 - 번역문을 Cleaning 결과 이미지에 그려 넣기")
    parser.add_argument("--json", type=str,
                        default=os.path.join(config.OCR_OUTPUT_DIR,
                                             config.paragraphs_json(config.SAMPLE_IMAGE_NAME)),
                        help="paragraphs JSON 경로 (masking 출력)")
    parser.add_argument("--cleaned-image", type=str,
                        default=os.path.join(config.CLEANED_OUTPUT_DIR,
                                             config.cleaned_image(config.SAMPLE_IMAGE_NAME)),
                        help="Cleaning 결과 이미지 경로")
    parser.add_argument("--out", type=str,
                        default=os.path.join(config.RENDERED_OUTPUT_DIR,
                                             config.rendered_image(config.SAMPLE_IMAGE_NAME)),
                        help="렌더링 결과 저장 경로")
    parser.add_argument("--save-json", type=str, default=None,
                        help="번역문(translated_text) 포함해서 저장할 JSON 경로 (지정 안 하면 저장 안 함)")

    parser.add_argument("--interior-font", type=str, default=DEFAULT_INTERIOR_FONT,
                        help="말풍선 안쪽 텍스트용 폰트 경로")
    parser.add_argument("--exterior-font", type=str, default=DEFAULT_EXTERIOR_FONT,
                        help="말풍선 밖(효과음) 텍스트용 폰트 경로")
    parser.add_argument("--max-font-size", type=int, default=config.MAX_FONT_SIZE,
                        help=f"폰트 크기 자동 조절 시작값 (기본값: {config.MAX_FONT_SIZE})")
    parser.add_argument("--min-font-size", type=int, default=config.MIN_FONT_SIZE,
                        help=f"폰트 크기 자동 조절 최솟값, 이 이하로는 안 줄이고 넘치게 둠 "
                             f"(기본값: {config.MIN_FONT_SIZE})")
    parser.add_argument("--stroke-width", type=int, default=config.STROKE_WIDTH,
                        help=f"말풍선 밖 텍스트 테두리 두께 (기본값: {config.STROKE_WIDTH})")
    parser.add_argument("--bubble-inset-ratio", type=float, default=config.BUBBLE_INSET_RATIO,
                        help=f"말풍선 bubble_bbox를 안쪽으로 줄이는 비율 "
                             f"(기본값: {config.BUBBLE_INSET_RATIO})")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="번역용 HuggingFace 모델 이름 또는 경로")
    parser.add_argument("--lora-path", type=str, default=DEFAULT_LORA,
                        help="번역 LoRA 어댑터 경로 (빈 문자열이면 base 모델만 사용)")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                        help="연산 디바이스 (기본값: cuda > mps > cpu 자동 감지)")
    parser.add_argument("--use-existing-translation", action="store_true",
                        help="JSON에 이미 채워진 translated_text(예: recognition_translation_gemini.py "
                             "결과)를 그대로 쓰고 hell0ks 번역 모델을 로드하지 않음")

    args = parser.parse_args()
    run_rendering(args)


if __name__ == "__main__":
    main()
