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

# 번역 모델 (translation_hell0ks.py와 동일한 방식)
DEFAULT_MODEL = "hell0ks/ja-ko-vn-7b-v1"
DEFAULT_LORA = "./../../models/translation/hell0ks_ja-ko-vn-7b-v1/lora/v1"

# 폰트 (font_test.py로 미리보기 확인 후, 4종 실사용 비교까지 거쳐서 고른 것들)
DEFAULT_INTERIOR_FONT = r"C:\Users\a\AppData\Local\Microsoft\Windows\Fonts\GmarketSansMedium.otf"
DEFAULT_EXTERIOR_FONT = r"C:\Users\a\AppData\Local\Microsoft\Windows\Fonts\HY피오피M.TTF"


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


def get_box_for_paragraph(para, inset_ratio=0.15):
    """
    텍스트를 배치할 영역을 결정한다.
    - 말풍선 안: bubble_bbox를 안쪽으로 살짝 줄여서 사용 (테두리에 안 닿게).
      bubble_bbox는 사각형이라 실제 말풍선(둥근/뾰족한 모양)보다 크므로,
      비율로 줄여서 안전 여백을 둔다.
    - 말풍선 밖: mask_bbox 그대로 사용.
    반환: ((x1,y1,x2,y2), is_interior)
    """
    if para.get("bubble_bbox"):
        x1, y1, x2, y2 = para["bubble_bbox"]
        w, h = x2 - x1, y2 - y1
        ix = min(int(w * inset_ratio), max(0, w // 2 - 5))
        iy = min(int(h * inset_ratio), max(0, h // 2 - 5))
        return (x1 + ix, y1 + iy, x2 - ix, y2 - iy), True
    else:
        x1, y1, x2, y2 = para["mask_bbox"]
        return (x1, y1, x2, y2), False


def text_width(draw, text, font, stroke_width=0):
    """텍스트 폭을 잰다. 설치된 Pillow(9.5) textlength()는 stroke_width를 지원하지
    않아서, stroke 포함 실제 폭이 필요할 땐 textbbox로 잰다."""
    if not text:
        return 0
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    return bbox[2] - bbox[0]


def wrap_lines(draw, text, font, box_w, stroke_width=0, allow_char_split=False):
    """
    박스 너비에 맞춰 띄어쓰기 기준으로 줄바꿈한다. allow_char_split=False면
    단어(어절)를 절대 쪼개지 않는다 - 한 단어가 박스보다 넓어도 그 줄이 넘치는
    채로 그대로 반환한다 (가독성: 단어/조사가 잘리면 안 되므로, 이 경우는
    fit_text 쪽에서 폰트 크기를 더 줄여서 해결한다).
    allow_char_split=True일 때만(폰트를 최소 크기까지 줄여도 안 들어가는
    최후의 경우) 글자 단위로 쪼갠다.
    """
    def width(s):
        return text_width(draw, s, font, stroke_width)

    lines = []
    cur = ""
    for word in text.split(" "):
        candidate = f"{cur} {word}".strip() if cur else word
        if not cur:
            cur = candidate
        elif width(candidate) <= box_w:
            cur = candidate
        else:
            lines.append(cur)
            cur = word

        if allow_char_split and width(cur) > box_w:
            # cur(방금 확정/시작된 단어)만으로도 이미 박스보다 넓음 -> 글자 단위로 쪼갬
            sub = ""
            chunks = []
            for ch in cur:
                cand2 = sub + ch
                if width(cand2) <= box_w or not sub:
                    sub = cand2
                else:
                    chunks.append(sub)
                    sub = ch
            lines.extend(chunks)
            cur = sub
    if cur:
        lines.append(cur)
    return lines or [""]


def _measure(draw, font, lines, stroke_width, line_spacing):
    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * line_spacing)
    total_h = line_h * len(lines)
    max_line_w = max((text_width(draw, l, font, stroke_width) for l in lines), default=0)
    return line_h, total_h, max_line_w


def fit_text(draw, text, font_path, box_w, box_h, max_size, min_size,
             stroke_width=0, line_spacing=1.25):
    """
    박스 경계를 절대 넘지 않는 걸 최우선으로, 그 안에서 가장 읽기 좋은(줄이
    적고, 그 다음으로 글자가 크고, 단어가 안 잘리는) 렌더링을 찾는다.

    1단계: max_size~min_size 범위 전체를 훑어(띄어쓰기 단위 줄바꿈만 허용,
           단어를 안 쪼갬) 박스 안에 들어가는 조합들 중, 줄 수가 가장 적은
           것을 고르고, 줄 수가 같으면 그중 폰트가 가장 큰 것을 고른다.
           (원문이 한 줄이었는데 번역하면서 살짝 길어졌다고 무작정 두 줄로
           쪼개기보다, 폰트를 조금 줄여서 한 줄을 유지하는 쪽을 우선한다.)
    2단계: min_size까지 줄여도 박스에 안 들어가면(단어 하나가 너무 길거나
           텍스트가 너무 많음) - 경계를 넘지 않는 게 단어를 안 쪼개는 것보다
           우선이므로, 글자 단위 줄바꿈으로 전환해서 1px까지 계속 줄여서라도
           반드시 박스 안에 맞춘다.
    """
    box_w = max(1, box_w)
    box_h = max(1, box_h)

    fits = []
    for size in range(max_size, min_size - 1, -1):
        font = ImageFont.truetype(font_path, size)
        lines = wrap_lines(draw, text, font, box_w, stroke_width, allow_char_split=False)
        line_h, total_h, max_line_w = _measure(draw, font, lines, stroke_width, line_spacing)
        if total_h <= box_h and max_line_w <= box_w:
            fits.append((len(lines), size, font, lines, line_h))

    if fits:
        fits.sort(key=lambda f: (f[0], -f[1]))  # 줄 수 적은 것 우선, 동률이면 큰 폰트 우선
        _, _, font, lines, line_h = fits[0]
        return (font, lines, line_h)

    last = None
    for size in range(min_size - 1, 0, -1):
        font = ImageFont.truetype(font_path, size)
        lines = wrap_lines(draw, text, font, box_w, stroke_width, allow_char_split=True)
        line_h, total_h, max_line_w = _measure(draw, font, lines, stroke_width, line_spacing)
        last = (font, lines, line_h)
        if total_h <= box_h and max_line_w <= box_w:
            return last

    # size=1까지도 이론상 안 맞는 극단적인 경우(텍스트가 지나치게 많음) -
    # 그래도 마지막(가장 작은 글자) 결과를 반환한다.
    return last


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

    font, lines, line_h = fit_text(draw, text, font_path, box_w, box_h, max_size, min_size, sw)

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
        # translation_correction.py가 OCR 오인식을 교정한 corrected_text,
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
                        default="./../../output/ocr/masking_test/shirobako_paragraphs.json",
                        help="paragraphs JSON 경로 (masking 출력)")
    parser.add_argument("--cleaned-image", type=str,
                        default="./../../output/inpainting/cleaned/shirobako_cleaned_D_final.png",
                        help="Cleaning 결과 이미지 경로")
    parser.add_argument("--out", type=str,
                        default="./../../output/translation/rendering/shirobako_rendered.png",
                        help="렌더링 결과 저장 경로")
    parser.add_argument("--save-json", type=str, default=None,
                        help="번역문(translated_text) 포함해서 저장할 JSON 경로 (지정 안 하면 저장 안 함)")

    parser.add_argument("--interior-font", type=str, default=DEFAULT_INTERIOR_FONT,
                        help="말풍선 안쪽 텍스트용 폰트 경로")
    parser.add_argument("--exterior-font", type=str, default=DEFAULT_EXTERIOR_FONT,
                        help="말풍선 밖(효과음) 텍스트용 폰트 경로")
    parser.add_argument("--max-font-size", type=int, default=36,
                        help="폰트 크기 자동 조절 시작값 (기본값: 36)")
    parser.add_argument("--min-font-size", type=int, default=12,
                        help="폰트 크기 자동 조절 최솟값, 이 이하로는 안 줄이고 넘치게 둠 (기본값: 12)")
    parser.add_argument("--stroke-width", type=int, default=2,
                        help="말풍선 밖 텍스트 테두리 두께 (기본값: 2)")
    parser.add_argument("--bubble-inset-ratio", type=float, default=0.15,
                        help="말풍선 bubble_bbox를 안쪽으로 줄이는 비율 (기본값: 0.15)")

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
