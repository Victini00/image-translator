import os
import argparse
from PIL import Image, ImageDraw, ImageFont

"""
<사용법>

렌더링(inpainting_rendering.py)에 쓸 폰트를 고르기 전에, 후보 폰트들이 실제로
어떻게 나오는지 한 장의 미리보기 이미지로 모아서 확인한다.

- INTERIOR_CANDIDATES: 말풍선 안쪽용 후보 (일반 렌더링)
- EXTERIOR_CANDIDATES: 말풍선 밖(효과음 등)용 후보 (굵은 폰트 + 테두리 렌더링,
  ImageDraw.text의 stroke_width/stroke_fill로 테두리를 입힌다 - 폰트 파일
  자체가 아니라 렌더링 기법으로 테두리를 만드는 것)
"""

INTERIOR_SAMPLE = "안녕? 오늘 날씨 진짜 좋다!"
EXTERIOR_SAMPLE = "쿵! 콰과광!!"

INTERIOR_CANDIDATES = [
    ("Hancom Gothic Regular", r"C:\Windows\Fonts\Hancom Gothic Regular.ttf"),
    ("Hancom Gothic Bold", r"C:\Windows\Fonts\Hancom Gothic Bold.ttf"),
    ("HANBatang", r"C:\Windows\Fonts\HANBatang.ttf"),
    ("HANBatang Bold", r"C:\Windows\Fonts\HANBatangB.ttf"),
    # BIZ UDGothic은 설치는 되어 있으나 한글 글리프가 없어 제외 (일본어 전용 폰트)
    ("MaruBuri", r"C:\Users\a\AppData\Local\Microsoft\Windows\Fonts\MaruBuri-Regular.otf"),
    ("Gmarket Sans Medium", r"C:\Users\a\AppData\Local\Microsoft\Windows\Fonts\GmarketSansMedium.otf"),
]

EXTERIOR_CANDIDATES = [
    ("SB Aggro Bold", r"C:\Users\a\AppData\Local\Microsoft\Windows\Fonts\SB 어그로 B.ttf"),
    ("SB Aggro Medium", r"C:\Users\a\AppData\Local\Microsoft\Windows\Fonts\SB 어그로 M.ttf"),
    ("HY POP M", r"C:\Users\a\AppData\Local\Microsoft\Windows\Fonts\HY피오피M.TTF"),
    ("h2pop Bold", r"C:\Users\a\AppData\Local\Microsoft\Windows\Fonts\h2popb.TTF"),
]


LABEL_FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"  # 한글 파일명 라벨을 그리기 위한 고정 폰트


def render_row(font_label, font_path, sample_text, row_w, row_h, font_size,
                stroke_width=0, stroke_fill=(255, 255, 255), bg_color=(255, 255, 255)):
    row = Image.new("RGB", (row_w, row_h), bg_color)
    draw = ImageDraw.Draw(row)
    draw.rectangle([0, 0, row_w - 1, row_h - 1], outline=(120, 120, 120))

    label_color = (120, 120, 120) if sum(bg_color) > 380 else (230, 230, 230)
    label = f"{font_label}  ({os.path.basename(font_path)})"
    label_font = ImageFont.truetype(LABEL_FONT_PATH, 14)
    draw.text((16, 8), label, font=label_font, fill=label_color)

    if not os.path.exists(font_path):
        draw.text((16, row_h // 2 - 10), "폰트 파일 없음", font=label_font, fill=(200, 0, 0))
        return row

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        draw.text((16, row_h // 2 - 10), f"로드 실패: {e}", font=label_font, fill=(200, 0, 0))
        return row

    text_kwargs = {"font": font, "fill": (20, 20, 20)}
    if stroke_width > 0:
        text_kwargs["stroke_width"] = stroke_width
        text_kwargs["stroke_fill"] = stroke_fill
    draw.text((16, 36), sample_text, **text_kwargs)
    return row


def build_preview(candidates, sample_text, out_path, font_size=32,
                   row_w=760, row_h=100, stroke_width=0, stroke_fill=(255, 255, 255),
                   bg_color=(255, 255, 255)):
    rows = [
        render_row(label, path, sample_text, row_w, row_h, font_size,
                   stroke_width, stroke_fill, bg_color)
        for label, path in candidates
    ]
    sheet = Image.new("RGB", (row_w, row_h * len(rows)), bg_color)
    for i, row in enumerate(rows):
        sheet.paste(row, (0, i * row_h))
    sheet.save(out_path)
    print(f"저장 완료: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="렌더링용 후보 폰트 미리보기 생성")
    parser.add_argument("--out-dir", type=str, default="./../../output/translation/font_test",
                        help="미리보기 이미지 저장 폴더")
    parser.add_argument("--font-size", type=int, default=32,
                        help="미리보기 글자 크기 (기본값: 32)")
    parser.add_argument("--stroke-width", type=int, default=3,
                        help="외부(효과음)용 미리보기 테두리 두께 (기본값: 3)")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    build_preview(
        INTERIOR_CANDIDATES, INTERIOR_SAMPLE,
        os.path.join(args.out_dir, "interior_fonts.png"),
        font_size=args.font_size,
    )
    build_preview(
        EXTERIOR_CANDIDATES, EXTERIOR_SAMPLE,
        os.path.join(args.out_dir, "exterior_fonts.png"),
        font_size=args.font_size, stroke_width=args.stroke_width,
        stroke_fill=(255, 255, 255), bg_color=(130, 130, 130),
    )


if __name__ == "__main__":
    main()
