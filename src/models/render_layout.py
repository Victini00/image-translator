import os
import sys

from PIL import Image, ImageDraw, ImageFont

# 폰트 경로와 배치 파라미터는 src/config.py가 단일 출처
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

DEFAULT_INTERIOR_FONT = config.INTERIOR_FONT
DEFAULT_EXTERIOR_FONT = config.EXTERIOR_FONT

DEFAULT_MAX_FONT_SIZE = config.MAX_FONT_SIZE
DEFAULT_MIN_FONT_SIZE = config.MIN_FONT_SIZE
DEFAULT_STROKE_WIDTH = config.STROKE_WIDTH
DEFAULT_LINE_SPACING = config.LINE_SPACING
DEFAULT_BUBBLE_INSET_RATIO = config.BUBBLE_INSET_RATIO
LINE_INCREASE_GAIN = config.LINE_INCREASE_GAIN


def get_box_for_paragraph(para, inset_ratio=DEFAULT_BUBBLE_INSET_RATIO):
    """
    텍스트를 배치할 영역을 결정
    """
    if para.get("render_bbox"):
        x1, y1, x2, y2 = para["render_bbox"]
        # 폰트/테두리 종류는 위치를 옮겨도 원래 말풍선 안/밖 구분을 그대로 따른다.
        return (x1, y1, x2, y2), bool(para.get("bubble_bbox"))

    if para.get("bubble_bbox"):
        x1, y1, x2, y2 = para["bubble_bbox"]
        w, h = x2 - x1, y2 - y1
        ix = min(int(w * inset_ratio), max(0, w // 2 - 5))
        iy = min(int(h * inset_ratio), max(0, h // 2 - 5))
        return (x1 + ix, y1 + iy, x2 - ix, y2 - iy), True

    x1, y1, x2, y2 = para["mask_bbox"]
    return (x1, y1, x2, y2), False


def text_width(draw, text, font, stroke_width=0):
    """
    텍스트 폭을 측정
    """
    if not text:
        return 0
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    return bbox[2] - bbox[0]


def wrap_lines(draw, text, font, box_w, stroke_width=0, allow_char_split=False):
    """
    박스 너비에 맞춰 줄바꿈한다.
    * 텍스트에 줄바꿈 문자(\\n)가 들어 있으면 그 위치에서는 무조건 줄을 바꾼다
    """
    if "\n" in text:
        lines = []
        for segment in text.split("\n"):
            segment = segment.strip()
            if not segment:
                lines.append("")   # 빈 줄도 사용자가 의도한 것이므로 유지
                continue
            lines.extend(
                _wrap_segment(draw, segment, font, box_w, stroke_width, allow_char_split)
            )
        return lines or [""]
    return _wrap_segment(draw, text, font, box_w, stroke_width, allow_char_split)


def _wrap_segment(draw, text, font, box_w, stroke_width=0, allow_char_split=False):
    """
    줄바꿈 문자가 없는 한 덩어리를 박스 너비에 맞춰 자동 줄바꿈
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


def _pick_best_fit(fits):
    """
    박스에 들어가는 조합들 중 하나를 고른다. 
    """
    best_by_lines = {}
    for n_lines, size, font, lines, line_h in fits:
        if n_lines not in best_by_lines or size > best_by_lines[n_lines][1]:
            best_by_lines[n_lines] = (n_lines, size, font, lines, line_h)

    best = None
    for n_lines in sorted(best_by_lines):
        candidate = best_by_lines[n_lines]
        if best is None or candidate[1] >= best[1] * (1 + LINE_INCREASE_GAIN):
            best = candidate
    return best


def fit_text(draw, text, font_path, box_w, box_h, max_size, min_size,
             stroke_width=0, line_spacing=1.25, force_size=None):
    """
    박스 경계를 절대 넘지 않는 걸 최우선으로, 그 안에서 가장 읽기 좋은(줄이
    적고, 그 다음으로 글자가 크고, 단어가 안 잘리는) 렌더링을 찾는다.
    """
    box_w = max(1, box_w)
    box_h = max(1, box_h)

    if force_size:
        size = max(1, int(force_size))
        font = ImageFont.truetype(font_path, size)
        lines = wrap_lines(draw, text, font, box_w, stroke_width, allow_char_split=True)
        line_h, _, _ = _measure(draw, font, lines, stroke_width, line_spacing)
        return (font, lines, line_h)

    fits = []
    for size in range(max_size, min_size - 1, -1):
        font = ImageFont.truetype(font_path, size)
        lines = wrap_lines(draw, text, font, box_w, stroke_width, allow_char_split=False)
        line_h, total_h, max_line_w = _measure(draw, font, lines, stroke_width, line_spacing)
        if total_h <= box_h and max_line_w <= box_w:
            fits.append((len(lines), size, font, lines, line_h))

    if fits:
        _, _, font, lines, line_h = _pick_best_fit(fits)
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


def layout_text(text, box_w, box_h, is_interior,
                max_size=DEFAULT_MAX_FONT_SIZE, min_size=DEFAULT_MIN_FONT_SIZE,
                stroke_width=DEFAULT_STROKE_WIDTH, line_spacing=DEFAULT_LINE_SPACING,
                force_size=None):
    """
    웹 편집 UI 미리보기용 - 실제 렌더링이 글자를 어디에 어떤 크기로 찍을지
    그대로 계산해서 돌려준다.
    """
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    font_path = DEFAULT_INTERIOR_FONT if is_interior else DEFAULT_EXTERIOR_FONT
    sw = 0 if is_interior else stroke_width

    font, lines, line_h = fit_text(
        draw, text, font_path, box_w, box_h, max_size, min_size, sw, line_spacing,
        force_size=force_size,
    )

    # draw_paragraph_text()와 동일한 중앙 정렬 계산
    total_h = line_h * len(lines)
    block_y = max(0, (box_h - total_h) // 2)
    line_x = [
        max(0, (box_w - text_width(draw, line, font, sw)) // 2)
        for line in lines
    ]
    ascent, _descent = font.getmetrics()

    return {
        "size": font.size,
        "lines": lines,
        "line_h": line_h,
        "ascent": ascent,
        "stroke": sw,
        "line_x": line_x,
        "block_y": block_y,
    }
