"""
렌더링 배치 규칙과 폰트 경로 - 무거운 의존성 없는 공통 모듈.

inpainting_rendering.py(실제 렌더링)와 web/app.py(편집 UI 미리보기)가 같은 규칙을
써야 미리보기와 최종 결과가 어긋나지 않는다. 그런데 inpainting_rendering.py는
torch/transformers를 import하므로 웹 서버가 그걸 통째로 불러오면 기동이 느리고
환경 충돌(numpy/numba)도 난다. 그래서 양쪽이 공유해야 하는 부분만 여기로 뺐다.

여기에는 PIL(Pillow) 외 무거운 의존성을 추가하지 말 것.
"""

from PIL import Image, ImageDraw, ImageFont

# 폰트 (font_test.py로 미리보기 확인 후, 4종 실사용 비교까지 거쳐서 고른 것들)
DEFAULT_INTERIOR_FONT = r"C:\Users\a\AppData\Local\Microsoft\Windows\Fonts\GmarketSansMedium.otf"
DEFAULT_EXTERIOR_FONT = r"C:\Users\a\AppData\Local\Microsoft\Windows\Fonts\HY피오피M.TTF"

# 폰트 크기 자동 조절 / 줄바꿈 파라미터 (inpainting_rendering.py의 argparse 기본값과 동일)
DEFAULT_MAX_FONT_SIZE = 36
DEFAULT_MIN_FONT_SIZE = 12
DEFAULT_STROKE_WIDTH = 2
DEFAULT_LINE_SPACING = 1.25
DEFAULT_BUBBLE_INSET_RATIO = 0.15

# 줄을 하나 더 쓰는 것을 허용하는 기준. 줄 수가 늘어난 대신 글자 크기가 이 비율
# 이상 커질 때만 그 조합을 택한다(_pick_best_fit 참고).
# 낮추면 글자가 커지는 대신 줄이 잘게 나뉘고, 높이면 그 반대가 된다.
LINE_INCREASE_GAIN = 0.15


def get_box_for_paragraph(para, inset_ratio=DEFAULT_BUBBLE_INSET_RATIO):
    """
    텍스트를 배치할 영역을 결정한다.
    - render_bbox가 있으면(웹 UI에서 사용자가 위치/크기를 직접 조정한 경우)
      다른 모든 규칙보다 우선해서 그대로 사용한다.
    - 말풍선 안: bubble_bbox를 안쪽으로 살짝 줄여서 사용 (테두리에 안 닿게).
      bubble_bbox는 사각형이라 실제 말풍선(둥근/뾰족한 모양)보다 크므로,
      비율로 줄여서 안전 여백을 둔다.
    - 말풍선 밖: mask_bbox 그대로 사용.
    반환: ((x1,y1,x2,y2), is_interior)
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
    """텍스트 폭을 잰다. 설치된 Pillow(9.5) textlength()는 stroke_width를 지원하지
    않아서, stroke 포함 실제 폭이 필요할 땐 textbbox로 잰다."""
    if not text:
        return 0
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    return bbox[2] - bbox[0]


def wrap_lines(draw, text, font, box_w, stroke_width=0, allow_char_split=False):
    """
    박스 너비에 맞춰 줄바꿈한다.

    텍스트에 줄바꿈 문자(\\n)가 들어 있으면 그 위치에서는 무조건 줄을 바꾼다
    (웹 편집 UI에서 사용자가 직접 넣은 줄바꿈을 존중하기 위한 것). 그 사이 구간은
    아래 규칙대로 자동 줄바꿈한다.

    allow_char_split=False면 단어(어절)를 절대 쪼개지 않는다 - 한 단어가 박스보다
    넓어도 그 줄이 넘치는 채로 그대로 반환한다 (가독성: 단어/조사가 잘리면 안 되므로,
    이 경우는 fit_text 쪽에서 폰트 크기를 더 줄여서 해결한다).
    allow_char_split=True일 때만(폰트를 최소 크기까지 줄여도 안 들어가는
    최후의 경우) 글자 단위로 쪼갠다.
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
    """줄바꿈 문자가 없는 한 덩어리를 박스 너비에 맞춰 자동 줄바꿈한다."""
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
    """박스에 들어가는 조합들 중 하나를 고른다. fits: [(줄수, 크기, font, lines, line_h)]

    줄 수가 적은 쪽에서 시작해, 줄을 하나 더 쓰는 대신 글자가 LINE_INCREASE_GAIN
    이상 커지는 경우에만 그쪽으로 갈아탄다.

    줄 수만 최소화하면 세로로 긴 말풍선에서 공간을 크게 낭비한다(실측: 133x160
    박스에서 2줄 14px가 뽑혀 세로의 22%만 사용. 같은 박스에 3줄 21px이 들어갔다).
    반대로 크기만 최대화하면 짧은 대사가 불필요하게 여러 줄로 쪼개진다.
    "줄을 더 쓴 만큼 실제로 글자가 커졌는가"를 조건으로 두어 양쪽을 모두 피한다.

    줄바꿈이 띄어쓰기 기준이라 공백 없는 짧은 대사('영차')는 애초에 한 줄로만
    나오므로, 이 규칙 때문에 잘게 쪼개질 일이 없다."""
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

    force_size가 주어지면(웹 편집 UI에서 사용자가 크기를 직접 지정한 경우) 자동
    조절을 건너뛰고 그 크기를 그대로 쓴다. 가로로는 넘치지 않게 줄바꿈하지만,
    세로로 넘치는 것은 사용자의 선택이므로 막지 않는다.

    1단계: max_size~min_size 범위 전체를 훑어(띄어쓰기 단위 줄바꿈만 허용,
           단어를 안 쪼갬) 박스 안에 들어가는 조합을 모은 뒤, 줄 수가 적은 쪽부터
           보면서 "줄을 하나 더 쓰는 대신 글자가 그만큼 커지는가"를 따져 고른다.
           (LINE_INCREASE_GAIN 참고)
    2단계: min_size까지 줄여도 박스에 안 들어가면(단어 하나가 너무 길거나
           텍스트가 너무 많음) - 경계를 넘지 않는 게 단어를 안 쪼개는 것보다
           우선이므로, 글자 단위 줄바꿈으로 전환해서 1px까지 계속 줄여서라도
           반드시 박스 안에 맞춘다.
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

    draw_paragraph_text()의 배치 계산과 완전히 같은 식을 쓰므로, 브라우저가 이
    좌표대로 canvas에 그리면 최종 결과와 사실상 동일한 그림이 된다.
    반환하는 좌표는 전부 "박스 왼쪽 위(0,0) 기준"의 상대 좌표다.

    - size      : 폰트 크기(px)
    - lines     : 줄바꿈된 줄 목록
    - line_h    : 줄 간격(px)
    - ascent    : 글자 윗선에서 베이스라인까지 거리 - canvas fillText의 y 계산에 필요
    - stroke    : 테두리 두께(0이면 없음)
    - line_x    : 각 줄의 펜 시작 x 좌표
    - block_y   : 첫 줄의 윗선 y 좌표
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
