"""
일본어 세로쓰기 렌더링 - 폰트에 내장된 세로 글리프를 사용한다.

세로 조판에서는 「…」「ー」「〜」가 90도 누운 모양이 되고, 「、」「。」는 글자칸
오른쪽 위로 올라가며, 작은 가나(「ゃ」「ッ」)도 위치가 달라진다. 이것들은 회전으로
흉내낼 수 있는 게 아니라 폰트가 별도 글리프로 갖고 있고(OpenType `vert` 피처),
세로 진행 간격도 별도 테이블(`vhea`/`vmtx`)에 들어 있다.

HarfBuzz로 세로 방향 셰이핑을 하면 그 치환과 간격이 한꺼번에 적용된다.
Pillow만으로는 안 되는데, 이 환경의 Pillow에 raqm/harfbuzz가 빌드돼 있지 않아
features 인자를 쓸 수 없기 때문이다.

TRDG의 orientation=1은 가로 글리프를 그대로 세로로 쌓기만 해서 실제 만화와 다른
그림이 나온다(「ー」가 가로 막대, 「……」가 가로 3점 두 줄). 그래서 학습 데이터를
그 방식으로 만들면 세로 조판 기호를 배우지 못한다.
"""

import freetype
import numpy as np
import uharfbuzz as hb
from PIL import Image


class VerticalRenderer:
    """폰트 하나를 열어두고 세로쓰기 텍스트를 이미지로 그린다."""

    def __init__(self, font_path, font_size=48):
        self.font_path = font_path
        self.font_size = font_size

        with open(font_path, "rb") as f:
            self._blob = hb.Blob(f.read())
        self._face = hb.Face(self._blob)
        self._hb_font = hb.Font(self._face)
        self._upem = self._face.upem
        self._hb_font.scale = (self._upem, self._upem)

        self._ft = freetype.Face(font_path)
        self._ft.set_pixel_sizes(0, font_size)
        self._cell = None          # 세로쓰기 한 칸 높이(em 대비 비율), 최초 계산 후 재사용

    def set_size(self, font_size):
        """글자 크기를 바꾼다. 폰트를 다시 열지 않아 샘플마다 호출해도 싸다."""
        if font_size != self.font_size:
            self.font_size = font_size
            self._ft.set_pixel_sizes(0, font_size)

    def render_horizontal(self, text, padding=8, bg=255, fg=40):
        """가로쓰기. 세로 글리프 치환 없이 기본 글리프를 왼쪽에서 오른쪽으로 배치한다.

        만화에도 간판·표지처럼 가로로 조판된 글자가 나온다(평가셋의 NICU, NOTICE).
        """
        buf = hb.Buffer()
        buf.add_str(text)
        buf.direction = "ltr"
        buf.script = "Hani"
        buf.language = "ja"
        hb.shape(self._hb_font, buf)

        scale = self.font_size / self._upem
        pad = int(self.font_size)
        w = int(sum(p.x_advance * scale for p in buf.glyph_positions)) + pad * 2
        h = int(self.font_size * 2) + pad * 2
        canvas = np.full((h, w), bg, dtype=np.uint8)

        pen_x = float(pad)
        baseline = pad + self.font_size
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            self._ft.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
            g = self._ft.glyph
            bmp = g.bitmap
            if bmp.width and bmp.rows:
                arr = np.array(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.pitch)[:, :bmp.width]
                x0 = int(round(pen_x + g.bitmap_left + pos.x_offset * scale))
                y0 = int(round(baseline - g.bitmap_top - pos.y_offset * scale))
                x0 = max(0, min(w - bmp.width, x0))
                y0 = max(0, min(h - bmp.rows, y0))
                region = canvas[y0:y0 + bmp.rows, x0:x0 + bmp.width]
                sub = arr[:region.shape[0], :region.shape[1]].astype(np.int16)
                region[:] = np.minimum(region, (bg - sub * (bg - fg) // 255).astype(np.uint8))
            pen_x += pos.x_advance * scale

        return self._trim(Image.fromarray(canvas, mode="L").convert("RGB"), padding)

    @staticmethod
    def _is_upright_in_vertical(ch):
        """세로쓰기에서 눕히지 않고 한 글자씩 세워 쌓는 문자인지.

        일본어 조판에서 긴 영문은 90도 눕혀 쓰지만, 짧은 약어·숫자는 글자마다
        세워서 쌓는다(縦中横). 만화 간판·전문용어가 대체로 후자다 - 평가셋의
        「NICU」도 N/I/C/U가 각각 똑바로 선 채 세로로 쌓여 있다.
        HarfBuzz의 ttb 셰이핑은 라틴을 눕히므로, 이 문자들은 따로 처리한다.
        """
        return ch.isascii() and (ch.isalnum() or ch in "!?#&%")

    def _shape_upright(self, text):
        """라틴 구간: 글자마다 가로 글리프를 그대로 쓰고 세로 간격만 준다.

        칸 높이는 일본어 글자의 세로 진행량(vhea의 advanceHeight)과 같게 맞춘다.
        임의로 font_size를 쓰면 라틴 구간과 일본어 구간의 칸 크기가 어긋나,
        경계에서 글자가 겹쳐 사라진다.
        """
        buf = hb.Buffer()
        buf.add_str(text)
        buf.direction = "ltr"
        buf.script = "Latn"
        hb.shape(self._hb_font, buf)
        return [(i.codepoint, 0.0, 0.0, self._cell_height(), True)
                for i in buf.glyph_infos]

    def _cell_height(self):
        """세로쓰기 한 칸의 높이. 일본어 글자의 세로 진행량을 기준으로 삼는다."""
        if self._cell is None:
            buf = hb.Buffer()
            buf.add_str("あ")
            buf.direction = "ttb"
            buf.script = "Hani"
            buf.language = "ja"
            hb.shape(self._hb_font, buf, {"vert": True, "vrt2": True})
            adv = -buf.glyph_positions[0].y_advance / self._upem
            self._cell = adv
        return self._cell * self.font_size

    def _shape_ttb(self, text):
        """일본어 구간: 세로 전용 글리프로 치환하고 세로 간격을 적용한다."""
        buf = hb.Buffer()
        buf.add_str(text)
        buf.direction = "ttb"
        buf.script = "Hani"
        buf.language = "ja"
        # vert: 세로 전용 글리프로 치환. HarfBuzz가 ttb에서 기본 적용하지만
        # 폰트에 따라 vrt2만 있는 경우가 있어 명시해 둔다.
        hb.shape(self._hb_font, buf, {"vert": True, "vrt2": True})

        scale = self.font_size / self._upem
        return [(i.codepoint,
                 p.x_offset * scale,
                 p.y_offset * scale,
                 -p.y_advance * scale,   # ttb에서 y_advance는 음수로 나온다
                 False)
                for i, p in zip(buf.glyph_infos, buf.glyph_positions)]

    def shape(self, text):
        """(글리프id, x오프셋, y오프셋, 진행량, 세워쓰기여부) 목록을 돌려준다.

        라틴 구간과 일본어 구간을 나눠서 셰이핑한다. 한 번에 ttb로 넘기면 HarfBuzz가
        라틴까지 눕혀버려서, 회전 후에 영문만 혼자 똑바로 서는 그림이 된다.
        """
        out, buf_chars, buf_upright = [], [], None
        for ch in text:
            up = self._is_upright_in_vertical(ch)
            if buf_upright is None:
                buf_upright = up
            if up != buf_upright:
                out += (self._shape_upright("".join(buf_chars)) if buf_upright
                        else self._shape_ttb("".join(buf_chars)))
                buf_chars, buf_upright = [], up
            buf_chars.append(ch)
        if buf_chars:
            out += (self._shape_upright("".join(buf_chars)) if buf_upright
                    else self._shape_ttb("".join(buf_chars)))
        return out

    def render(self, text, padding=8, bg=255, fg=40):
        """세로쓰기 이미지를 만든다(흰 배경, 검은 글자).

        글자 위치는 세로 베어링(vertBearingX/Y)으로 잡는다. 가로쓰기 베어링으로
        놓으면 글자가 겹치고 첫 글자가 잘린다 - 세로 조판에서는 글자의 기준점이
        왼쪽 베이스라인이 아니라 글자칸 위쪽 중앙이기 때문이다.
        """
        glyphs = self.shape(text)
        if not glyphs:
            return Image.new("RGB", (1, 1), (bg, bg, bg))

        # 여유 있게 캔버스를 잡고 마지막에 실제 그린 범위로 자른다.
        pad = int(self.font_size)
        w = int(self.font_size * 2) + pad * 2
        h = int(sum(g[3] for g in glyphs)) + pad * 2
        canvas = np.full((h, w), bg, dtype=np.uint8)

        pen_x = w / 2.0          # 세로 진행의 중심선
        pen_y = float(pad)
        for gid, dx, dy, adv, upright in glyphs:
            self._ft.load_glyph(gid, freetype.FT_LOAD_RENDER)
            g = self._ft.glyph
            bmp = g.bitmap
            if bmp.width and bmp.rows:
                arr = np.array(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.pitch)[:, :bmp.width]
                # HarfBuzz가 ttb에서 돌려주는 y_offset에는 세로 원점 보정이 이미
                # 들어 있다. 여기에 FreeType의 vertBearingY까지 더하면 이중 적용이
                # 되어, 라틴 구간 다음의 첫 일본어 글자가 위로 끌려 올라가 앞 글자를
                # 덮어버린다. 그래서 표준 공식(펜 위치 - y_offset - bitmap_top)만 쓴다.
                # 여기에 칸 높이(adv)를 더하면 일본어 글자만 한 칸 아래로 밀려,
                # 제 칸에 그려지는 세워쓰기 글자와 경계에서 겹친다(「ためCIA」의
                # め·C, 「千500」의 千·5). 일본어만 있는 줄은 전부 같이 밀리고
                # 위쪽 여백이 잘려나가 티가 안 나므로 경계 문자열로 확인해야 한다.
                if upright:
                    # 세워쓰기(라틴 약어 등)는 셰이핑에 세로 정보가 없으므로 칸 중앙에 놓는다.
                    x0 = int(round(pen_x - bmp.width / 2 + dx))
                    y0 = int(round(pen_y + (adv - bmp.rows) / 2 + dy))
                else:
                    x0 = int(round(pen_x + dx + g.bitmap_left))
                    y0 = int(round(pen_y - dy - g.bitmap_top))
                x0 = max(0, min(w - bmp.width, x0))
                y0 = max(0, min(h - bmp.rows, y0))
                region = canvas[y0:y0 + bmp.rows, x0:x0 + bmp.width]
                sub = arr[:region.shape[0], :region.shape[1]].astype(np.int16)
                painted = (bg - sub * (bg - fg) // 255).astype(np.uint8)
                region[:] = np.minimum(region, painted)
            pen_y += adv

        img = Image.fromarray(canvas, mode="L").convert("RGB")
        return self._trim(img, padding)

    @staticmethod
    def _trim(img, padding):
        """그려진 영역에 맞춰 여백을 정리한다."""
        a = np.array(img.convert("L"))
        ys, xs = np.where(a < 250)
        if len(xs) == 0:
            return img
        x0, x1 = max(0, xs.min() - padding), min(a.shape[1], xs.max() + padding + 1)
        y0, y1 = max(0, ys.min() - padding), min(a.shape[0], ys.max() + padding + 1)
        return img.crop((x0, y0, x1, y1))
