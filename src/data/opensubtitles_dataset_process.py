"""
OpenSubtitles 일본어 자막에서 인식 학습용 문장을 뽑는다.

v4까지는 JParaCrawl(웹 크롤링 병렬 코퍼스)을 썼는데 두 가지 문제가 있었다.
  - 뉴스·광고·문서체가 대부분이라 만화 대사와 문체가 다르다.
  - 지금 남아 있는 JParaCrawl 파일은 이미 옛 필터를 거쳐서 「…」가 전부 사라진
    상태다(실측: 앞 20만 줄에 … 0개, 대신 치환된 'A' 3,615줄). 필터를 고쳐도
    원본이 없어 되살릴 수 없다.

자막은 영화·드라마의 실제 대사라 만화 말풍선과 문체가 가깝고, 한 줄 평균 12.8자로
말풍선 한 칸 분량과도 비슷하다.

말줄임표 표기를 만화 관행에 맞춰 「……」로 정규화한다. 자막은 「...」이나 「・・・」로
쓰는데(실측: 앞 30만 줄에서 ... 13,780줄 / ・・・ 2,252줄), 학습 라벨과 렌더링
이미지가 둘 다 「……」가 되어야 모델이 만화에서 보는 형태를 배운다.

사용법:
    python opensubtitles_dataset_process.py \
        --input data/raw/texts/ja-ko.txt_opensubtitles/OpenSubtitles.ja-ko.ja \
        --output data/processed/subtitles_sentences.txt
"""

import os
import re
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_special_letters import SPECIAL_PATTERN   # noqa: E402

# 문장 경계. 자막은 문장부호를 생략하는 경우가 많아 이것만으로는 안 끊기는 줄이 많고,
# 그래서 버퍼에 이어붙였다가 길이로 자르는 방식을 함께 쓴다.
SENTENCE_RE = re.compile(r"[^。！？!?\n]+[。！？!?]?")

# 말줄임표 표기 통일. 긴 것부터 바꿔야 「・・・・・・」가 두 번 잡히지 않는다.
ELLIPSIS_RULES = [
    ("・・・・・・", "……"),
    ("･･････", "……"),
    ("......", "……"),
    ("・・・", "……"),
    ("･･･", "……"),
    ("...", "……"),
    ("‥", "……"),
]

# 자막 특유의 표기. 화자 구분 하이픈(- )과 전각공백이 흔하다(앞 20만 줄에서 각각
# 14,976줄 / 13,524줄). 만화 말풍선에는 없는 것들이라 지운다.
SPEAKER_DASH_RE = re.compile(r"^[-–—]\s*")
TAG_RE = re.compile(r"<[^>]{1,20}>")
WRAPPED_RE = re.compile(r"^[（(\[【][^）)\]】]*[）)\]】]$")

# 자막 파일에 섞여 들어온 비(非)대사. 줄째로 버린다.
#   - ASS 서식 태그: {\bord3\1a&Hff&...}m 563 l 550 ... / {\a1\pos(113,268)} /
#     앞이 잘린 770)} 등. 태그 뒤가 벡터 그림 명령인 경우도 있어 태그만 지우면
#     숫자 쓰레기가 남는다(1,574줄).
#   - SRT 번호·타임스탬프: 539 00: / -1 00: 04: 12,205 (106줄).
#     「14,000」 같은 숫자 쉼표나 「12:30」 같은 시각은 정상 대사에도 있으므로,
#     콜론 뒤가 띄어져 있거나 줄이 콜론으로 끝나는 꼴(「00: 04」「00:」)만 잡는다.
ASS_TAG_RE = re.compile(r"[{}]")
TIMESTAMP_RE = re.compile(r"\d{2}:( \d{2}\b|\s*$)|-->")

# 영어 대사. 일본어 자막 파일에 번역 안 된 영어 줄이 섞여 있다. 만화 세로줄에는
# 긴 영문이 거의 없으므로 버린다. 「NSA」「OK!」 같은 짧은 약어는 만화에도
# 나오는 縦中横 표기라 남기도록, 영문자가 6자 이상이면서 절반을 넘을 때만 버린다.
ENGLISH_MIN_LETTERS = 6


def _is_english(s):
    letters = sum(c.isascii() and c.isalpha() for c in s)
    visible = sum(not c.isspace() for c in s)
    return letters >= ENGLISH_MIN_LETTERS and letters > visible / 2


# 공백. 자막은 대사 사이를 공백으로 띄우지만(「ああ そうか」) 만화 세로줄에는
# 빈칸이 없다 - 이어 쓰거나 줄을 바꾼다. 세로로 렌더링하면 공백이 한 칸 통째로
# 비어서 실제와 다른 그림이 된다(v5 첫 생성본에서 라벨의 92%). 영단어 사이
# 공백(「It's me」)만 남기고 일본어 글자에 붙은 공백은 지운다.
JA_SPACE_RE = re.compile(r"(?<=[^\x00-\x7F]) +| +(?=[^\x00-\x7F])")


def clean_line(line, stats=None):
    """자막 한 줄을 문장 추출에 쓸 수 있게 다듬는다. 버릴 줄이면 빈 문자열."""
    s = line.strip()
    if not s:
        return ""
    if ASS_TAG_RE.search(s) or TIMESTAMP_RE.search(s):
        if stats is not None:
            stats["junk"] += 1
        return ""
    s = TAG_RE.sub("", s)
    s = SPEAKER_DASH_RE.sub("", s)
    if WRAPPED_RE.match(s):        # (효과음) 같은 설명 줄
        return ""
    s = s.replace("　", " ")   # 전각공백 -> 보통공백
    for a, b in ELLIPSIS_RULES:
        s = s.replace(a, b)
    s = SPECIAL_PATTERN.sub("", s)          # 이모지·한글 등 학습셋 밖 문자 제거
    s = re.sub(r"\s+", " ", s).strip()
    if _is_english(s):
        if stats is not None:
            stats["english"] += 1
        return ""
    return JA_SPACE_RE.sub("", s)


def _join(buf, sent):
    """대사를 이어 붙인다. 영단어끼리 만날 때만 공백을 둔다(「NSA」+「OK」)."""
    if not buf:
        return sent
    if buf[-1].isascii() and buf[-1].isalnum() and sent[0].isascii() and sent[0].isalnum():
        return f"{buf} {sent}"
    return buf + sent


def extract(input_path, output_path, min_len, max_len, limit=None):
    """문장을 뽑아 max_len 글자씩 잘라 저장한다.

    짧은 대사는 버퍼에 이어붙였다가 자른다. 자막 한 줄이 max_len보다 짧은 경우가
    많은데, 그대로 쓰면 3~5자짜리 샘플만 잔뜩 생겨 학습이 한쪽으로 쏠린다.
    (v4가 쓰던 JParaCrawl_dataset_process.py와 같은 방식이되, 이어 붙일 때
    공백을 끼우지 않는다 - JA_SPACE_RE 설명 참고)
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    buf = ""
    written = with_ellipsis = 0
    stats = {"junk": 0, "english": 0}
    with open(input_path, encoding="utf-8", errors="replace") as f, \
         open(output_path, "w", encoding="utf-8") as out:
        for line in f:
            cleaned = clean_line(line, stats)
            if not cleaned:
                continue
            for sent in SENTENCE_RE.findall(cleaned):
                sent = sent.strip()
                if not sent:
                    continue
                buf = _join(buf, sent)
                while len(buf) >= min_len:
                    # 자른 조각의 끝이 공백이면 떼어낸다. 렌더링 후 여백을 잘라내면
                    # 그 공백은 그림에서 사라져 라벨과 어긋나기 때문이다.
                    piece = buf[:max_len].rstrip()
                    buf = buf[max_len:].strip()
                    # 일본어·영어가 섞인 줄(가사 등)은 자르고 나면 영어만 남은
                    # 조각이 생기므로 조각 단위로도 한 번 더 거른다.
                    if _is_english(piece):
                        stats["english"] += 1
                        continue
                    out.write(piece + "\n")
                    written += 1
                    with_ellipsis += ("…" in piece)
                    if limit and written >= limit:
                        _report(written, with_ellipsis, stats, output_path)
                        return
        if len(buf) >= min_len:
            out.write(buf[:max_len].rstrip() + "\n")
            written += 1
            with_ellipsis += ("…" in buf[:max_len])

    _report(written, with_ellipsis, stats, output_path)


def _report(written, with_ellipsis, stats, output_path):
    print(f"버린 줄: 서식 태그·타임스탬프 {stats['junk']:,} / 영어 대사 {stats['english']:,}")
    print(f"문장 {written:,}개 저장: {output_path}")
    if written:
        print(f"  「…」 포함: {with_ellipsis:,}개 ({with_ellipsis / written * 100:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description="OpenSubtitles 자막에서 학습용 문장 추출")
    ap.add_argument("--input", required=True, help="OpenSubtitles .ja 파일")
    ap.add_argument("--output", required=True)
    ap.add_argument("--min", type=int, default=12, dest="min_len")
    ap.add_argument("--max", type=int, default=12, dest="max_len",
                    help="한 샘플의 글자 수. v4와 같은 12를 기본으로 둔다 "
                         "(학습 설정의 max_text_length 20보다 짧게).")
    ap.add_argument("--limit", type=int, default=None, help="이만큼 뽑고 중단")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"에러: 입력 파일을 찾을 수 없음: {args.input}")
        sys.exit(1)
    extract(args.input, args.output, args.min_len, args.max_len, args.limit)


if __name__ == "__main__":
    main()
