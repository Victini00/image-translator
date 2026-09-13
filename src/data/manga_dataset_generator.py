"""
인식 모델 학습용 이미지-라벨 쌍 생성기

핵심은 세로쓰기 렌더링이다 - TRDG는 가로 글리프를 세로로 쌓기만 해서 「ー」가 가로 막대,
「……」가 가로 3점 두 줄로 그려졌고, 그 결과 모델이 세로 조판 기호를 배우지 못했다.
여기서는 폰트에 내장된 세로 글리프를 HarfBuzz로 꺼내 쓴다.

라벨 파일 옆에 {train|val}_meta.txt를 함께 남긴다(이미지, 세로/가로, 폰트, 배경).
학습에는 쓰지 않고, 데이터 구성을 정확히 세거나 특정 조건의 샘플을 찾을 때 쓴다.

사용법:
    python manga_dataset_generator.py --text <문장파일> --out <출력폴더> --count 1000000
"""

import os
import sys
import random
import argparse
import multiprocessing as mp

import freetype
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vertical_text import VerticalRenderer          # noqa: E402
import image_effects as fx                          # noqa: E402

CHARS_PER_SAMPLE = 12          # v4와 동일. 학습 설정의 max_text_length(20)보다 짧게 둔다.
VERTICAL_RATIO = 0.85          # 만화 대사는 대부분 세로쓰기. 나머지는 간판·효과음 등 가로.

# 배경 비율. 평가셋 실측으로 검출된 문단의 95%가 흰 말풍선 안이었다.
# 실제 만화 배경을 오려 붙이는 방식도 시도했으나, 합성 티가 나고 v4도 이런 학습
# 없이 배경 있는 글자를 잘 읽었기에 넣지 않는다.
BG_WHITE, BG_TONE = 0.80, 0.20

# 열화 적용 확률. v4의 실제 오류에 블러·노이즈로 설명되는 것이 없었으므로 낮게 둔다.
# 100만 장 기준이면 20%만 걸려도 20만 장이라 절대량은 충분하다.
P_JPEG, P_DOWNSCALE, P_BLUR, P_SKEW, P_NOISE, P_GRAY = 0.40, 0.25, 0.20, 0.15, 0.10, 0.15

FONT_SIZE_RANGE = (28, 52)


def load_vocab(dict_path):
    """학습에 쓸 문자 사전. 공백은 설정(use_space_char)에서 따로 허용되므로 추가한다."""
    vocab = {l.rstrip("\r\n") for l in open(dict_path, encoding="utf-8")}
    vocab.add(" ")
    return vocab


def load_chunks(path, vocab=None, limit=None):
    """문장 파일을 CHARS_PER_SAMPLE 글자 단위로 잘라 목록으로 만든다.

    사전에 없는 글자가 든 조각은 버린다. PaddleOCR은 라벨을 인코딩할 때 사전 밖
    글자를 조용히 지우는데, 이미지에는 그 글자가 그려져 있으므로 그림과 라벨이
    어긋난 샘플이 된다. 자막 코퍼스 기준 0.43%라 버려도 손실이 거의 없다.
    """
    chunks, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            for i in range(0, len(s), CHARS_PER_SAMPLE):
                # 앞뒤 공백은 떼어낸다. 렌더링 후 여백을 잘라내면 그림에서는
                # 사라지는 공백이라, 남겨두면 라벨에만 있는 글자가 된다.
                piece = s[i:i + CHARS_PER_SAMPLE].strip()
                if not piece:
                    continue
                if vocab is not None and any(c not in vocab for c in piece):
                    dropped += 1
                    continue
                chunks.append(piece)
                if limit and len(chunks) >= limit:
                    return chunks, dropped
    return chunks, dropped


def font_coverage(texts, font_paths):
    """글자마다, 그 글자를 가진 폰트 번호들을 비트마스크로 돌려준다.

    폰트에 없는 글자는 빈칸이나 네모 박스(.notdef)로 그려져 라벨과 그림이
    어긋난다. 실측: Zen 계열 3종에 「～」「－」「―」가 없고(빈칸), ShipporiMincho에
    반각 가타카나가 없다(☒). 폰트를 무작위로 고르면 이미지의 약 0.3%가 이렇게 됐다.
    """
    chars = {c for t in texts for c in t if not c.isspace()}
    faces = [freetype.Face(p) for p in font_paths]
    return {c: sum(1 << i for i, f in enumerate(faces) if f.get_char_index(ord(c)))
            for c in chars}


def usable_fonts(text, cover, n_fonts):
    """text의 글자를 전부 가진 폰트 번호 목록."""
    mask = (1 << n_fonts) - 1
    for c in text:
        if not c.isspace():
            mask &= cover.get(c, 0)
    return [i for i in range(n_fonts) if mask >> i & 1]


def make_sample(text, font_paths, renderers, cover, rng):
    """문장 하나를 이미지로 만든다. (이미지, 메타정보)를 돌려준다."""
    candidates = usable_fonts(text, cover, len(font_paths))
    font_path = font_paths[candidates[rng.integers(len(candidates))]]
    size = int(rng.integers(*FONT_SIZE_RANGE))
    vr = renderers[font_path]
    vr.set_size(size)

    vertical = rng.random() < VERTICAL_RATIO
    if vertical:
        # 세로로 그린 뒤 반시계 90도로 눕힌다. 인식기가 받는 형태가 그것이다.
        img = vr.render(text).rotate(90, expand=True)
    else:
        img = vr.render_horizontal(text)

    # ---- 배경 ----
    tone = rng.random() < BG_TONE
    if tone:
        img = fx.screentone(img, rng)

    # ---- 열화 ----
    if rng.random() < P_GRAY:
        img = fx.text_gray(img, rng)
    if rng.random() < P_SKEW:
        img = fx.skew(img, rng, max_deg=2.0)
    if rng.random() < P_BLUR:
        img = fx.blur(img, rng, 0.3, 0.8)
    if rng.random() < P_DOWNSCALE:
        img = fx.downscale(img, rng, 0.55, 1.0)
    if rng.random() < P_NOISE:
        img = fx.noise(img, rng, sigma=float(rng.uniform(2, 6)))
    if rng.random() < P_JPEG:
        img = fx.jpeg_artifacts(img, rng, 60, 92)

    meta = ("vertical" if vertical else "horizontal",
            os.path.splitext(os.path.basename(font_path))[0],
            "tone" if tone else "white")
    return img, meta


def worker(args):
    """프로세스 하나가 맡은 구간을 생성한다."""
    (start, texts, out_dir, prefix, font_paths, cover, seed, height) = args
    rng = np.random.default_rng(seed)
    renderers = {p: VerticalRenderer(p, 40) for p in font_paths}

    img_dir = os.path.join(out_dir, "images")
    rows = []
    for i, text in enumerate(texts):
        idx = start + i
        try:
            img, meta = make_sample(text, font_paths, renderers, cover, rng)
        except Exception:
            continue
        # 학습 입력 높이에 맞춰 비율 유지 리사이즈
        if img.height != height:
            w = max(1, int(img.width * height / img.height))
            img = img.resize((w, height), Image.LANCZOS)
        name = f"{prefix}_{idx:06d}.jpg"
        img.convert("RGB").save(os.path.join(img_dir, name), "JPEG", quality=92)
        rows.append((f"images/{name}\t{text}", f"images/{name}\t" + "\t".join(meta)))
    return rows


def generate(texts, out_dir, prefix, font_paths, cover, workers, seed, height):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    chunk = max(1, len(texts) // max(1, workers))
    jobs = []
    for w in range(workers):
        s = w * chunk
        e = len(texts) if w == workers - 1 else min(len(texts), s + chunk)
        if s >= e:
            break
        jobs.append((s, texts[s:e], out_dir, prefix, font_paths, cover, seed + w, height))

    rows = []
    if workers == 1:
        for j in jobs:
            rows += worker(j)
    else:
        with mp.Pool(len(jobs)) as pool:
            for part in pool.imap_unordered(worker, jobs):
                rows += part
                print(f"  {len(rows):,} / {len(texts):,}", flush=True)
    rows.sort()
    return [r[0] for r in rows], [r[1] for r in rows]


def main():
    ap = argparse.ArgumentParser(description="만화용 인식 학습 데이터 생성")
    ap.add_argument("--text", required=True, help="필터를 거친 일본어 문장 파일")
    ap.add_argument("--out", required=True, help="출력 폴더")
    ap.add_argument("--fonts", default="data/fonts/ja", help="폰트 폴더(.ttf)")
    ap.add_argument("--dict", default="data/processed/fine_tuning_answer_sheet_JParaCrawl_million/ppocrv5_dict.txt",
                    help="학습 설정의 character_dict_path와 같은 파일. 여기 없는 글자가 든 문장은 버린다.")
    ap.add_argument("--count", type=int, default=1000000, help="학습 샘플 수")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--eval-size", type=int, default=20000,
                    help="학습 중 평가용으로 검증셋에서 떼어낼 개수(val_eval.txt). 0이면 만들지 않음")
    ap.add_argument("--height", type=int, default=32, help="저장 높이(v4와 동일하게 32)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--seed", type=int, default=20260911)
    args = ap.parse_args()

    font_paths = [os.path.join(args.fonts, n) for n in sorted(os.listdir(args.fonts))
                  if n.lower().endswith((".ttf", ".otf"))]
    if not font_paths:
        print(f"에러: 폰트를 찾을 수 없음: {args.fonts}")
        sys.exit(1)

    want_val = int(args.count * args.val_ratio)
    vocab = load_vocab(args.dict) if args.dict else None
    chunks, dropped = load_chunks(args.text, vocab, limit=args.count + want_val)
    if dropped:
        print(f"사전 밖 글자가 든 조각 {dropped:,}개 제외")

    # 어느 폰트로도 온전히 그릴 수 없는 조각은 미리 뺀다. 생성 도중에 건너뛰면
    # 이미지 번호에 구멍이 생긴다.
    cover = font_coverage(chunks, font_paths)
    before = len(chunks)
    chunks = [c for c in chunks if usable_fonts(c, cover, len(font_paths))]
    if before - len(chunks):
        print(f"모든 폰트에 없는 글자가 든 조각 {before - len(chunks):,}개 제외")
    random.Random(args.seed).shuffle(chunks)

    # 문장이 요청량보다 적을 수 있다. 학습셋(--count)을 먼저 채우고 남는 것을
    # 검증셋으로 쓴다 - 학습 중 평가는 val_eval(2만 장)만 쓰므로 검증셋이 조금
    # 줄어도 괜찮다. 학습셋조차 못 채우면 비율대로 나눈다.
    if len(chunks) >= args.count + args.eval_size:
        n_val = min(want_val, len(chunks) - args.count)
    else:
        n_val = int(len(chunks) * args.val_ratio)
    train_texts, val_texts = chunks[n_val:], chunks[:n_val]
    if len(chunks) < args.count + want_val:
        print(f"[알림] 문장이 부족해 {len(chunks):,}개로 진행합니다 "
              f"(요청 {args.count + want_val:,})")

    print(f"폰트 {len(font_paths)}종 / 학습 {len(train_texts):,} / 검증 {len(val_texts):,}")
    print(f"세로 {VERTICAL_RATIO:.0%} · 가로 {1 - VERTICAL_RATIO:.0%}  |  프로세스 {args.workers}개")

    os.makedirs(args.out, exist_ok=True)
    for name, texts in [("train", train_texts), ("val", val_texts)]:
        print(f"[{name}] 생성 중...")
        labels, metas = generate(texts, args.out, name, font_paths, cover,
                                 args.workers, args.seed, args.height)
        with open(os.path.join(args.out, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(labels) + "\n")
        with open(os.path.join(args.out, f"{name}_meta.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(metas) + "\n")
        n_vert = sum(m.split("\t")[1] == "vertical" for m in metas)
        n_tone = sum(m.split("\t")[3] == "tone" for m in metas)
        print(f"[{name}] {len(labels):,}장 완료 | 세로 {n_vert:,} · 가로 {len(metas) - n_vert:,}"
              f" | 스크린톤 {n_tone:,}")

        # 학습 중 평가는 검증셋 일부로 충분하다. v4는 20만 장 전체를 5천 스텝마다
        # 돌려서 평가가 학습 시간의 37%를 먹었다. 문장을 이미 섞어 두었으므로
        # 번호 순 앞부분이 곧 무작위 표본이다.
        if name == "val" and 0 < args.eval_size < len(labels):
            with open(os.path.join(args.out, "val_eval.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(labels[:args.eval_size]) + "\n")
            print(f"[val_eval] {args.eval_size:,}장 (학습 중 평가용)")

    print(f"\n완료: {args.out}")


if __name__ == "__main__":
    main()
