"""
인식(recognition) 성능 채점 - 정답 대비 CER / 완전일치율을 시스템별로 낸다.

사용법:
    python score_recognition.py --labels <labels.csv> --base <폴더> --v4 <폴더> --v5 <폴더> --gemini <폴더>
"""

import os
import re
import csv
import sys
import json
import argparse
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402,F401

# 정규화에서 제거할 기호. 전각/반각은 NFKC가 알아서 통일하므로 여기 없어도 된다.
PUNCT_RE = re.compile(
    r"[\.…。、!\?\"“”~〜\-―─_"
    r"「」『』\(\)（）/\|:;,'・]"
)


def normalize(s):
    """전각/반각 통일 + 공백 제거. 공백은 문단을 이어붙일 때 생긴 것이라 모델 잘못이 아니다."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or ""))


def strip_punct(s):
    return PUNCT_RE.sub("", s)


def edit_distance(a, b):
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_labels(path):
    """정답 CSV를 읽는다. skip이 표시된 항목(글자가 아님/판독 불가)은 평가에서 뺀다."""
    refs = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("skip") or "").strip():
                continue
            text = (row.get("reference_ja") or "").strip()
            if text:
                refs[row["key"].strip()] = text
    return refs


def load_pages(json_dir):
    """{페이지: [문단, ...]}"""
    out = {}
    for name in sorted(os.listdir(json_dir)):
        if not name.endswith("_paragraphs.json") and not name.endswith("_gemini.json"):
            continue
        page = name.replace("_paragraphs.json", "").replace("_gemini.json", "")
        out[page] = json.load(open(os.path.join(json_dir, name), encoding="utf-8"))["paragraphs"]
    return out


def load_anchor(json_dir):
    """정답 키가 가리키는 위치. 정답 CSV를 만들 때 쓴 실행 결과를 기준으로 삼는다."""
    return {f"{page}#{p['id']}": (page, p["bbox"])
            for page, ps in load_pages(json_dir).items() for p in ps}


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0.0


def load_system(json_dir, field, anchor, min_iou=0.3):
    """{key: 인식결과}. 정답 위치와 가장 많이 겹치는 문단을 골라 짝짓는다."""
    pages = load_pages(json_dir)
    out = {}
    for key, (page, box) in anchor.items():
        cands = pages.get(page, [])
        best = max(cands, key=lambda p: iou(p["bbox"], box), default=None)
        out[key] = (best.get(field, "") or "") if best is not None and iou(best["bbox"], box) >= min_iou else ""
    return out


def score(refs, hyps, keys, drop_punct=False):
    """CER과 완전일치율을 계산한다. CER의 분모는 문단 수가 아니라 정답 전체 글자 수다."""
    dist = chars = exact = n = 0
    for k in keys:
        r = normalize(refs[k])
        h = normalize(hyps.get(k, ""))
        if drop_punct:
            r, h = strip_punct(r), strip_punct(h)
        if not r:
            continue
        dist += edit_distance(h, r)
        chars += len(r)
        exact += (h == r)
        n += 1
    return {
        "cer": dist / chars if chars else 0.0,
        "exact": exact / n if n else 0.0,
        "n": n,
        "chars": chars,
        "errors": dist,
    }


def interior_map(json_dir):
    """{key: 말풍선 안 여부}. 말풍선 안(대사)과 밖(효과음/간판)은 난이도가 달라 나눠서 본다."""
    return {f"{page}#{p['id']}": bool(p.get("bubble_bbox"))
            for page, ps in load_pages(json_dir).items() for p in ps}


def table(title, rows, systems):
    print(f"\n{title}")
    print(f"  {'시스템':<22}{'CER':>9}{'완전일치':>10}{'문단':>7}{'글자':>8}")
    print("  " + "-" * 56)
    for name in systems:
        r = rows[name]
        print(f"  {name:<22}{r['cer']*100:>8.1f}%{r['exact']*100:>9.1f}%"
              f"{r['n']:>7}{r['chars']:>8}")


def main():
    ap = argparse.ArgumentParser(description="인식 성능 채점 (CER / 완전일치율)")
    ap.add_argument("--labels", required=True, help="정답 CSV (key, reference_ja, skip)")
    ap.add_argument("--v4", required=True,
                    help="파인튜닝 1차(v4) 결과 폴더. 정답 키의 기준 위치로도 쓴다")
    ap.add_argument("--v5", default=None, help="파인튜닝 2차(v5) 결과 폴더")
    ap.add_argument("--base", required=True, help="PP-OCRv5 원본 결과 폴더")
    ap.add_argument("--gemini", required=True, help="Gemini 결과 폴더")
    ap.add_argument("--draft", default=None,
                    help="AI 초안 CSV. 주면 '초안 대비 사람이 얼마나 고쳤는지'도 함께 낸다.")
    ap.add_argument("--out", default=None, help="결과를 JSON으로도 저장할 경로")
    args = ap.parse_args()

    refs = load_labels(args.labels)
    anchor = load_anchor(args.v4)
    systems = {
        "PP-OCRv5 원본": load_system(args.base, "merged_text", anchor),
        "파인튜닝 1차(v4)": load_system(args.v4, "merged_text", anchor),
    }
    if args.v5:
        systems["파인튜닝 2차(v5)"] = load_system(args.v5, "merged_text", anchor)
    systems["Gemini"] = load_system(args.gemini, "gemini_text", anchor)

    keys = [k for k in refs if k in anchor]
    missing = len(refs) - len(keys)
    print(f"정답 {len(refs)}개 / 채점 대상 {len(keys)}개" + (f" (누락 {missing}개 제외)" if missing else ""))

    result = {"n_labeled": len(refs), "n_scored": len(keys), "sections": {}}

    for label, drop in [("전체 (구두점 포함)", False), ("전체 (구두점 제외)", True)]:
        rows = {n: score(refs, h, keys, drop) for n, h in systems.items()}
        table(label, rows, list(systems))
        result["sections"][label] = rows

    inter = interior_map(args.v4)
    for label, want in [("말풍선 안 (대사)", True), ("말풍선 밖 (효과음/간판)", False)]:
        sub = [k for k in keys if inter.get(k) is want]
        if not sub:
            continue
        rows = {n: score(refs, h, sub, False) for n, h in systems.items()}
        table(f"{label} - 구두점 포함", rows, list(systems))
        result["sections"][label] = rows

    if args.draft:
        draft = load_labels(args.draft)
        common = [k for k in keys if k in draft]
        s = score(refs, draft, common, False)
        print(f"\nAI 초안 vs 최종 정답 (사람이 얼마나 고쳤나)")
        print(f"  초안 CER {s['cer']*100:.1f}% / 그대로 통과 {s['exact']*100:.1f}% "
              f"({s['n']}개 중 {int(s['exact']*s['n'])}개)")
        print(f"  -> 이 수치가 높을수록 정답이 AI 초안 쪽으로 끌려갔을 위험이 크다.")
        result["draft_vs_final"] = s

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
