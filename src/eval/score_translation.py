"""
번역 품질 블라인드 평가 결과를 집계한다.

make_mqm_set.py가 만든 페이지에서 내보낸 CSV(key, system, score, tags)를 읽어
시스템별 평균 점수, 점수 분포, 오류 유형 분포, 시스템 간 승패를 낸다.

사용법:
    python score_translation.py --mqm <mqm.csv> [--translations <translations.json>]
"""

import os
import csv
import sys
import json
import argparse
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402,F401

TAG_LABEL = {
    "mistrans": "오역",
    "omission": "누락",
    "tone": "어조",
    "awkward": "어색",
}


def load(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            score = (r.get("score") or "").strip()
            if not score:
                continue
            rows.append({
                "key": r["key"].strip(),
                "system": r["system"].strip(),
                "score": int(score),
                "tags": [t for t in (r.get("tags") or "").split("|") if t],
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description="번역 블라인드 평가 집계")
    ap.add_argument("--mqm", required=True, help="평가 페이지에서 내보낸 CSV")
    ap.add_argument("--translations", default=None,
                    help="번역 결과 json. 주면 점수가 갈린 예시를 함께 보여준다.")
    ap.add_argument("--out", default=None, help="집계 결과를 저장할 json 경로")
    args = ap.parse_args()

    rows = load(args.mqm)
    systems = sorted({r["system"] for r in rows})
    by_sys = defaultdict(list)
    tags_by_sys = defaultdict(Counter)
    scores = defaultdict(dict)          # key -> system -> score

    for r in rows:
        by_sys[r["system"]].append(r["score"])
        scores[r["key"]][r["system"]] = r["score"]
        for t in r["tags"]:
            tags_by_sys[r["system"]][t] += 1

    n_items = len(scores)
    print(f"평가 항목 {n_items}개 / 판정 {len(rows)}회 / 시스템 {len(systems)}개")

    print(f"\n{'시스템':<20}{'평균':>7}{'5':>5}{'4':>5}{'3':>5}{'2':>5}{'1':>5}")
    print("-" * 52)
    for s in systems:
        v = by_sys[s]
        dist = Counter(v)
        avg = sum(v) / len(v) if v else 0
        print(f"{s:<20}{avg:>7.2f}" + "".join(f"{dist.get(k, 0):>5}" for k in (5, 4, 3, 2, 1)))

    print(f"\n오류 유형 (3점 이하에 표시한 것)")
    print(f"{'시스템':<20}" + "".join(f"{TAG_LABEL[t]:>7}" for t in TAG_LABEL) + f"{'합계':>7}")
    print("-" * 52)
    for s in systems:
        c = tags_by_sys[s]
        print(f"{s:<20}" + "".join(f"{c.get(t, 0):>7}" for t in TAG_LABEL) + f"{sum(c.values()):>7}")

    # 같은 문장에서의 직접 비교. 평균은 몇 개의 극단값에 흔들리지만 승패는 덜하다.
    print(f"\n같은 문장 직접 비교 (승-무-패)")
    for i, a in enumerate(systems):
        for b in systems[i + 1:]:
            w = l = t = 0
            for k, d in scores.items():
                if a in d and b in d:
                    if d[a] > d[b]:
                        w += 1
                    elif d[a] < d[b]:
                        l += 1
                    else:
                        t += 1
            print(f"  {a} vs {b}:  {w}승 {t}무 {l}패")

    result = {
        "n_items": n_items,
        "systems": {s: {"mean": sum(by_sys[s]) / len(by_sys[s]),
                        "dist": dict(Counter(by_sys[s])),
                        "tags": dict(tags_by_sys[s])} for s in systems},
    }

    if args.translations:
        data = json.load(open(args.translations, encoding="utf-8"))
        gaps = sorted(scores.items(),
                      key=lambda kv: -(max(kv[1].values()) - min(kv[1].values())))[:5]
        print(f"\n점수가 가장 많이 갈린 문장")
        for k, d in gaps:
            if max(d.values()) == min(d.values()):
                break
            print(f"\n  원문: {data['sources'].get(k, '?')}")
            for s in systems:
                if s in d:
                    print(f"    [{d[s]}] {s:<18}{data['systems'][s].get(k, '')}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n집계 저장: {args.out}")


if __name__ == "__main__":
    main()
