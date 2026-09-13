"""
CometKiwi로 번역 품질을 자동 채점한다.

CometKiwi(Unbabel/wmt22-cometkiwi-da)는 원문과 번역문만 보고 "사람이라면 몇 점을
줄까"를 예측하도록 학습된 모델이다(WMT 사람 평가 데이터로 회귀 학습). 

사용법:
    PYTHONNOUSERSITE=1 <comet env>/python.exe score_cometkiwi.py \
        --translations <translations.json> --out <결과 json>
"""

import os
import json
import argparse
from collections import defaultdict

MODEL = "Unbabel/wmt22-cometkiwi-da"


def main():
    ap = argparse.ArgumentParser(description="CometKiwi 자동 채점 (정답 번역 불필요)")
    ap.add_argument("--translations", required=True, help="run_translations.py 결과 json")
    ap.add_argument("--out", default=None, help="점수를 저장할 json 경로")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=0, help="0이면 CPU (comet 환경은 CPU 빌드)")
    args = ap.parse_args()

    from comet import download_model, load_from_checkpoint

    data = json.load(open(args.translations, encoding="utf-8"))
    sources, systems = data["sources"], data["systems"]

    print(f"모델 준비: {args.model}")
    model = load_from_checkpoint(download_model(args.model))

    # 전 시스템의 (원문, 번역) 쌍을 한 번에 넣고 뒤에서 시스템별로 나눈다.
    samples, index = [], []
    for name, trans in systems.items():
        for key, mt in trans.items():
            if key in sources and mt:
                samples.append({"src": sources[key], "mt": mt})
                index.append((name, key))

    print(f"채점 대상 {len(samples)}건 ({len(systems)}개 시스템)")
    out = model.predict(samples, batch_size=args.batch_size, gpus=args.gpus)
    scores = out["scores"] if isinstance(out, dict) else out.scores

    per_system, per_key = defaultdict(dict), defaultdict(dict)
    for (name, key), s in zip(index, scores):
        per_system[name][key] = float(s)
        per_key[key][name] = float(s)

    names = sorted(per_system, key=lambda n: -sum(per_system[n].values()) / len(per_system[n]))
    print(f"\n{'시스템':<22}{'평균':>9}{'문장':>7}")
    print("-" * 40)
    for n in names:
        v = list(per_system[n].values())
        print(f"{n:<22}{sum(v) / len(v):>9.4f}{len(v):>7}")

    # 같은 문장에서의 직접 비교. 평균은 몇 개의 극단값에 흔들리지만 승패는 덜하다.
    print(f"\n같은 문장 직접 비교 (승-패)")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            w = sum(1 for k, d in per_key.items()
                    if a in d and b in d and d[a] > d[b])
            l = sum(1 for k, d in per_key.items()
                    if a in d and b in d and d[a] < d[b])
            print(f"  {a} vs {b}:  {w}승 {l}패")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": args.model,
                       "mean": {n: sum(per_system[n].values()) / len(per_system[n])
                                for n in names},
                       "scores": {n: per_system[n] for n in names}},
                      f, ensure_ascii=False, indent=2)
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
