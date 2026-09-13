"""
LLM-as-judge용 블라인드 평가 세트를 만든다.

시스템 이름을 가리고 순서를 섞은 '문제지'와, 어느 슬롯이 어느 시스템인지 담은
'정답지'를 따로 저장한다. 채점하는 쪽은 문제지만 보고, 채점이 끝난 뒤에야
정답지로 되돌린다. 심판이 어느 것이 어느 모델인지 알면 평가가 성립하지 않는다.

사용법:
    python make_judge_set.py --translations <json> --out-quiz <문제지> --out-key <정답지>
"""

import os
import json
import random
import itertools
import argparse


def main():
    ap = argparse.ArgumentParser(description="LLM-as-judge 블라인드 세트 생성")
    ap.add_argument("--translations", required=True)
    ap.add_argument("--out-quiz", required=True, help="시스템 이름을 가린 문제지(json)")
    ap.add_argument("--out-key", required=True, help="슬롯->시스템 매핑(json)")
    ap.add_argument("--n", type=int, default=0, help="0이면 전체")
    ap.add_argument("--seed", type=int, default=20260909)
    args = ap.parse_args()

    data = json.load(open(args.translations, encoding="utf-8"))
    sources, systems = data["sources"], list(data["systems"])
    keys = [k for k in sources if all(k in data["systems"][s] for s in systems)]

    rng = random.Random(args.seed)
    rng.shuffle(keys)
    if args.n:
        keys = keys[:args.n]

    # 슬롯별로 특정 시스템이 몰리지 않도록 순열을 고르게 돌린다.
    perms = list(itertools.permutations(systems))
    schedule = [perms[i % len(perms)] for i in range(len(keys))]
    rng.shuffle(schedule)

    quiz, key = [], {}
    for i, k in enumerate(keys):
        order = schedule[i]
        quiz.append({
            "id": i,
            "source": sources[k],
            "candidates": {slot: data["systems"][s][k] for slot, s in zip("ABC", order)},
        })
        key[str(i)] = {"key": k, **{slot: s for slot, s in zip("ABC", order)}}

    for path, obj in [(args.out_quiz, quiz), (args.out_key, key)]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    print(f"문항 {len(quiz)}개 / 시스템 {len(systems)}개")
    print(f"문제지: {args.out_quiz}")
    print(f"정답지: {args.out_key}  (채점 끝날 때까지 열지 말 것)")


if __name__ == "__main__":
    main()
