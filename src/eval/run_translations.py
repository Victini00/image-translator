"""
번역 성능 비교용 - 같은 일본어 원문을 시스템에 넣어 번역 결과를 모은다.

로컬 모델은 greedy decoding으로 고정한다(파이프라인 기본값은 temperature 0.1의
샘플링이라 돌릴 때마다 결과가 달라져 재현이 안 된다).

사용법:
    python run_translations.py --labels <labels.csv> --out <결과 json>
"""

import os
import re
import csv
import sys
import json
import time
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

GEMINI_SYSTEM_PROMPT = """당신은 일본 만화(manga)의 대사를 한국어로 번역합니다.

- 만화 대사의 어조와 뉘앙스를 살려 자연스러운 한국어 구어체로 옮기세요.
- 직역보다 상황에 맞는 표현을 우선하세요.
- 각 id는 독립된 대사이므로 서로 이어붙이거나 순서를 바꾸지 마세요.
- 번역문에 줄바꿈을 넣지 마세요.

반드시 지정된 JSON 스키마로만 응답하고, 모든 id를 빠짐없이 반환하세요."""


def load_sources(path):
    """정답 CSV에서 (key, 일본어 원문) 목록을 읽는다. skip 항목은 제외."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if (r.get("skip") or "").strip():
                continue
            text = (r.get("reference_ja") or "").strip()
            if text:
                rows.append((r["key"].strip(), text))
    return rows


def translate_local(sources, lora_path, device_str=None):
    """hell0ks 모델로 번역. lora_path가 빈 값이면 베이스 모델만 쓴다."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    device = torch.device(device_str) if device_str else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    name = config.TRANSLATION_MODEL
    print(f"  모델 로딩: {name}" + (f" + LoRA" if lora_path else " (베이스만)"))

    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if device.type == "cuda":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(device)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    out = {}
    for i, (key, ja) in enumerate(sources, 1):
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": ja}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(model.device)
        inputs.pop("token_type_ids", None)
        n_in = inputs["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.pad_token_id)
        out[key] = tokenizer.decode(gen[0][n_in:], skip_special_tokens=True).strip()
        if i % 10 == 0:
            print(f"    {i}/{len(sources)}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def translate_gemini(sources, model_name, chunk_size=25):
    """Gemini에 전체를 한 번의 요청으로 보낸다(무료 티어 요청 수 절약)."""
    from google import genai
    from google.genai import types
    from google.genai import errors
    from pydantic import BaseModel
    from dotenv import load_dotenv

    load_dotenv(config.ENV_PATH)

    class Item(BaseModel):
        id: int
        korean_translation: str

    class Batch(BaseModel):
        items: list[Item]

    client = genai.Client()
    cfg = types.GenerateContentConfig(
        system_instruction=GEMINI_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=Batch,
    )

    def call(contents):
        """무료 티어에서 429(요청 한도)와 503("high demand")이 자주 난다.

        503은 요청 크기와 무관하게 간헐적으로 발생한다(실측: 5개짜리 요청은 실패하고
        73개짜리가 통과하기도 함). 순전히 서버 혼잡 상태라 끈질기게 다시 걸면 통과하므로
        재시도 횟수를 넉넉히 둔다. 429는 응답이 알려주는 대기 시간을 따른다."""
        max_retries = 4
        for attempt in range(max_retries + 1):
            try:
                return client.models.generate_content(
                    model=model_name, contents=contents, config=cfg)
            except errors.APIError as e:
                if attempt >= max_retries:
                    raise
                if e.code == 429:
                    # 일일 한도(PerDay)는 기다린다고 풀리지 않는다. 재시도할수록
                    # 남은 할당량만 더 깎아먹으므로 바로 포기하고 알려준다.
                    if "PerDay" in str(e) or "per day" in str(e).lower():
                        raise RuntimeError(
                            "Gemini 무료 티어 일일 요청 한도를 모두 썼습니다. "
                            "할당량이 초기화된 뒤 다시 실행하세요."
                        ) from e
                    m = re.search(r"retry in ([\d.]+)s", e.message or "")
                    delay = float(m.group(1)) + 1.0 if m else 15.0
                elif e.code in (500, 502, 503, 504):
                    delay = min(3.0 * (2 ** attempt), 30.0) + random.uniform(0, 2)
                else:
                    raise
                print(f"    [경고] {e.code}, {delay:.0f}초 후 재시도 ({attempt + 1}/{max_retries})")
                time.sleep(delay)

    # 한 번에 전부 보내면 요청이 무거워 503이 잘 난다. 적당히 잘라서 보낸다.
    out = {}
    for s in range(0, len(sources), chunk_size):
        chunk = sources[s:s + chunk_size]
        idx = {i: key for i, (key, _) in enumerate(chunk)}
        listing = "\n".join(f"{i}: {ja}" for i, (_, ja) in enumerate(chunk))
        print(f"  {s + 1}~{s + len(chunk)} / {len(sources)}")
        resp = call([f"다음 일본어 만화 대사들을 한국어로 번역해 주세요.\n\n{listing}"])
        if resp.parsed is None:
            raise RuntimeError("Gemini 응답 파싱 실패")
        for it in resp.parsed.items:
            if it.id in idx:
                out[idx[it.id]] = it.korean_translation.strip()
        time.sleep(1.0)
    return out


def main():
    ap = argparse.ArgumentParser(description="번역 3종 실행 (base / LoRA / Gemini)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    ap.add_argument("--gemini-model", default=config.GEMINI_MODEL)
    ap.add_argument("--skip-local", action="store_true", help="로컬 모델 두 개를 건너뛴다")
    ap.add_argument("--chunk-size", type=int, default=25,
                    help="Gemini에 한 번에 보낼 문장 수. 무료 티어는 하루 요청 수가 적고 "
                         "재시도도 그 수에 포함되므로, 한도가 빠듯하면 크게 잡아 호출 수를 줄인다.")
    args = ap.parse_args()

    sources = load_sources(args.labels)
    print(f"원문 {len(sources)}개")

    result = {"sources": {k: v for k, v in sources}, "systems": {}}
    if os.path.exists(args.out):
        result = json.load(open(args.out, encoding="utf-8"))
        result.setdefault("systems", {})

    if not args.skip_local:
        for label, lora in [("hell0ks base", ""), ("hell0ks + LoRA", config.TRANSLATION_LORA_DIR)]:
            if label in result["systems"]:
                print(f"[건너뜀] {label} - 이미 있음")
                continue
            print(f"[실행] {label}")
            result["systems"][label] = translate_local(sources, lora, args.device)
            json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"  저장: {args.out}")

    if "Gemini" not in result["systems"]:
        print(f"[실행] Gemini ({args.gemini_model}) - API 호출 1회")
        result["systems"]["Gemini"] = translate_gemini(sources, args.gemini_model, args.chunk_size)
        json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n완료. 시스템별 결과 수:")
    for k, v in result["systems"].items():
        print(f"  {k:<18}{len(v)}개")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
