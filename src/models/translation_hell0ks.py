import argparse
import datetime
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Windows 콘솔(cp949)이 일부 한자를 못 그려서 출력이 깨지는 것을 방지
sys.stdout.reconfigure(encoding="utf-8")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


DEFAULT_MODEL = "hell0ks/ja-ko-vn-7b-v1"
DEFAULT_LORA = "./../../models/translation/hell0ks_ja-ko-vn-7b-v1/lora/v1"
DEFAULT_COMPARE_OUTPUT_DIR = "./../../output/translation/text/hell0ks"

# 0100의 몇개 문장들(test 데이터임) - (일본어, 정답 한국어)
TEST_SET = [
    ("話の途中だったでしょ。来週の月曜から、いよいよ期末テスト！", "얘기하다 말았잖아. 다음 주 월요일부터 드디어 기말고사야!"),
    ("勉強もちゃんとしてますっ！きちんと計画立てて、ぬかりなく！！", "공부도 제대로 하고 있어요! 철저하게 계획을 세워서, 빈틈없이요!!"),
    ("あ…ありがと、助かる！お茶碗一杯分をラップして、置いといて。あら熱取れてから冷凍庫に入れるからっ", "아... 고마워, 살았다! 밥 한 공기 분량씩 랩으로 싸서 놔둬. 한 김 식으면 냉동실에 넣을 테니까"),
    ("その…ちょっと、気になって。今晩冷えるから", "그게... 조금 신경 쓰여서요. 오늘 밤은 추우니까"),
    ("それなのに、うぅ、風邪引いたくらいでっ…、テスト勉強をおろそかになんてできないわよっ！", "그런데도, 으으, 감기 좀 걸렸다고... 시험공부를 소홀히 할 순 없어!"),
]


def translate(model, tokenizer, text, num_return_sequences=1):
    # 모델이 요구하는 ChatML 채팅 템플릿(<|im_start|>user ... <|im_end|>)을 적용한다.
    # 태그 없이 원문만 넣으면 "번역하라"는 지시를 인식하지 못하고 그냥 다음 문장을
    # 이어 쓰는 completion처럼 동작해버린다.
    messages = [{"role": "user", "content": text}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    input_len = inputs["input_ids"].shape[1]

    print(f"  generating...", flush=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(f"  done.", flush=True)

    # 입력 토큰 제외하고 생성된 부분만 디코딩
    results = [
        tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in outputs
    ]
    return results


def load_model(model_name, lora_path, device):
    if device.type == "cuda":
        # 7B 모델이 12GB급 VRAM에 fp16 그대로는 안 들어가서 4bit로 로드한다
        # (LoRA 학습 때와 동일한 양자화 설정).
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device)

    if lora_path:
        print(f"LoRA 어댑터 적용: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return model


def run_test_set(model, tokenizer):
    results = []
    for ja, _ in TEST_SET:
        results.append(translate(model, tokenizer, ja)[0])
    return results


def run_compare(model_name, lora_path, tokenizer, device, output_dir):
    print("[1/2] base 모델(LoRA 없음) 번역 중...")
    base_model = load_model(model_name, lora_path="", device=device)
    base_results = run_test_set(base_model, tokenizer)
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[2/2] LoRA 적용 모델 번역 중...")
    lora_model = load_model(model_name, lora_path=lora_path, device=device)
    lora_results = run_test_set(lora_model, tokenizer)
    del lora_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"compare_{timestamp}.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("hell0ks 번역 비교 (정답 / base / LoRA)\n")
        f.write(f"생성 시각: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"LoRA 경로: {lora_path}\n")
        f.write("=" * 60 + "\n")
        for idx, (ja, ko_ref) in enumerate(TEST_SET, 1):
            f.write(f"\n[{idx}]\n")
            f.write(f"JP  : {ja}\n")
            f.write(f"정답 : {ko_ref}\n")
            f.write(f"Base: {base_results[idx - 1]}\n")
            f.write(f"LoRA: {lora_results[idx - 1]}\n")

    print(f"\n비교 결과 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="hell0ks ja-ko-vn-7b 번역 데모 (ja→ko) - LoRA 어댑터 적용 가능")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace 모델 이름 또는 로컬 경로")
    parser.add_argument("--lora_path", type=str, default=DEFAULT_LORA,
                        help="적용할 LoRA 어댑터 경로 (빈 문자열('')이면 base 모델만 사용)")
    parser.add_argument("--text", type=str, default=None,
                        help="번역할 일본어 문장")
    parser.add_argument("--compare", action="store_true",
                        help="base 모델과 LoRA 적용 모델을 둘 다 돌려서 정답/base/LoRA 비교 결과를 txt로 저장")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_COMPARE_OUTPUT_DIR,
                        help="--compare 결과 txt를 저장할 디렉토리")
    args = parser.parse_args()

    # 프로젝트 폴더로 별도 복사하지 않고 HuggingFace 기본 캐시
    # (~/.cache/huggingface)만 사용한다. 캐시에 없으면 자동으로 받고,
    # 있으면 그대로 재사용하므로 디스크에 모델이 두 벌 생기지 않는다.
    print(f"모델 로딩: {args.model} (HuggingFace 캐시 사용)")

    device = get_device()
    print(f"사용 디바이스: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if args.compare:
        run_compare(args.model, args.lora_path, tokenizer, device, args.output_dir)
        return

    model = load_model(args.model, args.lora_path, device)
    print("모델 로딩 완료!\n")

    sentences = [args.text] if args.text else [ja for ja, _ in TEST_SET]

    print("=" * 50)
    print("번역 결과 (ja → ko)")
    print("=" * 50)

    for sentence in sentences:
        results = translate(model, tokenizer, sentence)
        print(f"[JP] {sentence}")
        for i, result in enumerate(results, 1):
            print(f"[KR {i}] {result}")
        print("-" * 50)


if __name__ == "__main__":
    main()
