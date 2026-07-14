import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# 프로젝트 루트 기준 모델 저장 경로
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
LOCAL_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "translation")

DEFAULT_MODEL = "hell0ks/ja-ko-vn-7b-v1"


def translate(model, tokenizer, text, num_return_sequences=1):
    # Completions 모드: 일본어 원문만 입력
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
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
        pad_token_id=tokenizer.eos_token_id,
    )
    print(f"  done.", flush=True)

    # 입력 토큰 제외하고 생성된 부분만 디코딩
    results = [
        tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
        for out in outputs
    ]
    return results


def main():
    parser = argparse.ArgumentParser(description="hell0ks ja-ko-vn-7b 번역 데모 (ja→ko)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace 모델 이름 또는 로컬 경로")
    parser.add_argument("--text", type=str, default=None,
                        help="번역할 일본어 문장")
    args = parser.parse_args()

    model_name = args.model
    local_path = os.path.join(LOCAL_MODEL_DIR, model_name.replace("/", "_"))

    if os.path.exists(local_path):
        print(f"로컬 모델 로딩: {local_path}")
        load_path = local_path
    else:
        print(f"HuggingFace에서 다운로드: {model_name}")
        print("(첫 실행 시 ~14GB 다운로드, 이후 로컬에서 로드)")
        load_path = model_name

    device = get_device()
    print(f"사용 디바이스: {device}")

    tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.float16,
    ).to(device)

    if not os.path.exists(local_path):
        print(f"모델 로컬 저장 중: {local_path}")
        os.makedirs(local_path, exist_ok=True)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
        print("저장 완료!")

    print("모델 로딩 완료!\n")

    test_sentences = [
        "お前はもう死んでいる。",
        "私の名前は田中です。よろしくお願いします。",
        "今日の天気はとても良いですね。",
        "この漫画、マジで面白いんだけど！",
        "先輩、ちょっと待ってください！",
    ]

    if args.text:
        test_sentences = [args.text]

    print("=" * 50)
    print("번역 결과 (ja → ko)")
    print("=" * 50)

    for sentence in test_sentences:
        results = translate(model, tokenizer, sentence)
        print(f"[JP] {sentence}")
        for i, result in enumerate(results, 1):
            print(f"[KR {i}] {result}")
        print("-" * 50)


if __name__ == "__main__":
    main()
