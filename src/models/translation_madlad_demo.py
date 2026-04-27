import os
import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


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

# MADLAD-400 타겟 언어 프리픽스
TGT_LANG_PREFIX = "<2ko>"  # 한국어
DEFAULT_MODEL = "google/madlad400-3b-mt"


def translate(model, tokenizer, text, tgt_prefix=TGT_LANG_PREFIX):
    # MADLAD-400은 입력 앞에 타겟 언어 프리픽스를 붙이는 방식
    input_text = f"{tgt_prefix} {text}"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.strip()


def main():
    parser = argparse.ArgumentParser(description="MADLAD-400 번역 데모 (ja→ko)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace 모델 이름 또는 로컬 경로")
    parser.add_argument("--text", type=str, default=None,
                        help="번역할 일본어 문장")
    args = parser.parse_args()

    # 로컬에 저장된 모델이 있으면 거기서 로드, 없으면 HuggingFace에서 다운로드 후 저장
    model_name = args.model
    local_path = os.path.join(LOCAL_MODEL_DIR, model_name.replace("/", "_"))

    if os.path.exists(local_path):
        print(f"로컬 모델 로딩: {local_path}")
        load_path = local_path
    else:
        print(f"HuggingFace에서 다운로드: {model_name}")
        print("(첫 실행 시 ~12GB 다운로드, 이후 로컬에서 로드)")
        load_path = model_name

    device = get_device()
    print(f"사용 디바이스: {device}")

    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        load_path,
        torch_dtype=torch.float16,
    ).to(device)

    # 로컬에 아직 없으면 저장
    if not os.path.exists(local_path):
        print(f"모델 로컬 저장 중: {local_path}")
        os.makedirs(local_path, exist_ok=True)
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
        print("저장 완료!")

    print("모델 로딩 완료!\n")

    # 테스트 문장들 (일반 + 만화체)
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
        result = translate(model, tokenizer, sentence)
        print(f"[JP] {sentence}")
        print(f"[KR] {result}")
        print("-" * 50)


if __name__ == "__main__":
    main()
