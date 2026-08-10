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

# NLLB-200 언어 코드
SRC_LANG = "jpn_Jpan"   # 일본어
TGT_LANG = "kor_Hang"   # 한국어
DEFAULT_MODEL = "facebook/nllb-200-3.3B"


def translate(model, tokenizer, text, src_lang=SRC_LANG, tgt_lang=TGT_LANG):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    target_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=target_id,
        max_new_tokens=256,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        num_return_sequences=4,
    )

    results = [tokenizer.decode(out, skip_special_tokens=True).strip() for out in outputs]
    return results


def main():
    parser = argparse.ArgumentParser(description="NLLB-200 번역 데모 (ja→ko)")
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
        print("(첫 실행 시 ~13GB 다운로드, 이후 로컬에서 로드)")
        load_path = model_name

    device = get_device()
    print(f"사용 디바이스: {device}")

    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
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
