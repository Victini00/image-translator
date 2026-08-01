import argparse
from transformers import EncoderDecoderModel, BertJapaneseTokenizer, PreTrainedTokenizerFast

DEFAULT_MODEL = "sappho192/ffxiv-ja-ko-translator"
ENCODER_MODEL = "cl-tohoku/bert-base-japanese-v2"
DECODER_MODEL = "skt/kogpt2-base-v2"


def translate(model, src_tokenizer, trg_tokenizer, text, num_return_sequences=4):
    embeddings = src_tokenizer(
        text,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_tensors="pt",
    )

    outputs = model.generate(
        **embeddings,
        max_length=256,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        num_return_sequences=num_return_sequences,
    )

    results = [
        trg_tokenizer.decode(out[1:-1].cpu(), skip_special_tokens=True).strip()
        for out in outputs
    ]
    return results


def main():
    parser = argparse.ArgumentParser(description="ffxiv-ja-ko-translator 번역 데모 (ja→ko)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace 모델 이름 또는 로컬 경로")
    parser.add_argument("--text", type=str, default=None,
                        help="번역할 일본어 문장")
    args = parser.parse_args()

    # HF 캐시(~/.cache/huggingface/)를 그대로 사용 — EncoderDecoderModel은 save_pretrained 불가
    print("토크나이저 로딩...")
    src_tokenizer = BertJapaneseTokenizer.from_pretrained(ENCODER_MODEL)
    trg_tokenizer = PreTrainedTokenizerFast.from_pretrained(DECODER_MODEL)

    print(f"모델 로딩: {args.model}")
    model = EncoderDecoderModel.from_pretrained(args.model, low_cpu_mem_usage=False, _fast_init=False)
    model.eval()
    print("사용 디바이스: cpu\n")

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
        results = translate(model, src_tokenizer, trg_tokenizer, sentence)
        print(f"[JP] {sentence}")
        for i, result in enumerate(results, 1):
            print(f"[KR {i}] {result}")
        print("-" * 50)


if __name__ == "__main__":
    main()
