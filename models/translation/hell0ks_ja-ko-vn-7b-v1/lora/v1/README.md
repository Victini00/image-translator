# 번역 모델 LoRA 어댑터 (v1)

로컬 번역 경로(`image_translator_v1.py`)에서 쓰는 LoRA 어댑터를 두는 곳입니다.

베이스 모델 [hell0ks/ja-ko-vn-7b-v1](https://huggingface.co/hell0ks/ja-ko-vn-7b-v1)에
만화 대사 말투로 파인튜닝한 어댑터를 얹어서 씁니다.
베이스 모델 자체는 HuggingFace 캐시로 자동 다운로드되므로 여기 둘 필요가 없습니다.

## 필요한 파일

```
models/translation/hell0ks_ja-ko-vn-7b-v1/lora/v1/
  adapter_model.safetensors     약 153MB
  adapter_config.json
```

## 준비 방법

이 어댑터는 직접 학습한 것이라 저장소에 포함되어 있지 않습니다. 세 가지 선택지가 있습니다.

1. **Gemini 경로를 쓴다** (권장) — `image_translator_v1_using_gemini.py`는 이 모델이
   필요 없습니다. 인식·번역 품질도 더 낫습니다.
2. **베이스 모델만 쓴다** — 렌더링 단계에 `--lora-path ""`를 넘기면 LoRA 없이
   베이스 모델로만 번역합니다.
3. **직접 학습한다** — `src/models/translation_hell0ks_lora_train.py` 참고.
   학습 데이터는 `src/data/lora_pair_dataset_process.py`로 만듭니다.
