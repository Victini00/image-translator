import sys
import os
import re
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Windows 콘솔 기본 인코딩(cp949)이 일본어/특수문자 출력 시 깨지는 것 방지
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

"""
<사용법>

Masking(OCR) 결과의 merged_text는 recognition 오류(글자 오인식)를 포함할 수 있다
(예: もー→もI, でしょ→でしよ). 번역(hell0ks+LoRA)은 그대로 두고, 그 전에 별도의
범용 instruction-following 모델(Qwen2.5-7B-Instruct)에게 문맥상 원래 표기를
추론해서 고치게 시킨다.

(hell0ks 베이스 모델로도 먼저 시도해봤는데, "교정해줘" 같은 번역 외 지시를 전혀
이해 못 하고 무너졌음 - 지시문을 그대로 되뱉거나, 교정 대신 번역을 해버리거나,
반복 루프에 빠짐. 파라미터/프롬프트를 여러 번 고쳐도 마찬가지라서, hell0ks+LoRA는
번역 전용으로 그대로 두고 교정은 별도 모델로 분리함.)

결과는 원본 JSON에 paragraph별 "corrected_text" 필드를 추가해서 저장하고,
inpainting_rendering.py는 이 필드가 있으면 그걸, 없으면 merged_text를 그대로
번역 입력으로 쓴다 (하위 호환).
"""

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

CORRECTION_INSTRUCTION = (
    "너는 일본어 OCR 오탈자만 고치는 교정기야. 절대 지켜야 할 규칙:\n"
    "1. 오타로 확신되는 글자만 최소한으로 고쳐. 어미, 표현, 문장 구조는 절대 바꾸지 마 "
    "(예: 과거형을 현재형으로, 반말을 존댓말로 바꾸는 것도 금지).\n"
    "2. 오타인지 확신이 안 서면 절대 고치지 말고 입력을 한 글자도 안 바꾸고 그대로 돌려줘. "
    "이미 자연스러운 문장이면 손대지 마.\n"
    "3. 반드시 일본어(히라가나/가타카나/일본어 한자)만 써. 중국어 간체자나 로마자, "
    "특수 토큰은 절대 쓰지 마 - 입력에 이미 있던 알파벳/기호가 아니면 새로 만들지 마.\n"
    "4. 출력은 교정된 문장 딱 한 줄뿐, 설명이나 다른 말은 절대 붙이지 마.\n\n"
    "입력: {text}"
)

# Qwen이 확신 없이 문장을 새로 지어내거나(과도한 재작성), 중국어 간체자/원문에 없던
# 로마자를 섞어 쓰는 경우가 관찰됨 - 유사도로는 좋은 교정과 나쁜 교정을 못 가려서
# (둘 다 원문과 크게 달라짐), 대신 "명백히 잘못된 신호"만 감지해서 그런 경우엔
# 교정을 버리고 원문을 그대로 쓴다.
SIMPLIFIED_CHAR_DENYLIST = set("员绪确认贵头国岁儿")


def is_suspicious_correction(original, corrected):
    """새로 등장한 로마자/기호나 중국어 간체자가 있으면 교정을 신뢰하지 않는다."""
    orig_latin = set(re.findall(r"[a-zA-Z_<>]", original))
    new_latin = set(re.findall(r"[a-zA-Z_<>]", corrected)) - orig_latin
    if new_latin:
        return True
    new_simplified = [ch for ch in SIMPLIFIED_CHAR_DENYLIST if ch in corrected and ch not in original]
    if new_simplified:
        return True
    return False


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_correction_model(model_name, device):
    """교정용 범용 모델을 로드한다 (hell0ks+LoRA 번역 모델과는 완전히 별도)."""
    print(f"교정 모델 로딩: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if device.type == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16,
        ).to(device)

    model.eval()
    print("교정 모델 로딩 완료!")
    return tokenizer, model


def correct_text(tokenizer, model, text):
    """OCR 인식 원문(일본어) 한 문단을 교정한다.

    반환값: (최종 텍스트, 모델이 제안한 원본 결과, 거부 여부)
    - 모델 제안이 의심스러우면(중국어 간체/새 로마자 등) 원문을 그대로 최종 텍스트로 쓰되,
      모델이 뭘 제안했었는지도 같이 남겨서 로그에서 확인할 수 있게 한다.
    """
    if not text.strip():
        return text, text, False

    prompt = CORRECTION_INSTRUCTION.format(text=text)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    input_len = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max(16, len(text) + 10),  # 원문보다 살짝만 길게 - 가짜 대화/반복 못 늘어나게
        do_sample=False,  # 교정은 다양성보다 일관성이 중요 - greedy로 고정
        repetition_penalty=1.3,  # "でしよでしよでしよ..." 같은 반복 루프 억제
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    result = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    # 첫 줄만 사용 (모델이 그래도 다음 턴을 이어 쓰면 거기서 잘라냄)
    result = result.split("\n")[0].strip()

    if not result:
        return text, text, False
    if is_suspicious_correction(text, result):
        return text, result, True  # 신뢰 못 할 교정 - 원문 유지, 제안은 로그용으로 보존
    return result, result, False


def run_correction(json_path, save_json, model_name, device_str):
    device = torch.device(device_str) if device_str else get_device()
    print(f"사용 디바이스: {device}")
    tokenizer, model = load_correction_model(model_name, device)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"{len(data['paragraphs'])}개 문단 교정 시작...")
    for para in data["paragraphs"]:
        text = para["merged_text"].strip()
        if not text:
            para["corrected_text"] = text
            continue
        final_text, suggestion, rejected = correct_text(tokenizer, model, text)
        para["corrected_text"] = final_text
        if rejected:
            mark = f"  (거부됨 - 원문 유지, 모델 제안: {suggestion})"
        elif final_text != text:
            mark = "  (수정됨)"
        else:
            mark = ""
        print(f"  [문단 {para['id']}] {text} -> {final_text}{mark}")

    out_dir = os.path.dirname(save_json)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n교정 결과 저장: {save_json}")


def main():
    parser = argparse.ArgumentParser(description="OCR 인식 결과 교정 - 번역 전에 별도 범용 모델로 글자 오인식 교정")
    parser.add_argument("--json", type=str,
                        default="./../../output/ocr/masking_test/shirobako_paragraphs.json",
                        help="masking 출력 paragraphs JSON 경로")
    parser.add_argument("--save-json", type=str,
                        default="./../../output/ocr/masking_test/shirobako_paragraphs_corrected.json",
                        help="corrected_text 필드를 추가해서 저장할 경로")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="교정용 HuggingFace 모델 이름 또는 경로 (hell0ks+LoRA와 무관한 별도 모델)")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                        help="연산 디바이스 (기본값: cuda > mps > cpu 자동 감지)")
    args = parser.parse_args()

    run_correction(args.json, args.save_json, args.model, args.device)


if __name__ == "__main__":
    main()
