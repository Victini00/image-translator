import argparse
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


DEFAULT_MODEL = "hell0ks/ja-ko-vn-7b-v1"

# hell0ks 계열(Llama 아키텍처) 표준 LoRA target modules
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


class JaKoPairDataset(Dataset):
    """
    jsonl({"ja": ..., "ko": ...}) 쌍을 hell0ks의 ChatML 템플릿으로 인코딩한다.
    user(일본어 원문) 구간은 loss에서 마스킹하고, assistant(한국어 번역) 구간만 학습한다.
    """

    def __init__(self, jsonl_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.pairs.append((obj["ja"], obj["ko"]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ja, ko = self.pairs[idx]

        prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ja}],
            add_generation_prompt=True,
            tokenize=True,
        )
        full_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ja}, {"role": "assistant", "content": ko}],
            add_generation_prompt=False,
            tokenize=True,
        )

        full_ids = full_ids[: self.max_length]
        prompt_len = min(len(prompt_ids), len(full_ids))

        labels = list(full_ids)
        labels[:prompt_len] = [-100] * prompt_len

        return {"input_ids": full_ids, "labels": labels}


def collate_fn(batch, pad_token_id):
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser(description="hell0ks ja-ko-vn-7b 말투 교정용 QLoRA 파인튜닝")
    parser.add_argument("--base_model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=str,
                        default="./../../data/raw/texts/joujiboi/japanese-anime-speech-v2/translated/audio_transcription_list_processed_lora_train_set_v1.jsonl")
    parser.add_argument("--output_dir", type=str,
                        default="./../../models/translation/hell0ks_ja-ko-vn-7b-v1/lora/v1")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다 (4bit QLoRA는 CPU/MPS에서 지원되지 않음).")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    print(f"베이스 모델 로딩(4bit): {args.base_model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = JaKoPairDataset(args.data, tokenizer, args.max_length)
    print(f"학습 데이터: {len(dataset)}쌍")

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )

    trainer.train()

    print(f"LoRA 어댑터 저장: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("완료!")


if __name__ == "__main__":
    main()
