import json
import glob
import argparse
import os


def build_jsonl(input_dir, pattern, output_dir, output_name):
    """
    <사용법>

    [일본어]\\n[한국어]\\n(빈 줄) 형식으로 손번역된 *.txt 파일들을 모아
    {"ja": ..., "ko": ...} 형태의 jsonl 한 개로 통합 (hell0ks LoRA 학습용)
    """

    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError(f"입력 파일을 찾지 못함: {os.path.join(input_dir, pattern)}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_name)

    pair_count = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.read().split("\n")
            while lines and lines[-1] == "":
                lines.pop()

            i = 0
            file_pairs = 0
            while i + 1 < len(lines):
                ja, ko = lines[i], lines[i + 1]
                out_f.write(json.dumps({"ja": ja, "ko": ko}, ensure_ascii=False) + "\n")
                pair_count += 1
                file_pairs += 1
                i += 2
                if i < len(lines):
                    if lines[i] != "":
                        raise ValueError(f"{path}:{i + 1} 줄에서 빈 줄 구분자가 아님 - 형식 확인 필요")
                    i += 1

            print(f"  {os.path.basename(path)}: {file_pairs}쌍")

    print(f"\n작업 완료! 총 {len(files)}개 파일, {pair_count}쌍 통합.")
    print(f"결과 파일: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JA/KO 손번역 txt 파일들을 LoRA 학습용 jsonl로 통합")

    parser.add_argument("--input_dir", type=str,
                        default="./../../data/raw/texts/joujiboi/japanese-anime-speech-v2/translated",
                        help="ja/ko 쌍 txt 파일들이 있는 디렉토리")
    parser.add_argument("--pattern", type=str,
                        default="audio_transcription_list_processed_00[0-2][0-9].txt",
                        help="통합할 파일 glob 패턴 (0100은 테스트용이라 기본 패턴에서 제외됨)")
    parser.add_argument("--output_dir", type=str,
                        default="./../../data/raw/texts/joujiboi/japanese-anime-speech-v2/translated",
                        help="jsonl 저장 디렉토리")
    parser.add_argument("--output_name", type=str,
                        default="audio_transcription_list_processed_lora_train_set_v1.jsonl",
                        help="저장할 jsonl 파일 이름")

    args = parser.parse_args()

    build_jsonl(args.input_dir, args.pattern, args.output_dir, args.output_name)
