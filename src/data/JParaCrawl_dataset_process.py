import argparse
import os
import re
from tqdm import tqdm

def extract_and_save_sentences(input_file, min_len, max_len, output_dir, file_name):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, file_name)
    sentence_pattern = re.compile(r'[^。！？!?\n]+[。！？!?]?')
    
    # 파일의 전체 크기(Byte) 구하기
    file_size = os.path.getsize(input_file)
    processed_count = 0
    temp_buffer = ""

    print(f"데이터 분석 시작: {input_file} ({file_size / (1024**3):.2f} GB)")

    try:
        with open(input_file, 'r', encoding='utf-8') as f, \
             open(output_path, 'w', encoding='utf-8') as out_f:
            
            # unit='B', unit_scale=True를 설정하면 바이트 단위로 진행바가 표시됩니다.
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="처리 중") as pbar:
                for line in f:
                    # 현재 읽은 줄의 바이트 수를 tqdm에 업데이트
                    pbar.update(len(line.encode('utf-8')))
                    
                    parts = line.strip().split('\t')
                    if len(parts) < 5:
                        continue
                    
                    japanese_text = parts[-1]
                    sentences = sentence_pattern.findall(japanese_text)
                    
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent: continue

                        if temp_buffer:
                            temp_buffer += " " + sent
                        else:
                            temp_buffer = sent

                        while len(temp_buffer) >= min_len:
                            final_sent = temp_buffer[:max_len]
                            out_f.write(final_sent + '\n')
                            processed_count += 1
                            temp_buffer = temp_buffer[max_len:].strip()

            if len(temp_buffer) >= min_len:
                out_f.write(temp_buffer[:max_len] + '\n')
                processed_count += 1

    except FileNotFoundError:
        print(f"에러: 파일을 찾을 수 없습니다.")
        return

    print(f"\n✨ 작업 완료! 총 {processed_count}개의 문장이 생성되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="./../../data/raw/en-ja.bicleaner05.txt")
    parser.add_argument('--min', type=int, default=5)
    parser.add_argument('--max', type=int, default=12)
    parser.add_argument('--output', type=str, default="./../../data/processed/")
    parser.add_argument('--name', type=str, default="japanese_sentences_JParaCrawl.txt")

    args = parser.parse_args()
    extract_and_save_sentences(args.input, args.min, args.max, args.output, args.name)