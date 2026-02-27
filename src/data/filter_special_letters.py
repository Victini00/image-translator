import os
import re
import argparse
from tqdm import tqdm

def filter_special_letters(input_path, output_path, replacement):
    special_pattern = re.compile(
        r'[^\u0020-\u007E'
        r'\u3000-\u303F'
        r'\u3040-\u309F'
        r'\u30A0-\u30FF'
        r'\u4E00-\u9FFF'
        r'\uFF00-\uFFEF]'
    )

    if not os.path.exists(input_path):
        print(f"입력 파일을 찾을 수 없습니다: {input_path}")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_count = 0
    replaced_char_count = 0

    # total을 제거하여 파일을 미리 읽지 않고 바로 시작합니다.
    with open(input_path, 'r', encoding='utf-8') as f, \
         open(output_path, 'w', encoding='utf-8') as out_f:
        
        # unit="lines"를 추가해 초당 몇 줄을 처리하는지 보여줍니다.
        for line in tqdm(f, desc="Filtering", unit="lines"):
            sentence = line.strip()
            if not sentence:
                continue

            filtered_sentence, count = special_pattern.subn(replacement, sentence)
            
            out_f.write(filtered_sentence + '\n')
            
            if count > 0:
                replaced_char_count += count
            
            processed_count += 1

    print(f"\n✨ 필터링 완료!")
    print(f"   - 총 처리 문장: {processed_count}개")
    print(f"   - 대체된 문자 수: {replaced_char_count}개")
    print(f"   - 저장 위치: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="일본어 문장 내 이모지/특수문자 필터링 도구")
    parser.add_argument('--input', type=str, default="./../../data/processed/japanese_sentences_JParaCrawl.txt")
    parser.add_argument('--output', type=str, default="./../../data/processed/filtered_japanese_sentences_JParaCrawl.txt")
    parser.add_argument('--rep', type=str, default='A')

    args = parser.parse_args()
    filter_special_letters(args.input, args.output, args.rep)