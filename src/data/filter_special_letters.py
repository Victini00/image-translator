import os
import re
import argparse
from tqdm import tqdm

# 허용할 문자 범위. 여기 없는 문자는 replacement로 치환된다.
#
# General Punctuation(\u2000-\u206F)이 빠져 있으면 「…」(U+2026)가 걸러진다.
# 만화 대사에서 「……」는 가장 흔한 기호인데 이 필터 때문에 학습 데이터에서
# 통째로 사라졌고(v4 학습셋 80만 줄 중 0회), 모델이 이 글자를 출력하는 법을
# 배우지 못했다. 실측상 v4 인식 오류의 78%가 말줄임표에서 나온다.
# ‥(U+2025) —(U+2014) ―(U+2015) ‐(U+2010) 도 같은 범위라 함께 살아난다.
SPECIAL_PATTERN = re.compile(
    r'[^\u0020-\u007E'   # ASCII
    r'\u2000-\u206F'     # General Punctuation (… ‥ — ― ‐)
    r'\u3000-\u303F'     # CJK 기호·구두점 (。 、 「 」 〜)
    r'\u3040-\u309F'     # 히라가나
    r'\u30A0-\u30FF'     # 가타카나 (ー)
    r'\u4E00-\u9FFF'     # 한자
    r'\uFF00-\uFFEF]'    # 전각 형태 (！ ？)
)


def filter_special_letters(input_path, output_path, replacement):
    special_pattern = SPECIAL_PATTERN

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
    parser.add_argument('--rep', type=str, default='',
                        help="허용 범위 밖 문자를 무엇으로 바꿀지. 기본값은 삭제(''). "
                             "예전 기본값 'A'는 만화에 나오지도 않는 알파벳을 학습시켰다.")

    args = parser.parse_args()
    filter_special_letters(args.input, args.output, args.rep)