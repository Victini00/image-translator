import os
import argparse
from tqdm import tqdm
from itertools import islice

def extract_subset(input_path, output_path, n_lines):
    """
    입력 파일에서 첫 n_lines만큼의 줄만 읽어 새로운 파일로 복사합니다.
    """
    if not os.path.exists(input_path):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_path}")
        return

    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"🚀 {input_path}에서 {n_lines:,}줄 추출을 시작합니다.")

    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            # islice를 사용하여 메모리 부하 없이 n개 줄만 가져옵니다.
            subset = islice(f_in, n_lines)
            
            # tqdm을 사용해 진행 상황 표시 (단위: lines)
            for line in tqdm(subset, total=n_lines, desc="Extracting", unit="lines"):
                f_out.write(line)

    print(f"\n✨ 추출 완료!")
    print(f"   - 원본 파일: {input_path}")
    print(f"   - 저장 위치: {output_path}")
    print(f"   - 추출된 행 수: {n_lines:,}줄")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="대용량 텍스트 파일에서 상위 N개의 행 추출 도구")

    # 1. 입력 파일 경로
    parser.add_argument('--input', type=str, default="./../../data/processed/filtered_japanese_sentences_JParaCrawl.txt",
                        help='원본 대용량 .txt 파일 경로')
    
    # 2. 출력 파일 경로
    parser.add_argument('--output', type=str, default="./../../data/processed/filtered_japanese_sentences_JParaCrawl_million.txt",
                        help='추출된 내용을 저장할 파일 경로 및 이름')
    
    # 3. 추출할 행 수 (기본값 1,000,000)
    parser.add_argument('--lines', type=int, default=1000000,
                        help='추출할 행 수 (기본값: 1,000,000)')

    args = parser.parse_args()

    extract_subset(args.input, args.output, args.lines)