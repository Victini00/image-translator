import json
import argparse
import os
import re

def extract_and_save_sentences(input_file, min_len, max_len, output_dir, file_name):
    """
    <사용법>

    toxicity_dataset.jsonl 형태의 파일을
    각 줄에 min ~ max 개의 문자를 가진 적당한 길이로 잘라 넣음
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, file_name)
    
    # 문장 구분 패턴 (줄바꿈 포함 제거)
    sentence_pattern = re.compile(r'[^。！？!?\n]+[。！？!?]?')

    processed_count = 0
    temp_buffer = "" # 짧은 문장을 모아둘 버퍼

    with open(input_file, 'r', encoding='utf-8') as f, \
         open(output_path, 'w', encoding='utf-8') as out_f:
        
        for line in f:
            try:
                data = json.loads(line)
                text = data.get('text', '').replace('\n', ' ') # 줄바꿈을 공백으로 치환
                sentences = sentence_pattern.findall(text)
                
                for sent in sentences:
                    sent = sent.strip()
                    if not sent: continue # 빈 문자열 건너뛰기

                    # 현재 문장을 버퍼에 추가
                    if temp_buffer:
                        temp_buffer += " " + sent
                    else:
                        temp_buffer = sent

                    # 버퍼의 길이가 min_len 이상이 되면 저장 후보
                    if len(temp_buffer) >= min_len:
                        # 만약 max_len을 넘으면 max_len만큼만 자름
                        final_sent = temp_buffer[:max_len]
                        out_f.write(final_sent + '\n')
                        processed_count += 1
                        
                        # 사용한 만큼 버퍼에서 제거 (남은 부분은 다음 문장과 합침)
                        temp_buffer = temp_buffer[max_len:].strip()
                        
            except json.JSONDecodeError:
                continue
        
        # 모든 루프가 끝난 후 버퍼에 남은 것이 min_len 이상이면 마지막으로 저장
        if len(temp_buffer) >= min_len:
            out_f.write(temp_buffer[:max_len] + '\n')
            processed_count += 1

    print(f"작업 완료! 총 {processed_count}개의 문장이 생성되었습니다.")
    print(f"결과 파일: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="일본어 문장 추출 및 길이별 커팅 도구")
    
    # 실행 인자 설정 (기본값 포함)
    parser.add_argument('--input', type=str, default="./../../data/raw/toxicity_dataset.jsonl", help='입력 jsonl 파일 경로')
    parser.add_argument('--min', type=int, default=10, help='문장 최소 길이 (default: 10)')
    parser.add_argument('--max', type=int, default=20, help='문장 최대 길이 (default: 50)')
    parser.add_argument('--output', type=str, default="./../../data/processed", help='저장할 디렉토리 경로 (default: ./output)')
    parser.add_argument('--name', type=str, default="japanese_sentences_toxicity.txt", help='저장할 파일 이름')

    args = parser.parse_args()

    # 함수 실행
    extract_and_save_sentences(args.input, args.min, args.max, args.output, args.name)