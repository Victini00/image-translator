import os
import cv2
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="PaddleOCR ds_width: true 설정을 위한 가로/세로 정보 추가 스크립트")
    
    # 기본값들을 현재 사용하시는 경로에 맞춰 설정했습니다.
    parser.add_argument("--img_dir", type=str, 
                        default="./../../data/processed/fine_tuning_answer_sheet_JParaCrawl_million/images/", 
                        help="이미지 파일들이 실제로 들어있는 폴더 경로")
    parser.add_argument("--label_file", type=str, 
                        default="./../../data/processed/fine_tuning_answer_sheet_JParaCrawl_million/val.txt", 
                        help="기존 텍스트 파일(label) 경로")
    parser.add_argument("--output_name", type=str, 
                        default="val_size_info.txt", 
                        help="새로 생성될 텍스트 파일 이름")

    args = parser.parse_args()

    # 출력 파일의 전체 경로 설정
    output_path = os.path.join(os.path.dirname(args.label_file), args.output_name)

    if not os.path.exists(args.label_file):
        print(f"Error: 라벨 파일을 찾을 수 없습니다: {args.label_file}")
        return

    with open(args.label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"작업 시작: {args.label_file}")
    print(f"이미지 참조 폴더: {args.img_dir}")
    print(f"총 {len(lines)}개의 데이터를 처리 중...")

    new_lines = []
    error_count = 0

    for line in tqdm(lines):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('\t')
        if len(parts) < 2:
            continue

        img_relative_path = parts[0] # 파일에 적힌 경로 (예: ./../../.../train_001.jpg)
        label_text = parts[1]        # 일본어 문장

        # [핵심] 파일명만 추출하여 실제 경로와 결합
        img_name = os.path.basename(img_relative_path)
        full_img_path = os.path.normpath(os.path.join(args.img_dir, img_name))

        # 이미지 읽기
        img = cv2.imread(full_img_path)
        
        if img is not None:
            h, w, _ = img.shape
            # PaddleOCR wh_aware 모드 형식: 경로 \t 텍스트 \t 너비 \t 높이
            new_line = f"{img_relative_path}\t{label_text}\t{w}\t{h}\n"
            new_lines.append(new_line)
        else:
            # 처음 5개 에러만 상세 출력하여 경로 확인 용도로 사용
            if error_count < 5:
                print(f"\n[Error] 파일을 찾을 수 없음: {full_img_path}")
            error_count += 1

    # 결과 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print("\n" + "="*40)
    print(f"작업 완료!")
    print(f"생성된 파일: {output_path}")
    print(f"성공: {len(new_lines)}개 / 실패: {error_count}개")
    print("="*40)

if __name__ == "__main__":
    main()