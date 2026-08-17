import os
import argparse
import subprocess
import shutil

def ocr_rec_answer_label_generator(input_file, output_dir, train_ratio, dict_name):
    """
    Paddleocr에서 recognition model의 fine-tuning에 사용할 데이터들의
    정답지 세트 제작 (문장 순서 유지 버전)
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 기존 dict.txt 읽기
    dict_path = os.path.join(output_dir, dict_name)
    existing_chars = set()
    
    if os.path.exists(dict_path):
        with open(dict_path, 'r', encoding='utf-8') as f:
            existing_chars = set(line.rstrip('\n') for line in f)

    # 2. 입력 문장 읽기 
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]

    # 3. 새로운 문자 추출 및 dict.txt 업데이트
    new_chars = set()
    for sent in sentences:
        for char in sent:
            if char not in existing_chars:
                new_chars.add(char)

    if new_chars:
        with open(dict_path, 'a', encoding='utf-8') as f:
            for char in sorted(list(new_chars)):
                f.write(char + '\n')
        print(f"dict.txt 업데이트 완료: {len(new_chars)}개의 새 문자 추가")

    # 4. 데이터 분할 (Shuffle 없이 순차적으로 분할)
    # 원본 파일의 앞부분은 Train, 뒷부분은 Val로 할당
    split_idx = int(len(sentences) * train_ratio)
    train_set = sentences[:split_idx]
    val_set = sentences[split_idx:]

    def save_annotation(data_list, mode, start_index=0):
        file_path = os.path.join(output_dir, f"{mode}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            for i, sent in enumerate(data_list):
                line = f"images/{mode}_{i:06d}.jpg\t{sent}\n"
                f.write(line)
        return file_path

    t_path = save_annotation(train_set, "train")
    v_path = save_annotation(val_set, "val")

    print(f"정답지 생성 완료! (원본 순서 유지)")
    print(f"   - Train: {len(train_set)}개 -> {t_path}")
    print(f"   - Val: {len(val_set)}개 -> {v_path}")

    return train_set, val_set

def run_trdg_auto(sentences, mode, output_dir, trdg_path, v_ratio=0.8):
    """
    TRDG 옵션 미지원 문제를 해결하기 위해 이미지를 임시 폴더에 생성 후 이름을 직접 변경
    """
    split_idx = int(len(sentences) * v_ratio)
    v_set = sentences[:split_idx]
    h_set = sentences[split_idx:]

    tasks = [
        {'name': f'{mode}_vertical', 'data': v_set, 'or': '1', 'offset': 0},
        {'name': f'{mode}_horizontal', 'data': h_set, 'or': '0', 'offset': split_idx}
    ]

    final_image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(final_image_dir):
        os.makedirs(final_image_dir)

    for task in tasks:
        if not task['data']: continue
        
        # 1. 임시 작업 폴더 및 텍스트 파일 생성
        temp_task_dir = os.path.join(output_dir, f"temp_{task['name']}")
        if not os.path.exists(temp_task_dir): os.makedirs(temp_task_dir)
        
        tmp_txt = os.path.join(output_dir, f"tmp_{task['name']}.txt")
        with open(tmp_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(task['data']))

        # 2. TRDG 실행 (문제가 된 --prefix, --start_index 제거)
        print(f"🚀 trdg 실행: {task['name']} (방향: {task['or']})")
        
        na_val = "3" if mode == "train" else "4"
        
        cmd = [
            "python", os.path.join(trdg_path, "run.py"),
            "-l", "ja",
            "-i", tmp_txt,
            "-c", str(len(task['data'])),
            "-or", task['or'],
            "-b", "1", # 배경을 white plain으로 수정
            "--output_dir", temp_task_dir, # 임시 폴더에 먼저 생성
            "-na", na_val
        ]
        
        subprocess.run(cmd, check=True)

        # 3. 파일 이름 변경 및 최종 폴더로 이동 (가장 중요)
        # TRDG가 생성한 파일들을 하나씩 읽어서 이름을 변경
        generated_files = sorted(os.listdir(temp_task_dir))
        
        for i, filename in enumerate(generated_files):
            # i + task['offset']을 통해 전체 순서에 맞는 인덱스 계산
            # 실제 TRDG가 생성한 파일 이름 구조에 따라 필요시 수정
            old_path = os.path.join(temp_task_dir, filename)
            
            # 우리가 정답지(txt)에 기록한 형식과 동일하게 이름 결정
            # (예: train_000000.jpg)
            new_name = f"{mode}_{(i + task['offset']):06d}{os.path.splitext(filename)[1]}"
            new_path = os.path.join(final_image_dir, new_name)
            
            shutil.move(old_path, new_path)

        # 4. 임시 파일 및 폴더 삭제
        os.remove(tmp_txt)
        shutil.rmtree(temp_task_dir)
        print(f" {task['name']} 이미지 이동 및 정리 완료")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    base_processed_path = './../../data/processed'
    trdg_run_py_path = './../../external/TextRecognitionDataGenerator/trdg'
    
    # ocr_rec_answer_label_generator
    parser.add_argument('--input', type=str, 
                        default=os.path.join(base_processed_path, 'filtered_japanese_sentences_JParaCrawl_million.txt'))
    parser.add_argument('--output', type=str, 
                        default=os.path.join(base_processed_path, 'fine_tuning_answer_sheet_JParaCrawl_million'))
    parser.add_argument('--dict_name', type=str, default='dict.txt')
    parser.add_argument('--ratio', type=float, default=0.8, help="train, val의 분할 비율")

    # run_trdg_auto
    parser.add_argument('--trdg_path', type=str,
                        default=trdg_run_py_path, 
                        help='TextRecognitionDataGenerator의 run.py가 있는 경로')
    parser.add_argument('--v_ratio', type=float, default=0.8, 
                        help='세로쓰기 문장의 비율 (default: 0.8)')

    args = parser.parse_args()

    # 1. 정답지 및 사전 생성
    train_sentences, val_sentences = ocr_rec_answer_label_generator(
        args.input, args.output, args.ratio, args.dict_name
    )

    # 2. 이미지 생성 (Train) - 내부적으로 vertical/horizontal 나누어 처리
    run_trdg_auto(train_sentences, "train", args.output, args.trdg_path, args.v_ratio)

    # 3. 이미지 생성 (Val)
    run_trdg_auto(val_sentences, "val", args.output, args.trdg_path, args.v_ratio)

    print(f"\n 작업 완료! 결과 폴더: {args.output}")