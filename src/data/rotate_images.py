import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

train_num = 640000
val_num = 160000

def rotate_single_image(args):
    """이미지 한 장을 시계 방향으로 90도 회전하여 저장"""
    full_path = args
    if not os.path.exists(full_path):
        return f"❌ 찾을 수 없음: {full_path}"
        
    try:
        with Image.open(full_path) as img:
            # 시계 방향 90도 회전
            rotated = img.rotate(180, expand=True)
            
            # 원본 포맷 유지하며 덮어쓰기
            rotated.save(full_path, quality=95)
        return "✅ 성공"
    except Exception as e:
        return f"⚠️ 에러: {e}"

def main():
    # 1. 경로 설정 (사용자 환경에 맞게 자동 조정)
    base_dir = base_dir = "./../../data/processed/fine_tuning_answer_sheet_JParaCrawl_million/images" 
    
    # 2. 작업 범위 설정
    configs = [
        {"prefix": "train_", "start": 0, "end": train_num-1},
        {"prefix": "val_", "prefix_val": "val_", "start": 0, "end": val_num-1}
    ]

    all_file_paths = []
    
    # train 파일 경로 생성
    for i in range(0, train_num):
        all_file_paths.append(os.path.join(base_dir, f"train_{i:06d}.jpg"))
        
    # val 파일 경로 생성
    for i in range(0, val_num):
       all_file_paths.append(os.path.join(base_dir, f"val_{i:06d}.jpg"))

    print(f"🚀 총 {len(all_file_paths)}개의 이미지 회전 작업을 시작합니다.")
    print(f"📍 대상 경로: {os.path.abspath(base_dir)}")

    # 3. 병렬 처리 실행
    with ProcessPoolExecutor() as executor:
        # 결과를 리스트로 받아 성공 여부 확인
        results = list(tqdm(executor.map(rotate_single_image, all_file_paths), total=len(all_file_paths)))

    # 4. 결과 요약 출력
    success_count = sum(1 for r in results if "✅" in r)
    fail_count = len(all_file_paths) - success_count
    
    print(f"\n📊 작업 완료!")
    print(f"   - 성공: {success_count}개")
    if fail_count > 0:
        print(f"   - 실패/미발견: {fail_count}개 (경로나 파일명 자릿수를 확인하세요)")

if __name__ == "__main__":
    main()

