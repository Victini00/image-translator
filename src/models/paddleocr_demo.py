import os
import argparse
from paddleocr import PaddleOCR

def main():
    parser = argparse.ArgumentParser(description="PaddleOCR 실행 스크립트")
    parser.add_argument('--det', type=str, default="PP-OCRv5_mobile_det")
    parser.add_argument('--rec', type=str, default="PP-OCRv5_mobile_rec")
    parser.add_argument('--img', type=str, default="./../../data/raw/images/shirobako_clockwise.png")
    parser.add_argument('--out', type=str, default="./../../output/v1")
    parser.add_argument('--lang', type=str, default="japan")
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # original
    '''
    ocr = PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        text_detection_model_name=args.det,
        text_recognition_model_name=args.rec,
        lang=args.lang,
        # use_gpu=True
    )
    '''
    # v1
    ocr = PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        text_detection_model_name=args.det,
        text_recognition_model_dir="./../../models/PP-OCRv5_mobile_rec_jp_fine_tuned/complete_model",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        lang=args.lang,
    )
    

    if not os.path.exists(args.img):
        print(f"❌ 에러: '{args.img}' 찾을 수 없음")
        return

    print(f"🚀 OCR 분석 시작: {args.img}")
    result = ocr.predict(args.img)

    print("\n" + "="*50)
    print("📖 인식된 문장 결과 (위->아래, 왼쪽->오른쪽 정렬)")
    print("="*50)

    for res in result:
        try:
            # [수정 포인트] 객체 속성(res.dt_polys)과 딕셔너리(res['dt_polys']) 방식을 모두 확인
            if hasattr(res, 'dt_polys'):
                # 속성 방식인 경우
                polys = res.dt_polys
                texts = res.rec_texts
                scores = res.rec_scores
            elif isinstance(res, dict) and 'dt_polys' in res:
                # 완전한 딕셔너리 방식인 경우
                polys = res['dt_polys']
                texts = res['rec_texts']
                scores = res['rec_scores']
            else:
                # 그 외의 경우 (딕셔너리처럼 접근 시도)
                polys = res.get('dt_polys', [])
                texts = res.get('rec_texts', [])
                scores = res.get('rec_scores', [])

            # 1. 데이터를 하나로 묶기
            combined_data = []
            for poly, text, score in zip(polys, texts, scores):
                combined_data.append({
                    'poly': poly,
                    'text': text,
                    'score': score
                })

            # 2. 정렬 (y좌표 -> x좌표 순)
            # poly[0][1]은 좌상단 y좌표
            sorted_res = sorted(combined_data, key=lambda x: (x['poly'][0][1], x['poly'][0][0]))

            # 3. 출력
            full_text = []
            for item in sorted_res:
                print(f"[{item['score']:.2f}] {item['text']}")
                full_text.append(item['text'])

            print("-" * 50)

        except Exception as e:
            print(f"⚠️ 결과 처리 중 다시 오류 발생: {e}")
            # 진짜 마지막으로 객체에 무엇이 들어있는지 강제로 출력해서 확인
            print("실제 데이터 구조 맛보기:", str(res)[:200]) 

        # 저장 기능 실행
        res.save_to_img(args.out)
        res.save_to_json(args.out)
        
    print(f"✅ 분석 완료! 결과 확인: {args.out}")

if __name__ == "__main__":
    main()