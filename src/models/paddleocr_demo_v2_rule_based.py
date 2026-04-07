import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR


def group_lines_into_paragraphs(polys, texts, scores, margin_ratio=1.5):
    """
    개별 텍스트 라인들을 문단 단위로 그룹핑한다.

    기준:
    1. 두 라인의 글자 높이가 비슷 (0.5~2.0배 범위)
    2. 수평 방향으로 x좌표 범위가 겹침
    3. 수직 간격이 평균 글자 높이의 margin_ratio배 이내

    Union-Find 방식으로 병합.
    """
    n = len(polys)
    if n == 0:
        return []

    # 각 라인의 bbox 정보 계산
    bboxes = []
    for poly in polys:
        poly = np.array(poly)
        x_min, y_min = poly.min(axis=0)
        x_max, y_max = poly.max(axis=0)
        h = y_max - y_min
        bboxes.append((x_min, y_min, x_max, y_max, h))

    # Union-Find
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        x_min_i, y_min_i, x_max_i, y_max_i, h_i = bboxes[i]
        for j in range(i + 1, n):
            x_min_j, y_min_j, x_max_j, y_max_j, h_j = bboxes[j]

            # 조건 1: 글자 높이 비슷
            if h_i == 0 or h_j == 0:
                continue
            ratio = h_i / h_j
            if ratio < 0.5 or ratio > 2.0:
                continue

            # 조건 2: x축 겹침
            x_overlap = min(x_max_i, x_max_j) - max(x_min_i, x_min_j)
            min_width = min(x_max_i - x_min_i, x_max_j - x_min_j)
            if min_width > 0 and x_overlap / min_width < 0.3:
                continue

            # 조건 3: y축 간격이 평균 높이의 margin_ratio배 이내
            avg_h = (h_i + h_j) / 2
            y_gap = max(0, max(y_min_i, y_min_j) - min(y_max_i, y_max_j))
            if y_gap > avg_h * margin_ratio:
                continue

            union(i, j)

    # 그룹별로 묶기
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    # 각 그룹을 y좌표 순으로 정렬
    paragraphs = []
    for indices in groups.values():
        indices_sorted = sorted(indices, key=lambda i: (bboxes[i][1], bboxes[i][0]))
        para = {
            'indices': indices_sorted,
            'polys': [polys[i] for i in indices_sorted],
            'texts': [texts[i] for i in indices_sorted],
            'scores': [scores[i] for i in indices_sorted],
            'merged_text': ' '.join(texts[i] for i in indices_sorted),
        }
        # merged bbox
        all_pts = np.concatenate([np.array(polys[i]) for i in indices_sorted])
        para['bbox'] = (
            int(all_pts[:, 0].min()),
            int(all_pts[:, 1].min()),
            int(all_pts[:, 0].max()),
            int(all_pts[:, 1].max()),
        )
        paragraphs.append(para)

    # 문단을 y좌표 순으로 정렬
    paragraphs.sort(key=lambda p: (p['bbox'][1], p['bbox'][0]))
    return paragraphs


def draw_paragraph_boxes_on_ocr_result(ocr_res_img_path, paragraphs):
    """save_to_img()가 만든 2패널 이미지의 오른쪽 패널에 문단 빨간 박스를 추가한다."""
    img = Image.open(ocr_res_img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # 2패널 이미지: 왼쪽=원본+detection, 오른쪽=recognition
    # 오른쪽 패널 시작 x좌표 = 전체 너비 / 2
    x_offset = img.width // 2

    for i, para in enumerate(paragraphs):
        x_min, y_min, x_max, y_max = para['bbox']
        # 오른쪽 패널에 그리기
        rx_min = x_offset + x_min
        rx_max = x_offset + x_max
        # 빨간 반투명 채우기
        draw.rectangle([rx_min, y_min, rx_max, y_max], fill=(255, 0, 0, 30))
        # 빨간 테두리
        draw.rectangle([rx_min, y_min, rx_max, y_max], outline=(255, 0, 0, 255), width=3)
        # 문단 번호 표시
        draw.text((rx_min + 4, y_min + 2), f"P{i+1}", fill=(255, 0, 0, 255))

    img = Image.alpha_composite(img, overlay).convert("RGB")
    img.save(ocr_res_img_path)
    print(f"문단 박스 추가 저장: {ocr_res_img_path}")


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR v2 - 문단 그룹핑")
    parser.add_argument('--det', type=str, default="PP-OCRv5_mobile_det")
    parser.add_argument('--rec', type=str, default="PP-OCRv5_mobile_rec")
    parser.add_argument('--img', type=str, default="./../../data/raw/images/shirobako.jpg")
    parser.add_argument('--out', type=str, default="./../../output/v4")
    parser.add_argument('--lang', type=str, default="japan")
    parser.add_argument('--margin', type=float, default=1.5,
                        help="문단 그룹핑 y간격 임계값 (글자높이 배수)")
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # v4 fine-tuned 모델
    ocr = PaddleOCR(
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
        text_detection_model_name=args.det,
        text_recognition_model_dir="./../../models/PP-OCRv5_mobile_rec_jp_fine_tuned_v4/complete_model",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        lang=args.lang,
    )

    if not os.path.exists(args.img):
        print(f"에러: '{args.img}' 찾을 수 없음")
        return

    print(f"OCR 분석 시작: {args.img}")
    result = ocr.predict(args.img)

    for res in result:
        # 기존 시각화 (PaddleOCR 내장)
        res.save_to_img(args.out)
        res.save_to_json(args.out)

        # 결과 추출
        if hasattr(res, 'dt_polys'):
            polys = res.dt_polys
            texts = res.rec_texts
            scores = res.rec_scores
        elif isinstance(res, dict) and 'dt_polys' in res:
            polys = res['dt_polys']
            texts = res['rec_texts']
            scores = res['rec_scores']
        else:
            polys = res.get('dt_polys', [])
            texts = res.get('rec_texts', [])
            scores = res.get('rec_scores', [])

        # 문단 그룹핑
        paragraphs = group_lines_into_paragraphs(polys, texts, scores, margin_ratio=args.margin)

        # 콘솔 출력
        print("\n" + "=" * 50)
        print(f"인식 결과: {len(polys)}개 라인 -> {len(paragraphs)}개 문단")
        print("=" * 50)

        for i, para in enumerate(paragraphs):
            print(f"\n[문단 {i+1}] ({len(para['texts'])}줄)")
            for text, score in zip(para['texts'], para['scores']):
                print(f"  [{score:.2f}] {text}")
            print(f"  => 합침: {para['merged_text']}")

        # save_to_img가 생성한 2패널 이미지 파일 찾기
        img_basename = os.path.splitext(os.path.basename(args.img))[0]
        ocr_res_img_path = os.path.join(args.out, f"{img_basename}_ocr_res_img.jpg")
        if os.path.exists(ocr_res_img_path):
            draw_paragraph_boxes_on_ocr_result(ocr_res_img_path, paragraphs)
        else:
            print(f"2패널 이미지를 찾을 수 없음: {ocr_res_img_path}")

    print(f"\n분석 완료! 결과 확인: {args.out}")


if __name__ == "__main__":
    main()
