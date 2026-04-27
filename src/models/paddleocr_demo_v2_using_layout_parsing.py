import os
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw
from paddleocr import PaddleOCR
from ultralytics import YOLO
from huggingface_hub import hf_hub_download


def load_bubble_detector():
    """YOLOv8 만화 말풍선 검출 모델 로드"""
    model_path = hf_hub_download(
        repo_id="ogkalu/comic-speech-bubble-detector-yolov8m",
        filename="comic-speech-bubble-detector.pt",
    )
    return YOLO(model_path)


def detect_bubbles(model, img_path, conf_thresh=0.5):
    """
    이미지에서 말풍선 영역을 검출하여 bbox 리스트 반환.
    PaddleOCR가 이미지를 90도 회전해서 처리하므로,
    버블 좌표도 같은 회전을 적용하여 OCR 좌표계에 맞춘다.
    """
    from PIL import Image as PILImage
    orig_w, orig_h = PILImage.open(img_path).size

    results = model(img_path, verbose=False)
    bubbles = []
    for r in results:
        for i in range(len(r.boxes)):
            if r.boxes.conf[i] >= conf_thresh:
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                # 원본 좌표 (x,y) -> 90도 시계방향 회전 좌표 (orig_h - y, x)
                # bbox 변환: (x1,y1,x2,y2) -> (orig_h-y2, x1, orig_h-y1, x2)
                rx1 = orig_h - y2
                ry1 = x1
                rx2 = orig_h - y1
                ry2 = x2
                bubbles.append((rx1, ry1, rx2, ry2))
    return bubbles


def group_lines_by_bubbles(polys, texts, scores, bubbles):
    """
    OCR 라인들을 말풍선 영역 기준으로 그룹핑한다.
    각 라인의 중심점이 어느 말풍선 안에 있는지 판단.
    말풍선에 속하지 않는 라인은 개별 문단으로 처리.
    """
    n = len(polys)
    # 각 라인의 중심점 계산
    centers = []
    for poly in polys:
        poly_arr = np.array(poly)
        cx = poly_arr[:, 0].mean()
        cy = poly_arr[:, 1].mean()
        centers.append((cx, cy))

    # 말풍선별 그룹
    bubble_groups = {i: [] for i in range(len(bubbles))}
    unassigned = []

    for li in range(n):
        cx, cy = centers[li]
        assigned = False
        for bi, (bx1, by1, bx2, by2) in enumerate(bubbles):
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                bubble_groups[bi].append(li)
                assigned = True
                break
        if not assigned:
            unassigned.append(li)

    # 문단 리스트 생성
    paragraphs = []

    for bi, indices in bubble_groups.items():
        if not indices:
            continue
        # bbox 계산 (말풍선 bbox 대신 실제 텍스트 영역 사용)
        all_pts = np.concatenate([np.array(polys[i]) for i in indices])
        bbox = (
            int(all_pts[:, 0].min()),
            int(all_pts[:, 1].min()),
            int(all_pts[:, 0].max()),
            int(all_pts[:, 1].max()),
        )
        # y좌표 순으로 정렬
        indices_sorted = sorted(indices, key=lambda i: (centers[i][1], centers[i][0]))
        paragraphs.append({
            'indices': indices_sorted,
            'polys': [polys[i] for i in indices_sorted],
            'texts': [texts[i] for i in indices_sorted],
            'scores': [scores[i] for i in indices_sorted],
            'merged_text': ' '.join(texts[i] for i in indices_sorted),
            'bbox': bbox,
            'bubble_bbox': bubbles[bi],
        })

    # 미할당 라인은 개별 문단
    for li in unassigned:
        poly_arr = np.array(polys[li])
        bbox = (
            int(poly_arr[:, 0].min()),
            int(poly_arr[:, 1].min()),
            int(poly_arr[:, 0].max()),
            int(poly_arr[:, 1].max()),
        )
        paragraphs.append({
            'indices': [li],
            'polys': [polys[li]],
            'texts': [texts[li]],
            'scores': [scores[li]],
            'merged_text': texts[li],
            'bbox': bbox,
            'bubble_bbox': None,
        })

    # y좌표 순 정렬
    paragraphs.sort(key=lambda p: (p['bbox'][1], p['bbox'][0]))
    return paragraphs


def save_paragraphs_json(img_path, paragraphs, out_dir, mask_pad=10):
    """
    paragraphs 정보를 JSON으로 저장한다.
    이후 Masking / Cleaning / Rendering / Blending 단계에서 불러와 사용.

    mask_bbox 결정 규칙:
      - bubble_bbox가 있으면 말풍선 전체를 마스크 영역으로 사용
      - 없으면 OCR bbox에 mask_pad 픽셀 패딩을 추가
    """
    img = Image.open(img_path)
    img_w, img_h = img.size

    data = {
        "image_path": os.path.abspath(img_path),
        "image_width": img_w,
        "image_height": img_h,
        "paragraphs": [],
    }

    for i, para in enumerate(paragraphs):
        x_min, y_min, x_max, y_max = para["bbox"]

        if para["bubble_bbox"] is not None:
            mx1, my1, mx2, my2 = [int(v) for v in para["bubble_bbox"]]
        else:
            mx1 = max(0, x_min - mask_pad)
            my1 = max(0, y_min - mask_pad)
            mx2 = min(img_w, x_max + mask_pad)
            my2 = min(img_h, y_max + mask_pad)

        data["paragraphs"].append({
            "id": i,
            "merged_text": para["merged_text"],
            "lines": [
                {
                    "text": t,
                    "score": round(float(s), 4),
                    "poly": [[int(pt[0]), int(pt[1])] for pt in poly],
                }
                for t, s, poly in zip(para["texts"], para["scores"], para["polys"])
            ],
            "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
            "bubble_bbox": [int(v) for v in para["bubble_bbox"]] if para["bubble_bbox"] else None,
            "mask_bbox": [mx1, my1, mx2, my2],
        })

    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, f"{img_basename}_paragraphs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"문단 정보 저장: {out_path}")
    return out_path


def draw_paragraph_boxes_on_ocr_result(ocr_res_img_path, paragraphs):
    """save_to_img()가 만든 2패널 이미지의 오른쪽 패널에 문단 빨간 박스를 추가한다."""
    img = Image.open(ocr_res_img_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x_offset = img.width // 2

    for i, para in enumerate(paragraphs):
        x_min, y_min, x_max, y_max = para['bbox']
        rx_min = x_offset + x_min
        rx_max = x_offset + x_max
        draw.rectangle([rx_min, y_min, rx_max, y_max], fill=(255, 0, 0, 30))
        draw.rectangle([rx_min, y_min, rx_max, y_max], outline=(255, 0, 0, 255), width=3)
        draw.text((rx_min + 4, y_min + 2), f"P{i+1}", fill=(255, 0, 0, 255))

    img = Image.alpha_composite(img, overlay).convert("RGB")
    img.save(ocr_res_img_path)
    print(f"문단 박스 추가 저장: {ocr_res_img_path}")


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR v2 - 말풍선 기반 문단 그룹핑")
    parser.add_argument('--det', type=str, default="PP-OCRv5_mobile_det")
    parser.add_argument('--rec', type=str, default="PP-OCRv5_mobile_rec")
    parser.add_argument('--img', type=str, default="./../../data/raw/images/example_image.png")
    parser.add_argument('--out', type=str, default="./../../output/v4")
    parser.add_argument('--lang', type=str, default="japan")
    parser.add_argument('--bubble-conf', type=float, default=0.5,
                        help="말풍선 검출 confidence 임계값")
    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # 1. 말풍선 검출 모델 로드
    print("말풍선 검출 모델 로드 중...")
    bubble_model = load_bubble_detector()

    # 2. OCR 모델 로드 (v4 fine-tuned)
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

    # 3. 말풍선 검출
    print(f"말풍선 검출 중: {args.img}")
    bubbles = detect_bubbles(bubble_model, args.img, conf_thresh=args.bubble_conf)
    print(f"검출된 말풍선: {len(bubbles)}개")

    # 4. OCR 실행
    print(f"OCR 분석 시작: {args.img}")
    result = ocr.predict(args.img)

    for res in result:
        # 기존 시각화
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

        # 말풍선 기반 문단 그룹핑
        paragraphs = group_lines_by_bubbles(polys, texts, scores, bubbles)

        # 콘솔 출력
        print("\n" + "=" * 50)
        print(f"인식 결과: {len(polys)}개 라인 -> {len(paragraphs)}개 문단")
        print("=" * 50)

        for i, para in enumerate(paragraphs):
            bubble_str = "말풍선" if para['bubble_bbox'] else "미할당"
            print(f"\n[문단 {i+1}] ({len(para['texts'])}줄, {bubble_str})")
            for text, score in zip(para['texts'], para['scores']):
                print(f"  [{score:.2f}] {text}")
            print(f"  => 합침: {para['merged_text']}")

        # 문단 정보 JSON 저장 (이후 파이프라인 단계용)
        save_paragraphs_json(args.img, paragraphs, args.out)

        # 2패널 이미지에 빨간 박스 그리기
        img_basename = os.path.splitext(os.path.basename(args.img))[0]
        ocr_res_img_path = os.path.join(args.out, f"{img_basename}_ocr_res_img.jpg")
        if os.path.exists(ocr_res_img_path):
            draw_paragraph_boxes_on_ocr_result(ocr_res_img_path, paragraphs)
        else:
            print(f"2패널 이미지를 찾을 수 없음: {ocr_res_img_path}")

    print(f"\n분석 완료! 결과 확인: {args.out}")


if __name__ == "__main__":
    main()
