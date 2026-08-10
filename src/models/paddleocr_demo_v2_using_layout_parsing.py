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


def _rotate_bbox_to_ocr_frame(bbox, orig_w, orig_h, angle):
    """
    원본 이미지 좌표계의 bbox를, PaddleOCR가 문서 방향 보정을 위해 실제로
    회전 처리한 좌표계로 변환한다. angle은 PaddleOCR(use_doc_orientation_classify)가
    감지한 보정 각도(0/90/180/270, 반시계 기준)로, 이미지마다 다를 수 있다
    (shirobako.jpg=90, 다른 이미지=270인 경우를 실측으로 확인함 - 하드코딩 금지).
    """
    x1, y1, x2, y2 = bbox
    if angle == 0:
        return (x1, y1, x2, y2)
    if angle == 90:
        return (y1, orig_w - x2, y2, orig_w - x1)
    if angle == 180:
        return (orig_w - x2, orig_h - y2, orig_w - x1, orig_h - y1)
    if angle == 270:
        return (orig_h - y2, x1, orig_h - y1, x2)
    raise ValueError(f"지원하지 않는 회전 각도: {angle}")


def detect_bubbles(model, img_path, conf_thresh=0.5, angle=90):
    """
    이미지에서 말풍선 영역을 검출하여 bbox 리스트 반환.
    PaddleOCR가 실제로 처리한 회전 좌표계(angle)에 맞춰 버블 좌표도 변환한다.
    """
    from PIL import Image as PILImage
    orig_w, orig_h = PILImage.open(img_path).size

    results = model(img_path, verbose=False)
    bubbles = []
    for r in results:
        for i in range(len(r.boxes)):
            if r.boxes.conf[i] >= conf_thresh:
                bbox = tuple(r.boxes.xyxy[i].tolist())
                bubbles.append(_rotate_bbox_to_ocr_frame(bbox, orig_w, orig_h, angle))
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


def _rotated_bbox_to_original(bbox, orig_w, orig_h, angle):
    """
    OCR가 실제로 처리한 회전 좌표계(angle)에 있는 bbox를 원본 이미지
    좌표계로 되돌린다. OCR 결과(polys/bbox)도 전부 이 회전 좌표계에 있으므로,
    원본 이미지 파일에 그대로 크롭/렌더링하려면 반드시 이 변환을 거쳐야 한다.
    angle은 이미지마다 다를 수 있어(0/90/180/270) 고정값으로 가정하면 안 된다.
    """
    x1, y1, x2, y2 = bbox
    if angle == 0:
        return (x1, y1, x2, y2)
    if angle == 90:
        return (orig_w - y2, x1, orig_w - y1, x2)
    if angle == 180:
        return (orig_w - x2, orig_h - y2, orig_w - x1, orig_h - y1)
    if angle == 270:
        return (y1, orig_h - x2, y2, orig_h - x1)
    raise ValueError(f"지원하지 않는 회전 각도: {angle}")


def _rotated_point_to_original(pt, orig_w, orig_h, angle):
    x, y = pt
    if angle == 0:
        return (x, y)
    if angle == 90:
        return (orig_w - y, x)
    if angle == 180:
        return (orig_w - x, orig_h - y)
    if angle == 270:
        return (y, orig_h - x)
    raise ValueError(f"지원하지 않는 회전 각도: {angle}")


def get_doc_angle(res):
    """OCR 결과에서 문서 방향 보정 각도(0/90/180/270)를 추출한다. 못 찾으면 0."""
    dp_res = res.get('doc_preprocessor_res') if hasattr(res, 'get') else getattr(res, 'doc_preprocessor_res', None)
    if dp_res is None:
        return 0
    angle = dp_res.get('angle') if hasattr(dp_res, 'get') else getattr(dp_res, 'angle', None)
    return int(angle) if angle is not None else 0


def save_paragraphs_json(img_path, paragraphs, out_dir, angle, mask_pad=10, json_name=None):
    """
    paragraphs 정보를 JSON으로 저장한다.
    이후 Masking / Cleaning / Rendering / Blending 단계에서 불러와 사용.

    OCR/버블 좌표는 전부 detect_bubbles()가 쓰는 회전 좌표계에 있으므로,
    저장 직전에 원본 이미지 좌표계로 변환한다 (안 그러면 Cleaning 단계가
    원본 이미지 파일에 엉뚱한 영역을 크롭하게 됨).

    mask_bbox 결정 규칙:
      - 항상 OCR bbox(텍스트 실제 영역)에 mask_pad 픽셀 패딩만 추가한다.
      - 말풍선 전체를 지우면 LaMa가 말풍선 테두리/흰 배경까지 주변 배경으로
        지워버려서(bubble_bbox를 mask로 쓰던 예전 방식) 렌더링 때 말풍선을
        통째로 다시 그려야 하는 문제가 생김. bubble_bbox는 참고용으로만 저장.
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
        x_min, y_min, x_max, y_max = _rotated_bbox_to_original(para["bbox"], img_w, img_h, angle)
        bubble_orig = (
            _rotated_bbox_to_original(para["bubble_bbox"], img_w, img_h, angle)
            if para["bubble_bbox"] is not None else None
        )

        mx1 = max(0, int(x_min - mask_pad))
        my1 = max(0, int(y_min - mask_pad))
        mx2 = min(img_w, int(x_max + mask_pad))
        my2 = min(img_h, int(y_max + mask_pad))

        data["paragraphs"].append({
            "id": i,
            "merged_text": para["merged_text"],
            "lines": [
                {
                    "text": t,
                    "score": round(float(s), 4),
                    "poly": [
                        [int(v) for v in _rotated_point_to_original(pt, img_w, img_h, angle)]
                        for pt in poly
                    ],
                }
                for t, s, poly in zip(para["texts"], para["scores"], para["polys"])
            ],
            "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
            "bubble_bbox": [int(v) for v in bubble_orig] if bubble_orig else None,
            "mask_bbox": [mx1, my1, mx2, my2],
        })

    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, json_name or f"{img_basename}_paragraphs.json")
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
    parser.add_argument('--det', type=str, default="PP-OCRv5_server_det",
                        help="텍스트 detection 모델 (기본값: server - mobile 대비 실측 비교 결과 "
                             "detection 누락이 더 적어서 채택. 단, 아주 작은 반복 의성어(パチパチ류)는 "
                             "mobile보다 못 잡는 경우도 있었음)")
    parser.add_argument('--rec', type=str, default="PP-OCRv5_mobile_rec")
    parser.add_argument('--img', type=str, default="./../../data/raw/images/example_image.png")
    parser.add_argument('--out', type=str, default="./../../output/ocr/v4")
    parser.add_argument('--lang', type=str, default="japan")
    parser.add_argument('--bubble-conf', type=float, default=0.5,
                        help="말풍선 검출 confidence 임계값")
    parser.add_argument('--mask-pad', type=int, default=10,
                        help="mask_bbox에 추가할 여유 픽셀 (기본값: 10)")
    parser.add_argument('--json-name', type=str, default=None,
                        help="저장할 paragraphs json 파일 이름 (기본값: {이미지명}_paragraphs.json)")
    parser.add_argument('--rec-score-thresh', type=float, default=0.4,
                        help="OCR 인식 신뢰도 임계값 - 이보다 낮은 줄은 애초에 결과에서 제외 "
                             "(배경 패턴/장식 요소를 글자로 잘못 인식하는 노이즈는 대부분 신뢰도가 "
                             "0에 가까워서 이걸로 걸러짐. shirobako/test1 실측 기준 노이즈 최고점 "
                             "0.28, 실제 텍스트 최저점 0.544였어서 그 사이로 0.4 채택, 기본값: 0.4)")
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
        text_recognition_model_dir="./../../models/ocr/PP-OCRv5_mobile_rec_jp_fine_tuned_v4/complete_model",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        lang=args.lang,
    )

    if not os.path.exists(args.img):
        print(f"에러: '{args.img}' 찾을 수 없음")
        return

    # 3. OCR 실행 (문서 회전 보정 각도는 OCR을 돌려봐야 알 수 있어서, 말풍선
    # 좌표 변환에 쓸 각도도 이 결과를 먼저 받은 후에 계산한다)
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

        # 인식 신뢰도(score)가 낮은 줄 제거 (배경 패턴/장식 요소를 글자로
        # 잘못 인식하는 노이즈는 대부분 score가 0에 가까움).
        # PaddleOCR 생성자의 text_rec_score_thresh를 쓰면 rec_texts/rec_scores만
        # 걸러지고 dt_polys는 그대로 남아 길이가 안 맞아 IndexError가 나므로,
        # 여기서 polys/texts/scores를 같이 zip해서 걸러야 세 배열 길이가 유지됨.
        n_before = len(texts)
        if args.rec_score_thresh > 0:
            filtered = [
                (p, t, s) for p, t, s in zip(polys, texts, scores)
                if s >= args.rec_score_thresh
            ]
            polys = [f[0] for f in filtered]
            texts = [f[1] for f in filtered]
            scores = [f[2] for f in filtered]
        print(f"인식 신뢰도 {args.rec_score_thresh} 미만 제거: {n_before}줄 -> {len(texts)}줄")

        # 4. 말풍선 검출 (이 이미지에 실제로 적용된 회전 각도에 맞춰 좌표 변환)
        angle = get_doc_angle(res)
        print(f"문서 회전 보정 각도: {angle}도")
        print(f"말풍선 검출 중: {args.img}")
        bubbles = detect_bubbles(bubble_model, args.img, conf_thresh=args.bubble_conf, angle=angle)
        print(f"검출된 말풍선: {len(bubbles)}개")

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
        save_paragraphs_json(args.img, paragraphs, args.out, angle, mask_pad=args.mask_pad, json_name=args.json_name)

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
