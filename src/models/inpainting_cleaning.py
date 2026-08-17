import os
import sys
import json
import argparse
import urllib.request
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

# 모델 경로·다운로드 URL·기본 파라미터는 src/config.py가 단일 출처다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def get_lama_model(model_dir, device):
    """LaMa 모델(TorchScript) 로드. 가중치 없으면 다운로드."""
    os.makedirs(model_dir, exist_ok=True)
    local_path = os.path.join(model_dir, config.LAMA_FILENAME)
    if not os.path.exists(local_path):
        print("LaMa 가중치 없음. 다운로드 중...")
        urllib.request.urlretrieve(config.LAMA_DOWNLOAD_URL, local_path)
        print(f"다운로드 완료: {local_path}")
    model = torch.jit.load(config.native_path(local_path), map_location=device)
    model.eval()
    return model


def pad_to_div8(img_arr, mask_arr):
    """H, W를 8의 배수로 패딩. LaMa 입력 요구사항."""
    h, w = img_arr.shape[:2]
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        img_arr = np.pad(img_arr, ((0, pad_h), (0, pad_w), (0, 0)))
        mask_arr = np.pad(mask_arr, ((0, pad_h), (0, pad_w)))
    return img_arr, mask_arr, h, w


def build_polygon_mask(size, polys, offset=(0, 0), dilate_px=4):
    ox, oy = offset
    mask_img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in polys:
        pts = [(pt[0] - ox, pt[1] - oy) for pt in poly]
        draw.polygon(pts, fill=255)
    if dilate_px > 0:
        mask_img = mask_img.filter(ImageFilter.MaxFilter(dilate_px * 2 + 1))
    return mask_img


def inpaint_region(model, image, mask_bbox, polys, context_pad, device, dilate_px=4):
    """
    말풍선 밖 텍스트(효과음 등)를 LaMa로 지움
    """
    x1, y1, x2, y2 = mask_bbox
    W, H = image.size

    cx1 = max(0, x1 - context_pad)
    cy1 = max(0, y1 - context_pad)
    cx2 = min(W, x2 + context_pad)
    cy2 = min(H, y2 + context_pad)

    crop_arr = np.array(image.crop((cx1, cy1, cx2, cy2)).convert("RGB"), dtype=np.float32) / 255.0
    mask_img = build_polygon_mask((cx2 - cx1, cy2 - cy1), polys, offset=(cx1, cy1), dilate_px=dilate_px)
    mask_arr = np.array(mask_img, dtype=np.float32) / 255.0

    img_padded, mask_padded, orig_h, orig_w = pad_to_div8(crop_arr, mask_arr)

    img_t = torch.from_numpy(img_padded).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask_padded).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t, mask_t)

    out_arr = output[0].permute(1, 2, 0).cpu().numpy()
    out_arr = np.clip(out_arr * 255, 0, 255).astype(np.uint8)[:orig_h, :orig_w]

    orig_crop_u8 = (crop_arr * 255).astype(np.uint8)
    mask_bool = mask_arr > 0.5
    composite = np.where(mask_bool[..., None], out_arr, orig_crop_u8)

    result = image.copy()
    result.paste(Image.fromarray(composite), (cx1, cy1))
    return result


def fill_region_flat(image, mask_bbox, polys, fill_color, dilate_px=1, clip_bbox=None, clip_inset=1):
    """
    말풍선 안 텍스트를 단색으로 채운다. 
    """
    x1, y1, x2, y2 = mask_bbox
    mask_img = build_polygon_mask((x2 - x1, y2 - y1), polys, offset=(x1, y1), dilate_px=dilate_px)

    if clip_bbox is not None:
        cx1, cy1, cx2, cy2 = clip_bbox
        cx1, cy1, cx2, cy2 = cx1 + clip_inset, cy1 + clip_inset, cx2 - clip_inset, cy2 - clip_inset
        local_cx1 = max(0, cx1 - x1)
        local_cy1 = max(0, cy1 - y1)
        local_cx2 = min(x2 - x1, cx2 - x1)
        local_cy2 = min(y2 - y1, cy2 - y1)

        clip_arr = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        if local_cx2 > local_cx1 and local_cy2 > local_cy1:
            clip_arr[int(local_cy1):int(local_cy2), int(local_cx1):int(local_cx2)] = 255

        mask_arr = np.minimum(np.array(mask_img), clip_arr)
        mask_img = Image.fromarray(mask_arr)

    solid = Image.new("RGB", (x2 - x1, y2 - y1), fill_color)
    result = image.copy()
    result.paste(solid, (x1, y1), mask_img)
    return result


def sample_fill_color(image, mask_bbox, polys, dilate_px=1, ring_px=15):
    """
    텍스트 폴리곤 근처에서 배경색을 뽑는다.
    """
    x1, y1, x2, y2 = mask_bbox
    W, H = image.size
    ex1 = max(0, x1 - ring_px)
    ey1 = max(0, y1 - ring_px)
    ex2 = min(W, x2 + ring_px)
    ey2 = min(H, y2 + ring_px)

    arr = np.array(image.crop((ex1, ey1, ex2, ey2)).convert("RGB"))
    inner = np.array(build_polygon_mask((ex2 - ex1, ey2 - ey1), polys, offset=(ex1, ey1), dilate_px=dilate_px)) > 0
    outer = np.array(build_polygon_mask((ex2 - ex1, ey2 - ey1), polys, offset=(ex1, ey1), dilate_px=dilate_px + ring_px)) > 0
    ring = outer & ~inner

    sample = arr[ring] if ring.sum() >= 10 else arr.reshape(-1, 3)

    # 링이 좁은(텍스트가 말풍선을 거의 꽉 채운) 경우, 검은 테두리 선까지 같이
    # 잡혀서 회색으로 뭉개질 수 있다. 밝기 상위 50%(=테두리/텍스트 잔여물이
    # 아닌 진짜 배경 쪽)만 남겨서 중앙값을 뽑으면 이런 오염에 훨씬 강해진다.
    brightness = sample.mean(axis=1)
    bright_sample = sample[brightness >= np.percentile(brightness, 50)]
    if len(bright_sample) >= 5:
        sample = bright_sample

    return tuple(int(v) for v in np.median(sample, axis=0))


def run_cleaning(json_path, out_dir, model_dir, context_pad, device_str,
                  out_name=None, dilate_px=config.LAMA_DILATE_PX,
                  fill_dilate_px=config.FILL_DILATE_PX):
    device = torch.device(device_str)
    model = get_lama_model(model_dir, device)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image = Image.open(data["image_path"]).convert("RGB")
    print(f"이미지 로드: {data['image_path']}")
    print(f"{len(data['paragraphs'])}개 영역 처리 시작...")

    for para in data["paragraphs"]:
        polys = [line["poly"] for line in para["lines"]]

        if para.get("bubble_bbox"):
            # 말풍선 안 텍스트: 단색 채우기
            fill_color = sample_fill_color(image, para["mask_bbox"], polys, fill_dilate_px)
            image = fill_region_flat(image, para["mask_bbox"], polys, fill_color, fill_dilate_px,
                                      clip_bbox=para["bubble_bbox"])
            print(f"  [문단 {para['id']}] 단색 채우기 완료 - color: {fill_color}")
        else:
            # 말풍선 밖 텍스트(효과음 등): LaMa inpainting
            image = inpaint_region(model, image, para["mask_bbox"], polys, context_pad, device, dilate_px)
            print(f"  [문단 {para['id']}] LaMa 완료 - mask_bbox: {para['mask_bbox']}")

    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(data["image_path"]))[0]
    out_path = os.path.join(out_dir, out_name or config.cleaned_image(basename))
    image.save(out_path)
    print(f"\n저장 완료: {out_path}")


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="LaMa inpainting - 텍스트 영역 제거 및 배경 복원")
    parser.add_argument("--json", type=str,
                        default=os.path.join(config.OCR_OUTPUT_DIR,
                                             config.paragraphs_json(config.SAMPLE_IMAGE_NAME)),
                        help="paragraphs JSON 경로 (using_layout_parsing.py 출력)")
    parser.add_argument("--out", type=str, default=config.CLEANED_OUTPUT_DIR,
                        help="결과 이미지 저장 폴더")
    parser.add_argument("--model-dir", type=str, default=config.LAMA_DIR,
                        help="LaMa 가중치 폴더 (없으면 자동 다운로드)")
    parser.add_argument("--context-pad", type=int, default=config.LAMA_CONTEXT_PAD,
                        help=f"말풍선 밖 텍스트 inpainting 시 마스크 주변 context 픽셀 "
                             f"(기본값: {config.LAMA_CONTEXT_PAD})")
    parser.add_argument("--device", type=str, default=default_device(),
                        choices=["cpu", "cuda", "mps"],
                        help="연산 디바이스 (기본값: cuda > mps > cpu 자동 감지)")
    parser.add_argument("--out-name", type=str, default=None,
                        help="저장할 파일 이름 (기본값: {이미지명}_cleaned.png)")
    parser.add_argument("--dilate-px", type=int, default=config.LAMA_DILATE_PX,
                        help=f"말풍선 밖 텍스트(LaMa) 폴리곤 팽창 픽셀 "
                             f"(기본값: {config.LAMA_DILATE_PX})")
    parser.add_argument("--fill-dilate-px", type=int, default=config.FILL_DILATE_PX,
                        help=f"말풍선 안 텍스트(단색 채우기) 폴리곤 팽창 픽셀 "
                             f"(기본값: {config.FILL_DILATE_PX})")
    args = parser.parse_args()

    run_cleaning(args.json, args.out, args.model_dir, args.context_pad, args.device,
                 out_name=args.out_name, dilate_px=args.dilate_px, fill_dilate_px=args.fill_dilate_px)


if __name__ == "__main__":
    main()
