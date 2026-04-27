import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

LAMA_REPO_ID = "smartywu/big-lama"
LAMA_FILENAME = "big-lama.pt"


def get_lama_model(model_dir, device):
    """LaMa 모델 로드. 가중치 없으면 HuggingFace에서 다운로드."""
    os.makedirs(model_dir, exist_ok=True)
    local_path = os.path.join(model_dir, LAMA_FILENAME)
    if not os.path.exists(local_path):
        print("LaMa 가중치 없음. HuggingFace에서 다운로드 중...")
        local_path = hf_hub_download(
            repo_id=LAMA_REPO_ID,
            filename=LAMA_FILENAME,
            local_dir=model_dir,
        )
        print(f"다운로드 완료: {local_path}")
    model = torch.jit.load(local_path, map_location=device)
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


def inpaint_region(model, image, mask_bbox, context_pad, device):
    """
    mask_bbox 영역을 context_pad 픽셀 여유를 두고 잘라 inpainting 후 원본에 붙여넣기.
    context_pad: 마스크 주변 배경 패턴을 모델에 제공하기 위한 여유 픽셀
    """
    x1, y1, x2, y2 = mask_bbox
    W, H = image.size

    cx1 = max(0, x1 - context_pad)
    cy1 = max(0, y1 - context_pad)
    cx2 = min(W, x2 + context_pad)
    cy2 = min(H, y2 + context_pad)

    crop_arr = np.array(image.crop((cx1, cy1, cx2, cy2)).convert("RGB"), dtype=np.float32) / 255.0

    mask_arr = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.float32)
    mask_arr[y1 - cy1: y2 - cy1, x1 - cx1: x2 - cx1] = 1.0

    img_padded, mask_padded, orig_h, orig_w = pad_to_div8(crop_arr, mask_arr)

    img_t = torch.from_numpy(img_padded).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask_padded).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t, mask_t)

    out_arr = output[0].permute(1, 2, 0).cpu().numpy()
    out_arr = np.clip(out_arr * 255, 0, 255).astype(np.uint8)[:orig_h, :orig_w]

    result = image.copy()
    result.paste(Image.fromarray(out_arr), (cx1, cy1))
    return result


def run_cleaning(json_path, out_dir, model_dir, context_pad, device_str):
    device = torch.device(device_str)
    model = get_lama_model(model_dir, device)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image = Image.open(data["image_path"]).convert("RGB")
    print(f"이미지 로드: {data['image_path']}")
    print(f"{len(data['paragraphs'])}개 영역 inpainting 시작...")

    for para in data["paragraphs"]:
        image = inpaint_region(model, image, para["mask_bbox"], context_pad, device)
        print(f"  [문단 {para['id']}] 완료 - mask_bbox: {para['mask_bbox']}")

    os.makedirs(out_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(data["image_path"]))[0]
    out_path = os.path.join(out_dir, f"{basename}_cleaned.png")
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
                        default="./../../output/v4/shirobako_paragraphs.json",
                        help="paragraphs JSON 경로 (using_layout_parsing.py 출력)")
    parser.add_argument("--out", type=str, default="./../../output/inpainting/cleaned",
                        help="결과 이미지 저장 폴더")
    parser.add_argument("--model-dir", type=str, default="./../../models/inpainting/LaMa",
                        help="LaMa 가중치 폴더 (없으면 자동 다운로드)")
    parser.add_argument("--context-pad", type=int, default=30,
                        help="inpainting 시 마스크 주변 context 픽셀 (기본값: 30)")
    parser.add_argument("--device", type=str, default=default_device(),
                        choices=["cpu", "cuda", "mps"],
                        help="연산 디바이스 (기본값: cuda > mps > cpu 자동 감지)")
    args = parser.parse_args()

    run_cleaning(args.json, args.out, args.model_dir, args.context_pad, args.device)


if __name__ == "__main__":
    main()
