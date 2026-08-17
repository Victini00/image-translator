import sys
import os
import argparse
import subprocess

# Windows 콘솔 기본 인코딩(cp949)이 일본어/한국어 혼용 출력 시 깨지는 것 방지
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

"""
<사용법>

전체 파이프라인(Masking -> Cleaning -> Rendering)을 이미지 한 장으로 한 번에 돌리는
통합 스크립트. 새 로직을 짜지 않고, 이미 검증된 각 단계 스크립트
(paddleocr_demo_v2_using_layout_parsing.py / inpainting_cleaning.py /
inpainting_rendering.py)를 그대로 순서대로 subprocess로 호출해서 체이닝만 한다.

필수 인자는 --img(원본 이미지 경로) 하나

예시:
    python image_translator_v1.py --img ./../data/raw/images/test2.png
    python image_translator_v1.py --img ./../data/raw/images/test2.png --rec-score-thresh 0.5
"""

# 경로·기본값·산출물 파일명 규칙은 src/config.py가 단일 출처다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

MODELS_DIR = config.CODE_MODELS_DIR

MASKING_SCRIPT = os.path.join(MODELS_DIR, "paddleocr_demo_v2_using_layout_parsing.py")
CLEANING_SCRIPT = os.path.join(MODELS_DIR, "inpainting_cleaning.py")
RENDERING_SCRIPT = os.path.join(MODELS_DIR, "inpainting_rendering.py")


def run_step(step_name, script_path, extra_args):
    """각 단계 스크립트를 subprocess로 실행한다. 실패하면 바로 중단."""
    cmd = [sys.executable, script_path] + extra_args
    print(f"\n{'=' * 60}\n[{step_name}] 실행\n{'=' * 60}")
    result = subprocess.run(cmd, cwd=MODELS_DIR)
    if result.returncode != 0:
        print(f"\n에러: [{step_name}] 단계 실패 (exit code {result.returncode})")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="이미지 번역 파이프라인 v1 - Masking -> Cleaning -> Rendering 한 번에 실행"
    )
    parser.add_argument("--img", type=str, required=True,
                        help="번역할 원본 이미지 경로 (필수)")
    parser.add_argument("--out-dir", type=str, default=config.PIPELINE_V1_OUTPUT_DIR,
                        help="결과물(paragraphs json, cleaned/rendered 이미지) 저장 폴더")
    parser.add_argument("--name", type=str, default=None,
                        help="결과 파일 이름에 쓸 베이스 이름 (기본값: 원본 이미지 파일명)")

    # ---- Masking 옵션 (안 주면 masking 스크립트 기본값 사용) ----
    parser.add_argument("--det", type=str, default=None,
                        help=f"detection 모델 (기본값: {config.DETECTION_MODEL})")
    parser.add_argument("--rec-score-thresh", type=float, default=None,
                        help=f"OCR 인식 신뢰도 임계값 (기본값: {config.REC_SCORE_THRESH})")
    parser.add_argument("--bubble-conf", type=float, default=None,
                        help=f"말풍선 검출 confidence 임계값 (기본값: {config.BUBBLE_CONF_THRESH})")
    parser.add_argument("--mask-pad", type=int, default=None,
                        help=f"mask_bbox에 추가할 여유 픽셀 (기본값: {config.MASK_PAD})")

    # ---- Cleaning 옵션 ----
    parser.add_argument("--context-pad", type=int, default=None,
                        help=f"말풍선 밖 텍스트 LaMa inpainting 시 context 픽셀 (기본값: {config.LAMA_CONTEXT_PAD})")
    parser.add_argument("--dilate-px", type=int, default=None,
                        help=f"말풍선 밖 텍스트(LaMa) 폴리곤 팽창 픽셀 (기본값: {config.LAMA_DILATE_PX})")
    parser.add_argument("--fill-dilate-px", type=int, default=None,
                        help=f"말풍선 안 텍스트(단색 채우기) 폴리곤 팽창 픽셀 (기본값: {config.FILL_DILATE_PX})")

    # ---- Rendering 옵션 ----
    parser.add_argument("--interior-font", type=str, default=None,
                        help="말풍선 안쪽 텍스트 폰트 경로 (기본값: Gmarket Sans Medium)")
    parser.add_argument("--exterior-font", type=str, default=None,
                        help="말풍선 밖 텍스트 폰트 경로 (기본값: HY POP M)")
    parser.add_argument("--max-font-size", type=int, default=None,
                        help=f"기본값: {config.MAX_FONT_SIZE}")
    parser.add_argument("--min-font-size", type=int, default=None,
                        help=f"기본값: {config.MIN_FONT_SIZE}")
    parser.add_argument("--stroke-width", type=int, default=None,
                        help=f"기본값: {config.STROKE_WIDTH}")
    parser.add_argument("--bubble-inset-ratio", type=float, default=None,
                        help=f"기본값: {config.BUBBLE_INSET_RATIO}")
    parser.add_argument("--model", type=str, default=None,
                        help=f"번역용 모델 (기본값: {config.TRANSLATION_MODEL})")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="번역 LoRA 어댑터 경로 (빈 문자열이면 LoRA 없이 base만 사용)")

    # ---- 공통 옵션 ----
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                        help="연산 디바이스 (기본값: cuda > mps > cpu 자동 감지)")

    args = parser.parse_args()

    img_path = os.path.abspath(args.img)
    if not os.path.exists(img_path):
        print(f"에러: 이미지를 찾을 수 없음: {img_path}")
        sys.exit(1)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    base_name = args.name or os.path.splitext(os.path.basename(img_path))[0]

    json_name = config.paragraphs_json(base_name)
    json_path = os.path.join(out_dir, json_name)
    cleaned_name = config.cleaned_image(base_name)
    cleaned_path = os.path.join(out_dir, cleaned_name)
    rendered_path = os.path.join(out_dir, config.rendered_image(base_name))
    translated_json_path = os.path.join(out_dir, config.translated_json(base_name))

    print(f"이미지: {img_path}")
    print(f"결과 저장 폴더: {out_dir}")

    # ---------- 1. Masking ----------
    masking_args = ["--img", img_path, "--out", out_dir, "--json-name", json_name]
    if args.det:
        masking_args += ["--det", args.det]
    if args.rec_score_thresh is not None:
        masking_args += ["--rec-score-thresh", str(args.rec_score_thresh)]
    if args.bubble_conf is not None:
        masking_args += ["--bubble-conf", str(args.bubble_conf)]
    if args.mask_pad is not None:
        masking_args += ["--mask-pad", str(args.mask_pad)]
    run_step("1/3 Masking", MASKING_SCRIPT, masking_args)

    # ---------- 2. Cleaning ----------
    cleaning_args = ["--json", json_path, "--out", out_dir, "--out-name", cleaned_name]
    if args.context_pad is not None:
        cleaning_args += ["--context-pad", str(args.context_pad)]
    if args.dilate_px is not None:
        cleaning_args += ["--dilate-px", str(args.dilate_px)]
    if args.fill_dilate_px is not None:
        cleaning_args += ["--fill-dilate-px", str(args.fill_dilate_px)]
    if args.device:
        cleaning_args += ["--device", args.device]
    run_step("2/3 Cleaning", CLEANING_SCRIPT, cleaning_args)

    # ---------- 3. Rendering ----------
    rendering_args = [
        "--json", json_path,
        "--cleaned-image", cleaned_path,
        "--out", rendered_path,
        "--save-json", translated_json_path,
    ]
    if args.interior_font:
        rendering_args += ["--interior-font", args.interior_font]
    if args.exterior_font:
        rendering_args += ["--exterior-font", args.exterior_font]
    if args.max_font_size is not None:
        rendering_args += ["--max-font-size", str(args.max_font_size)]
    if args.min_font_size is not None:
        rendering_args += ["--min-font-size", str(args.min_font_size)]
    if args.stroke_width is not None:
        rendering_args += ["--stroke-width", str(args.stroke_width)]
    if args.bubble_inset_ratio is not None:
        rendering_args += ["--bubble-inset-ratio", str(args.bubble_inset_ratio)]
    if args.model:
        rendering_args += ["--model", args.model]
    if args.lora_path is not None:  # 빈 문자열도 의미 있는 값(LoRA 비활성화)이라 truthy 체크 안 함
        rendering_args += ["--lora-path", args.lora_path]
    if args.device:
        rendering_args += ["--device", args.device]
    run_step("3/3 Rendering", RENDERING_SCRIPT, rendering_args)

    print(f"\n{'=' * 60}")
    print("전체 파이프라인 완료!")
    print(f"  - 문단 정보: {json_path}")
    print(f"  - 클리닝 결과: {cleaned_path}")
    print(f"  - 최종 렌더링 결과: {rendered_path}")
    print(f"  - 번역문 포함 JSON: {translated_json_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
