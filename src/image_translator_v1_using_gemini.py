import sys
import os
import argparse
import subprocess

# Windows 콘솔 기본 인코딩(cp949)이 일본어/한국어 혼용 출력 시 깨지는 것 방지
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

"""
<사용법>

image_translator_v1.py의 하이브리드 버전. detection(말풍선/문단 위치)은 그대로
PaddleOCR+YOLO를 쓰지만, recognition(글자 읽기)과 번역은 hell0ks 대신 Google
Gemini API(비전)가 대신한다. Masking -> Gemini 인식+번역 -> Cleaning -> Rendering
순서로 4단계를 subprocess로 체이닝만 한다 (각 단계 스크립트는 그대로 재사용).
Gemini 호출은 문단 수와 무관하게 이미지당 2회(인식 1회 + 번역 1회)만 나가서
무료 티어 일일 한도를 크게 아낄 수 있다.

사전 준비: 프로젝트 루트 .env 파일에 GEMINI_API_KEY=... 설정 필요
(https://aistudio.google.com 에서 카드 등록 없이 무료 발급 가능).

필수 인자는 --img(원본 이미지 경로) 하나뿐이고, 그 외 옵션은 안 주면 각 단계
스크립트 자체의 기본값이 적용된다.

예시:
    python image_translator_v1_using_gemini.py --img ./../data/raw/images/test2.png
"""

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(SRC_DIR, "models")

MASKING_SCRIPT = os.path.join(MODELS_DIR, "paddleocr_demo_v2_using_layout_parsing.py")
GEMINI_SCRIPT = os.path.join(MODELS_DIR, "recognition_translation_gemini.py")
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
        description="이미지 번역 파이프라인 v1 (Gemini 하이브리드) - Masking -> Gemini 인식/번역 -> Cleaning -> Rendering"
    )
    parser.add_argument("--img", type=str, required=True,
                        help="번역할 원본 이미지 경로 (필수)")
    parser.add_argument("--out-dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "output", "pipeline_v1_gemini"),
                        help="결과물(paragraphs json, cleaned/rendered 이미지) 저장 폴더")
    parser.add_argument("--name", type=str, default=None,
                        help="결과 파일 이름에 쓸 베이스 이름 (기본값: 원본 이미지 파일명)")

    # ---- Masking 옵션 (안 주면 masking 스크립트 기본값 사용) ----
    parser.add_argument("--det", type=str, default=None,
                        help="detection 모델 (기본값: PP-OCRv5_server_det)")
    parser.add_argument("--rec-score-thresh", type=float, default=None,
                        help="OCR 인식 신뢰도 임계값 (기본값: 0.4) - detection/문단 그룹핑용, "
                             "실제 읽기는 Gemini가 다시 하지만 너무 낮추면 노이즈 문단이 생길 수 있음")
    parser.add_argument("--bubble-conf", type=float, default=None,
                        help="말풍선 검출 confidence 임계값 (기본값: 0.5)")
    parser.add_argument("--mask-pad", type=int, default=None,
                        help="mask_bbox에 추가할 여유 픽셀 (기본값: 10)")

    # ---- Gemini 인식+번역 옵션 ----
    parser.add_argument("--gemini-model", type=str, default=None,
                        help="Gemini 모델 (기본값: gemini-flash-latest)")
    parser.add_argument("--crop-pad", type=int, default=None,
                        help="문단 crop 시 여유 픽셀 (기본값: 20)")
    parser.add_argument("--request-interval", type=float, default=None,
                        help="Gemini 인식 호출과 번역 호출 사이 대기 시간(초) (기본값: 2.0)")

    # ---- Cleaning 옵션 ----
    parser.add_argument("--context-pad", type=int, default=None,
                        help="말풍선 밖 텍스트 LaMa inpainting 시 context 픽셀 (기본값: 30)")
    parser.add_argument("--dilate-px", type=int, default=None,
                        help="말풍선 밖 텍스트(LaMa) 폴리곤 팽창 픽셀 (기본값: 4)")
    parser.add_argument("--fill-dilate-px", type=int, default=None,
                        help="말풍선 안 텍스트(단색 채우기) 폴리곤 팽창 픽셀 (기본값: 1)")

    # ---- Rendering 옵션 (번역 모델 옵션은 Gemini 단계가 실패한 문단의 폴백용) ----
    parser.add_argument("--interior-font", type=str, default=None,
                        help="말풍선 안쪽 텍스트 폰트 경로 (기본값: Gmarket Sans Medium)")
    parser.add_argument("--exterior-font", type=str, default=None,
                        help="말풍선 밖 텍스트 폰트 경로 (기본값: HY POP M)")
    parser.add_argument("--max-font-size", type=int, default=None, help="기본값: 36")
    parser.add_argument("--min-font-size", type=int, default=None, help="기본값: 12")
    parser.add_argument("--stroke-width", type=int, default=None, help="기본값: 2")
    parser.add_argument("--bubble-inset-ratio", type=float, default=None, help="기본값: 0.15")

    # ---- 공통 옵션 ----
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"],
                        help="연산 디바이스 (기본값: cuda > mps > cpu 자동 감지) - Cleaning/폴백 번역용")

    args = parser.parse_args()

    img_path = os.path.abspath(args.img)
    if not os.path.exists(img_path):
        print(f"에러: 이미지를 찾을 수 없음: {img_path}")
        sys.exit(1)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    base_name = args.name or os.path.splitext(os.path.basename(img_path))[0]

    json_name = f"{base_name}_paragraphs.json"
    json_path = os.path.join(out_dir, json_name)
    cleaned_name = f"{base_name}_cleaned.png"
    cleaned_path = os.path.join(out_dir, cleaned_name)
    rendered_path = os.path.join(out_dir, f"{base_name}_rendered.png")
    translated_json_path = os.path.join(out_dir, f"{base_name}_paragraphs_translated.json")

    print(f"이미지: {img_path}")
    print(f"결과 저장 폴더: {out_dir}")

    # ---------- 1. Masking (detection + 말풍선 그룹핑, PaddleOCR+YOLO 그대로) ----------
    masking_args = ["--img", img_path, "--out", out_dir, "--json-name", json_name]
    if args.det:
        masking_args += ["--det", args.det]
    if args.rec_score_thresh is not None:
        masking_args += ["--rec-score-thresh", str(args.rec_score_thresh)]
    if args.bubble_conf is not None:
        masking_args += ["--bubble-conf", str(args.bubble_conf)]
    if args.mask_pad is not None:
        masking_args += ["--mask-pad", str(args.mask_pad)]
    run_step("1/4 Masking", MASKING_SCRIPT, masking_args)

    # ---------- 2. Gemini 인식 + 번역 (recognition+translation 대체) ----------
    gemini_args = ["--json", json_path, "--img", img_path, "--out", json_path]
    if args.gemini_model:
        gemini_args += ["--model", args.gemini_model]
    if args.crop_pad is not None:
        gemini_args += ["--crop-pad", str(args.crop_pad)]
    if args.request_interval is not None:
        gemini_args += ["--request-interval", str(args.request_interval)]
    run_step("2/4 Gemini 인식+번역", GEMINI_SCRIPT, gemini_args)

    # ---------- 3. Cleaning ----------
    cleaning_args = ["--json", json_path, "--out", out_dir, "--out-name", cleaned_name]
    if args.context_pad is not None:
        cleaning_args += ["--context-pad", str(args.context_pad)]
    if args.dilate_px is not None:
        cleaning_args += ["--dilate-px", str(args.dilate_px)]
    if args.fill_dilate_px is not None:
        cleaning_args += ["--fill-dilate-px", str(args.fill_dilate_px)]
    if args.device:
        cleaning_args += ["--device", args.device]
    run_step("3/4 Cleaning", CLEANING_SCRIPT, cleaning_args)

    # ---------- 4. Rendering (--use-existing-translation: Gemini 번역 그대로 사용) ----------
    rendering_args = [
        "--json", json_path,
        "--cleaned-image", cleaned_path,
        "--out", rendered_path,
        "--save-json", translated_json_path,
        "--use-existing-translation",
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
    if args.device:
        rendering_args += ["--device", args.device]
    run_step("4/4 Rendering", RENDERING_SCRIPT, rendering_args)

    print(f"\n{'=' * 60}")
    print("전체 파이프라인(Gemini 하이브리드) 완료!")
    print(f"  - 문단 정보(Gemini 인식/번역 포함): {json_path}")
    print(f"  - 클리닝 결과: {cleaned_path}")
    print(f"  - 최종 렌더링 결과: {rendered_path}")
    print(f"  - 번역문 포함 JSON: {translated_json_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
