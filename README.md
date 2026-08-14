# image-translator

일본어 만화 이미지를 한국어로 번역해서, **원본 글자를 지우고 그 자리에 번역문을 그려 넣는** 파이프라인입니다.

말풍선을 검출해 문단 단위로 묶고 → 글자를 읽고 → 번역하고 → 원본 글자를 지우고 → 말풍선 안/밖에 맞는 글꼴과 크기로 번역문을 배치합니다.
결과를 브라우저에서 직접 고칠 수 있는 편집 UI도 포함되어 있습니다.

> 개인 학습용 프로젝트입니다. 번역 대상 이미지의 저작권은 사용자 본인이 확인해야 합니다.

---

## 목차

- [동작 방식](#동작-방식)
- [두 가지 실행 경로](#두-가지-실행-경로)
- [설치](#설치)
- [모델 준비](#모델-준비)
- [사용법](#사용법)
- [웹 UI](#웹-ui)
- [프로젝트 구조](#프로젝트-구조)
- [사용한 외부 프로젝트](#사용한-외부-프로젝트)
- [사용한 모델](#사용한-모델)
- [알려진 한계](#알려진-한계)

---

## 동작 방식

파이프라인은 크게 네 단계입니다.

| 단계 | 하는 일 | 사용 모델 |
|---|---|---|
| **Masking** | 글자 영역 검출 + 말풍선 검출 → 문단 단위로 그룹핑 | PP-OCRv5 (detection/recognition), YOLOv8 말풍선 검출기 |
| **Recognition / Translation** | 일본어 원문을 읽고 한국어로 번역 | PP-OCRv5 파인튜닝 모델 + hell0ks LoRA, **또는** Gemini API |
| **Cleaning** | 원본 글자 지우기 | LaMa (말풍선 밖), 단색 채우기 (말풍선 안) |
| **Rendering** | 번역문을 박스에 맞춰 자동 줄바꿈·크기 조절해 그리기 | Pillow |

**문단 그룹핑**: 검출된 글자 줄의 중심점이 어느 말풍선 안에 있는지로 묶습니다. 말풍선에 속하지 않는 줄(효과음 등)은 각각 독립 문단이 됩니다.

**지우는 방식이 안/밖에 따라 다릅니다**: 말풍선 안은 배경이 단색이라 단색으로 채우고, 말풍선 밖은 배경 그림이 살아 있어야 하므로 LaMa 인페인팅을 씁니다. 마스크는 말풍선 전체가 아니라 **글자 줄 폴리곤 기준**으로 만듭니다(말풍선 테두리까지 지워지는 것을 막기 위해).

**글자 배치**: 박스를 넘지 않는 선에서 `줄 수가 가장 적은 것` → `그중 글자가 가장 큰 것` 순으로 고릅니다. 단어(어절)는 쪼개지 않는 것을 원칙으로 하고, 최소 크기까지 줄여도 안 들어갈 때만 글자 단위로 나눕니다.

---

## 두 가지 실행 경로

같은 파이프라인이지만 **글자를 읽고 번역하는 부분**만 다릅니다.

### `image_translator_v1.py` — 로컬 전용

PaddleOCR로 읽고 hell0ks(7B) + LoRA로 번역합니다. 외부 API가 필요 없지만 GPU가 필요하고(7B 4bit 양자화), 세로쓰기 특수문자를 오독하는 문제가 있습니다.

### `image_translator_v1_using_gemini.py` — 하이브리드 (권장)

글자 **위치 검출**은 PaddleOCR + YOLO를 그대로 쓰고, **읽기와 번역**만 Gemini API에 맡깁니다.

이렇게 나눈 이유는, 위치 검출은 로컬 모델이 픽셀 단위로 정확한 반면 **글자 인식 품질은 API 쪽이 확실히 낫기 때문**입니다. 실제로 로컬 인식이 반복적으로 틀리던 아래 사례들이 해결됩니다.

| 원본 | 로컬(PaddleOCR) | Gemini |
|---|---|---|
| `アニメーター` | `ア二xIタI` | `アニメーター` |
| `おい望月〜` | `おい望月り` | `おい望月〜` |
| `ごめ〜ん` | `ごめりん` | `ごめ〜ん` |
| `今日という` | `介日という` | `今日という` |

세로쓰기에서 `ー`(장음)나 `〜`가 90도 회전되어 보이는 것을 로컬 모델이 다른 글자로 읽는 문제인데, 학습 데이터에 세로쓰기 특수문자 배치가 없는 것이 원인으로 보입니다(→ [알려진 한계](#알려진-한계)).

API 호출은 **이미지 1장당 1회**입니다(문단 전체를 한 번에 보내 인식과 번역을 같이 처리). 호출이 실패하면 자동으로 로컬 모델로 폴백됩니다.

---

## 설치

### 1. 저장소와 외부 프로젝트

```bash
git clone https://github.com/Victini00/image-translator.git
cd image-translator

# 외부 프로젝트 (자세한 내용은 아래 '사용한 외부 프로젝트' 참고)
git clone https://github.com/PaddlePaddle/PaddleOCR.git external/PaddleOCR
```

### 2. 파이썬 환경

Python 3.10 기준입니다.

```bash
conda create -n image-translator python=3.10
conda activate image-translator
pip install -r requirements.txt
```

`requirements.txt`의 아래 3개는 일반 PyPI 패키지가 아니라 별도 설치가 필요합니다(파일 상단 주석 참고).

- `torch` / `torchvision` — CUDA 빌드(`+cu128`)
- `paddleocr` — GitHub 특정 커밋에서 직접 설치
- `paddlepaddle-gpu` — PaddlePaddle 개발 빌드 wheel

### 3. API 키 (Gemini 경로를 쓸 경우)

프로젝트 루트에 `.env` 파일을 만들고 한 줄 넣습니다. 키는 [Google AI Studio](https://aistudio.google.com)에서 카드 등록 없이 발급받을 수 있습니다.

```
GEMINI_API_KEY=발급받은_키
```

`.env`는 `.gitignore`에 걸려 있어 저장소에 올라가지 않습니다.

---

## 모델 준비

| 모델 | 위치 | 받는 방법 |
|---|---|---|
| 일본어 인식 파인튜닝 (v4) | `models/ocr/PP-OCRv5_mobile_rec_jp_fine_tuned_v4/complete_model/` | **저장소에 포함** (약 17MB) |
| PP-OCRv5 detection | `~/.paddlex/official_models/` | 첫 실행 시 자동 다운로드 |
| 말풍선 검출 (YOLOv8) | HuggingFace 캐시 | 첫 실행 시 자동 다운로드 |
| LaMa 인페인팅 | `models/inpainting/LaMa/big-lama.pt` | 첫 실행 시 자동 다운로드 (약 196MB) |
| hell0ks 번역 모델 | HuggingFace 캐시 | 첫 실행 시 자동 다운로드 (7B) |
| hell0ks LoRA (직접 학습) | `models/translation/hell0ks_ja-ko-vn-7b-v1/lora/v1/` | 저장소 미포함 — 아래 참고 |

학습 체크포인트(`.pdopt`, `.pdparams`), 학습 로그 등은 수백 MB라 저장소에 포함하지 않았습니다. 추론에는 `complete_model/`만 있으면 됩니다.

> **LoRA 어댑터**는 직접 학습한 것이라 저장소에 포함되지 않습니다. 없으면 `--lora-path ""`로 base 모델만 쓰거나, Gemini 경로를 사용하세요.

---

## 사용법

필수 인자는 이미지 경로 하나뿐이고, 나머지는 각 단계 스크립트의 기본값이 적용됩니다.

```bash
cd src

# 하이브리드 (권장)
python image_translator_v1_using_gemini.py --img ../data/raw/images/test2.png

# 로컬 전용
python image_translator_v1.py --img ../data/raw/images/test2.png
```

결과는 `output/pipeline_v1_gemini/` (또는 `output/pipeline_v1/`)에 저장됩니다.

| 파일 | 내용 |
|---|---|
| `{이름}_paragraphs.json` | 문단별 좌표·원문·번역문 |
| `{이름}_cleaned.png` | 원본 글자를 지운 이미지 |
| `{이름}_rendered.png` | **최종 결과** |

### 자주 쓰는 옵션

```bash
# 인식 신뢰도 임계값 (기본 0.4)
--rec-score-thresh 0.5

# 말풍선 검출 임계값 (기본 0.5)
--bubble-conf 0.4

# 글꼴/크기
--interior-font "폰트경로.otf" --max-font-size 40

# Gemini 모델 (기본 gemini-flash-latest)
--gemini-model gemini-3.6-flash
```

각 단계를 따로 실행할 수도 있습니다(`src/models/` 안의 스크립트를 직접 호출). 모든 스크립트가 `--help`를 지원합니다.

---

## 웹 UI

번역 결과를 브라우저에서 직접 고칠 수 있습니다.

```bash
python src/web/app.py       # http://127.0.0.1:5000
```

- 이미지 업로드, 모델 선택, **단계별 진행률** 표시
- Gemini API 키를 UI에서 입력 → `.env`에 저장되어 다음부터 자동 사용
- API 호출이 실패해 로컬 모델로 폴백되면 **원인(한도 초과 / 서버 혼잡 / 키 오류)을 구분해서 안내**
- **번역문 직접 수정** — 자유 줄바꿈, 박스 위치·크기 조절, 글자 크기 직접 지정
- **덧칠 브러시** — 검출이 놓쳐 남은 원본 글자를 색으로 덮기 (스포이드/굵기/되돌리기)
- 편집 후 재렌더링은 **번역을 다시 하지 않고** 그리기 단계만 다시 실행

편집 화면의 미리보기는 최종 결과와 **같은 글꼴·같은 조판**으로 그려집니다. 글꼴 크기와 줄바꿈, 각 줄의 좌표를 서버가 실제 렌더링과 동일한 함수로 계산해 내려주고, 브라우저는 그 좌표대로 canvas에 그립니다(픽셀 비교 시 세로 오차 0, 가로 1px 이내).

---

## 프로젝트 구조

```
src/
  image_translator_v1.py              전체 파이프라인 (로컬 전용)
  image_translator_v1_using_gemini.py 전체 파이프라인 (Gemini 하이브리드)
  models/
    paddleocr_demo_v2_using_layout_parsing.py   Masking (검출 + 문단 그룹핑)
    recognition_translation_gemini.py           Gemini 인식 + 번역
    inpainting_cleaning.py                      Cleaning (LaMa / 단색 채우기)
    inpainting_rendering.py                     Rendering (번역 + 그리기)
    render_layout.py                            글꼴·배치 규칙 (웹 UI와 공유)
    translation_hell0ks*.py                     로컬 번역 모델 / LoRA 학습
    experiments/                                모델 비교용 실험 스크립트
  data/                               학습 데이터 생성·전처리 스크립트
  web/                                편집 웹 UI (Flask + 브라우저 편집기)
config/                               학습 설정 (yml)
models/                               모델 파일 (대부분 gitignore)
external/                             외부 프로젝트 (gitignore)
data/, output/                        데이터·결과물 (gitignore)
```

---

## 사용한 외부 프로젝트

아래 저장소들을 `external/` 아래에 clone해서 사용했습니다. 용량이 커서 저장소에는 포함하지 않았습니다.

| 프로젝트 | 저장소 | 사용한 커밋 | 용도 |
|---|---|---|---|
| **PaddleOCR** | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | `8a4dd540e1` (2026-02-04) | 일본어 인식 모델 파인튜닝 학습 코드. `config/PP-OCRv5_mobile_rec.yml`이 이 학습 스크립트 기준 설정입니다. |
| **TextRecognitionDataGenerator** | [Victini00/TextRecognitionDataGenerator](https://github.com/Victini00/TextRecognitionDataGenerator) (fork) | `dd9d7d3` | 인식 모델 학습용 합성 이미지 생성. 원본은 [Belval/TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator). |
| **parseq** | [baudm/parseq](https://github.com/baudm/parseq) | `1902db0` | 인식 모델 후보 검토용. 최종 파이프라인에는 사용하지 않았습니다. |

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git external/PaddleOCR
git clone https://github.com/Victini00/TextRecognitionDataGenerator.git external/TextRecognitionDataGenerator
git clone https://github.com/baudm/parseq.git external/parseq
```

---

## 사용한 모델

| 용도 | 모델 | 비고 |
|---|---|---|
| 글자 영역 검출 | `PP-OCRv5_server_det` | mobile과 실측 비교 후 채택 (검출 누락이 더 적음) |
| 글자 인식 | `PP-OCRv5_mobile_rec` 일본어 파인튜닝 (v4) | 직접 학습 |
| 말풍선 검출 | [ogkalu/comic-speech-bubble-detector-yolov8m](https://huggingface.co/ogkalu/comic-speech-bubble-detector-yolov8m) | |
| 인페인팅 | [LaMa (big-lama)](https://github.com/advimman/lama) | [Sanster/models](https://github.com/Sanster/models) 배포본 사용 |
| 번역 (로컬) | [hell0ks/ja-ko-vn-7b-v1](https://huggingface.co/hell0ks/ja-ko-vn-7b-v1) + 직접 학습한 LoRA | 4bit 양자화 |
| 인식 + 번역 (API) | Google Gemini (`gemini-flash-latest`) | |

### 사용 글꼴

| 위치 | 글꼴 |
|---|---|
| 말풍선 안 | Gmarket Sans Medium |
| 말풍선 밖 (효과음) | HY피오피M |

> 웹 UI는 미리보기를 위해 글꼴 파일을 브라우저로 전송합니다. 이 프로젝트를 **공개 서비스로 배포한다면** 글꼴의 웹 사용/재배포 라이선스를 반드시 확인하고, 필요하면 라이선스가 명확한 글꼴로 교체해야 합니다.

---

## 알려진 한계

- **세로쓰기 특수문자 오독** — 로컬 인식 모델이 세로로 눕혀진 `ー`, `〜`를 알파벳 등으로 잘못 읽습니다. 학습 데이터 생성 시 세로쓰기 방향의 특수문자 회전이 반영되지 않은 것이 원인으로 보입니다. Gemini 경로에서는 발생하지 않습니다.
- **검출 누락** — 특이한 글꼴, 획이 적은 글자, 후리가나가 조밀한 영역은 검출 단계에서 놓치는 경우가 있습니다. 웹 UI의 덧칠 기능으로 보정할 수 있습니다.
- **OCR 교정 단계 보류** — 번역 전 원문을 교정하는 단계를 시도했으나(hell0ks, Qwen2.5-7B-Instruct), 지시 무시·중국어 간체 혼입 문제로 파이프라인에 넣지 않았습니다. 코드는 `src/models/translation_correction.py`에 참고용으로 남아 있습니다.
- **Gemini 무료 티어 한도** — 공개 서비스로 쓰기에는 일일 요청 한도가 부족합니다.
- **웹 UI는 1인용 로컬 도구 전제** — 개발 서버, 메모리 기반 작업 상태, 단일 API 키 등 다중 사용자를 고려하지 않았습니다.

---

## 향후 계획

- 인식 모델 재학습 — 실제 만화 스타일의 글꼴·배치를 반영한 데이터셋, 세로쓰기 특수문자 회전이 적용된 데이터셋 추가
