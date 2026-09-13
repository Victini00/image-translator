"""
학습 이미지에 실제 만화 스캔의 조건을 입히는 변환들.
"""

import numpy as np
from PIL import Image, ImageFilter


def screentone(img, rng, period=None, strength=None):
    """만화 특유의 망점(스크린톤) 배경을 깔아준다.

    글자보다 밝은 회색 격자를 만들고 원본과 어두운 쪽을 취한다. 글자 획은 그대로
    남고 배경만 톤이 깔린 것처럼 보인다.
    """
    period = period or rng.integers(2, 5)
    strength = strength if strength is not None else rng.integers(20, 60)

    a = np.array(img.convert("L")).astype(np.int16)
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dots = (((xx // period + yy // period) % 2) == 0).astype(np.int16) * strength
    return Image.fromarray(np.clip(np.minimum(a, 255 - dots), 0, 255).astype(np.uint8)).convert("RGB")


def text_gray(img, rng, low=60, high=130):
    """새까만 글자 대신 회색조 글자. 톤 위에 인쇄된 글자의 낮은 대비를 흉내낸다."""
    level = int(rng.integers(low, high))
    a = np.array(img.convert("L")).astype(np.int16)
    a = np.where(a < 200, level + a * (255 - level) // 255, a)
    return Image.fromarray(a.astype(np.uint8)).convert("RGB")


def blur(img, rng, lo=0.3, hi=1.2):
    """인쇄·스캔으로 뭉개진 획."""
    return img.filter(ImageFilter.GaussianBlur(float(rng.uniform(lo, hi))))


def skew(img, rng, max_deg=4.0):
    """검출 박스가 살짝 틀어져 잡히는 경우."""
    deg = float(rng.uniform(-max_deg, max_deg))
    return img.rotate(deg, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def downscale(img, rng, lo=0.35, hi=1.0):
    """저해상도 페이지. 줄였다가 되돌려 정보 손실만 남긴다.

    평가셋 만화가 폭 436~508px이라 글자 획이 몇 픽셀밖에 안 된다. 고해상도만
    학습하면 이런 페이지에서 무너진다.
    """
    f = float(rng.uniform(lo, hi))
    if f >= 0.99:
        return img
    small = (max(1, int(img.width * f)), max(1, int(img.height * f)))
    return img.resize(small, Image.LANCZOS).resize(img.size, Image.LANCZOS)


def jpeg_artifacts(img, rng, lo=45, hi=92):
    """JPEG 압축 흔적. 스캔본은 대부분 손실 압축을 거친다."""
    import io
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=int(rng.integers(lo, hi)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def noise(img, rng, sigma=None):
    """스캔 잡티."""
    sigma = sigma if sigma is not None else float(rng.uniform(2, 9))
    a = np.array(img.convert("L")).astype(np.float32)
    a = a + rng.normal(0, sigma, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("RGB")
