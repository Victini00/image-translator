"""
PaddleOCR train.py를 돌리되, 백본 마지막 풀링을 추론과 같게 맞춘다.

PPLCNetV3(rec)의 마지막 풀링이 학습과 추론에서 다르다.
    학습:  F.adaptive_avg_pool2d(x, [1, 40])   # 입력 너비와 무관하게 40칸
    추론:  F.avg_pool2d(x, [3, 2])             # 너비 8px당 1칸
공식 설정처럼 입력 너비가 320이면 둘 다 40칸이라 같지만, 이 프로젝트는 한 줄에
12글자가 들어가서 너비를 640으로 쓴다. 그러면 학습은 40칸(16px당 1칸), 추론은
80칸(8px당 1칸)이 되어 어긋난다.

PaddleOCR 원본은 건드리지 않고 여기서 forward만 바꿔 끼운다.

이 파일은 train_rec_early_stop.py가 external/PaddleOCR/tools 에서 실행한다.
"""

import os
import runpy
import sys

import paddle.nn.functional as F

sys.path.insert(0, os.getcwd())                       # external/PaddleOCR/tools
sys.path.insert(0, os.path.dirname(os.getcwd()))      # external/PaddleOCR

import ppocr.modeling.backbones.rec_lcnetv3 as lcnetv3  # noqa: E402


def _forward(self, x):
    """원본 forward에서 마지막 풀링만 추론과 같은 방식으로 고정한다."""
    x = self.conv1(x)
    x = self.blocks2(x)
    x = self.blocks3(x)
    x = self.blocks4(x)
    x = self.blocks5(x)
    x = self.blocks6(x)
    if self.det:                      # 검출용 백본은 중간 출력들을 돌려줘야 한다
        raise RuntimeError("이 패치는 인식(rec) 학습 전용입니다")
    return F.avg_pool2d(x, [3, 2])


lcnetv3.PPLCNetV3.forward = _forward
print("[patch] PPLCNetV3 마지막 풀링을 추론과 동일하게(avg_pool 3x2) 고정", flush=True)

runpy.run_path("train.py", run_name="__main__")
