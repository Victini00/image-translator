"""
인식 모델 학습 실행기 - 최소 에포크 + 조기 종료.

PaddleOCR의 train.py에는 조기 종료가 없어서 epoch_num을 끝까지 돈다. 이 스크립트는
train.py를 자식 프로세스로 띄워 출력을 로그 파일로 옮기면서 평가 결과를 읽고,
최소 에포크를 채운 뒤 검증 정확도가 정해진 횟수만큼 연속으로 오르지 않으면
학습을 멈춘다. 가장 좋았던 가중치는 PaddleOCR이 best_accuracy로 따로 저장하므로
중간에 멈춰도 잃지 않는다.

사용법 (저장소 루트에서):
    python src/models/train_rec_early_stop.py --config config/PP-OCRv5_mobile_rec_v5.yml
    # 끊긴 학습 이어서 돌리기
    python src/models/train_rec_early_stop.py --config config/PP-OCRv5_mobile_rec_v5.yml \\
        -o Global.checkpoints=../../../models/ocr/PP-OCRv5_mobile_rec_jp_fine_tuned_v5/latest
"""

import os
import re
import sys
import ctypes
import argparse
import subprocess

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "external", "PaddleOCR", "tools")

STEP_RE = re.compile(r"epoch: \[(\d+)/(\d+)\], global_step: (\d+)")
METRIC_RE = re.compile(r"cur metric, acc: ([0-9.]+)")


def keep_awake():
    if sys.platform == "win32":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def kill_tree(proc):
    """train.py와 데이터 로더 워커까지 한꺼번에 끝낸다."""
    if sys.platform == "win32":
        subprocess.call(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc.terminate()


def save_dir_of(config_path, overrides):
    """로그를 둘 곳 = 학습 설정의 save_model_dir (train.py 기준 상대경로)."""
    for o in overrides:
        if o.startswith("Global.save_model_dir="):
            rel = o.split("=", 1)[1]
            break
    else:
        with open(config_path, encoding="utf-8") as f:
            rel = yaml.safe_load(f)["Global"]["save_model_dir"]
    return os.path.normpath(os.path.join(TOOLS, rel))


def main():
    ap = argparse.ArgumentParser(description="PaddleOCR 인식 학습 + 조기 종료")
    ap.add_argument("--config", required=True, help="학습 설정 yml (저장소 루트 기준)")
    ap.add_argument("--min-epochs", type=int, default=5,
                    help="이 에포크 수를 다 채우기 전에는 멈추지 않는다")
    ap.add_argument("--patience", type=int, default=6,
                    help="최고 정확도가 이 횟수의 평가 동안 안 오르면 멈춘다. "
                         "평가가 1만 스텝마다라 6회가 약 1 에포크(62,500스텝)")
    ap.add_argument("--min-delta", type=float, default=0.001,
                    help="이만큼 넘게 올라야 개선으로 친다(0.001 = 0.1%%p, 평가셋 2만 장 중 20장)")
    ap.add_argument("-o", "--override", nargs="*", default=[],
                    help="train.py -o 로 넘길 설정 덮어쓰기")
    args = ap.parse_args()

    config_path = os.path.join(ROOT, args.config)
    out_dir = save_dir_of(config_path, args.override)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "stdout.log")

    keep_awake()
    # train.py를 직접 부르지 않고 train_rec_entry.py를 거친다(백본 풀링을 추론과
    # 맞추는 패치가 들어 있다. 이유는 그 파일 설명 참고).
    entry = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_rec_entry.py")
    cmd = [sys.executable, entry, "-c", os.path.relpath(config_path, TOOLS)]
    if args.override:
        cmd += ["-o"] + args.override
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, cwd=TOOLS, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1)

    best, since_best, epoch, n_eval = -1.0, 0, 0, 0
    stop_reason = None
    with open(log_path, "a", encoding="utf-8") as log:
        def note(msg):
            log.write(f"[early-stop] {msg}\n")
            log.flush()

        note(f"시작: 최소 {args.min_epochs} 에포크, patience {args.patience}회, "
             f"min_delta {args.min_delta}")
        for line in proc.stdout:
            log.write(line)
            log.flush()

            m = STEP_RE.search(line)
            if m:
                epoch = int(m.group(1))
                if stop_reason:            # 평가·저장이 끝나고 학습이 다시 돈 시점
                    note(f"학습 중단: {stop_reason}")
                    kill_tree(proc)
                    break
                continue

            m = METRIC_RE.search(line)
            if not m:
                continue
            acc = float(m.group(1))
            n_eval += 1
            if acc > best + args.min_delta:
                best, since_best = acc, 0
            else:
                since_best += 1
            note(f"평가 {n_eval}회 (에포크 {epoch}): acc {acc:.4f} / 최고 {best:.4f} / "
                 f"연속 개선 없음 {since_best}회")
            if epoch > args.min_epochs and since_best >= args.patience:
                stop_reason = (f"{args.min_epochs} 에포크 이후 평가 {since_best}회 연속 "
                               f"개선 없음 (최고 acc {best:.4f})")

    code = proc.wait()
    with open(os.path.join(out_dir, "exit_status.txt"), "w", encoding="utf-8") as f:
        f.write(f"early_stop: {stop_reason}\n" if stop_reason else f"exit_code: {code}\n")
        f.write(f"last_epoch: {epoch}\nevals: {n_eval}\nbest_acc: {best:.4f}\n")


if __name__ == "__main__":
    main()
