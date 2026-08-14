# LaMa 인페인팅 모델

말풍선 **밖**의 글자(효과음 등)를 지울 때 쓰는 인페인팅 모델을 두는 곳입니다.
(말풍선 안쪽은 배경이 단색이라 인페인팅 없이 단색으로 채웁니다.)

## 필요한 파일

```
models/inpainting/LaMa/big-lama.pt        약 196MB
```

## 받는 방법

**따로 받지 않아도 됩니다.** `inpainting_cleaning.py`가 처음 실행될 때 이 폴더에
없으면 자동으로 내려받습니다.

수동으로 받으려면:

```bash
curl -L -o big-lama.pt \
  https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt
```

원본 모델은 [advimman/lama](https://github.com/advimman/lama)이고, 위 링크는
torch.jit 형태로 변환된 [Sanster/models](https://github.com/Sanster/models) 배포본입니다
(원본 체크포인트는 별도 설정 파일이 필요해서 이쪽을 씁니다).
