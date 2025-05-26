Stable Video Diffusion
Stable Video Diffusion 은 이미지를 이용해 고해상도(576x1024)의 2~4초의 영상을 생성할 수 있는 생성 AI 모델입니다.

이 모델은 논문 Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets 에 기반하고 있습니다.

Stable Video Diffusion 은 이미지에서 영상을 생성하는 모델로, 텍스트에서 이미지를 생성하는 Stable Diffusion에 기반한 영상 생성 모델입니다.

이번 실습에서는 Stable Video Diffusion을 사용하는 과정을 실습해 보겠습니다.

1. Stable Video Diffusion 파이프라인
파이프라인 셋업
Stable Video Diffusion은 파이프라인을 구성하여 이미지를 입력해 영상을 생성할 수 있습니다.

몇 줄의 간단한 코드를 이용하면 영상 생성 파이프라인을 불러올 수 있습니다.

먼저, 필요한 라이브러리를 불러옵니다

```python
import random

import PIL
import torch
from diffusers import (
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    StableVideoDiffusionPipeline,
)
from diffusers.utils import export_to_video, load_image
from IPython.display import Video, display
```

StableVideoDiffusionPipeline는 이미지를 입력하면 짧은 영상를 생성해주는 end-to-end 파이프라인입니다.

```python
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16, #모델 연산에 사용할 데이터 타입
    variant="fp16",
)
pipe = pipe.to("cuda") # 병렬연산을 하기위하여 하는 명령

```
