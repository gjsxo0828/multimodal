# Stable Video Diffusion
[Stable Video Diffusion](https://stability.ai/news/stable-video-diffusion-open-ai-video-model) 은 이미지를 이용해 고해상도(576x1024)의 2~4초의 영상을 생성할 수 있는 생성 AI 모델입니다.

이 모델은 논문 [Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets](https://arxiv.org/abs/2311.15127) 에 기반하고 있습니다.

Stable Video Diffusion 은 이미지에서 영상을 생성하는 모델로, 텍스트에서 이미지를 생성하는 [Stable Diffusion](https://github.com/CompVis/stable-diffusion)에 기반한 영상 생성 모델입니다.

이번 실습에서는 Stable Video Diffusion을 사용하는 과정을 실습해 보겠습니다.

---

## 1. Stable Video Diffusion 파이프라인

### 파이프라인 셋업
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

### 도우미 함수    
영상을 생성하는데에 도움을 주는 함수를 정의합니다.

```python
def generate_video(image_url: str, video_file: str, seed: int = 42):
    image = load_image(image_url)
    image = image.resize((1024, 576)) #이미지 너비(가로), 높이(세로) 정의

    if seed == -1:
        seed = random.randint(0, 2**32) #정수를 랜덤하게 추출, 매번 다른 결과를 얻고 싶을 때는 -1 값 사용

    generator = torch.manual_seed(seed)  # 시드를 고정
    # 초기 노이즈 패턴, 이노이징 과정에서 노이즈 제거 패턴 등이 달라지면 다른 이미지가 생성됨

    frames = pipe(
        image,
        decode_chunk_size=2, # 디코딩(생성) 과정에서 한번에 생성할 프레임 수
        generator=generator, #시드 고정
        num_frames=10, # 생성할 비디오 프레임 수
        num_inference_steps=10, #프레임 당 추론 단계 수 (클수록 고품질, 저속)
    ).frames[0] #생성된 프레임만 접근

    export_to_video(frames, "tmp.mp4", fps=7)
        # export_to_video : 이미지 프레임 시퀀스 -> 비디오
        # 'tmp.mp4' : dlatlfh wjwkdgkf qleldh vkdlf dlfma(경로)
        # fps : 초당 재생시킬 프레임 수
        
    !ffmpeg -y -hide_banner -loglevel error -i tmp.mp4 {video_file}
        # ffmpeg : 멀티미디어(비디오, 오디오, 이미지) 처리에 사용되는 오픈소스 소프트웨어로, 동영상 변환, 
        # tmp.mp4를 최종 비디오 파일명으로{video_file} 변환하거나 복사
        # -y : 기존에 같은 이름의 파일이 있으면 덮어쓰기
        # -hide_banner : ffmpeg 실행 시 불피룡한 배너 정보 숨김
        # -loglevel error : 에러 메시지만 출력(불필요한 로그 숨김)
        # -i tmp.mp4 : 입력 파일로 tmp.mp4를 사용
        # {video_file} : 저장할 파일명
   
    
    print("seed", seed)
    return Video(video_file)
```

## 2. 영상 생성하기
### 로켓 발사 장면
영상을 생성하려면 파이프라인에 생성하려는 영상의 기반이 되는 이미지를 불러와 입력해야 합니다.
간단하게 아래의 이미지를 이용해 영상을 생성해 보겠습니다.
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png" height="50%" width="50%">

```python
generate_video(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png",
    "rocket.mp4",
)
```

## 3. 다른 이미지로 영상 생성하기
다른 이미지를 불러와 영상을 생성해 보겠습니다.
<img src="https://cdn-api.elice.io/api-attachment/attachment/99c03dcce847455d826d16bd396c0af3/image.png" height="50%" width="50%">

```python
generate_video(
    "https://cdn-api.elice.io/api-attachment/attachment/99c03dcce847455d826d16bd396c0af3/image.png",
    "rabbit.mp4",
)
```

## 4. 이미지 생성 모델과 함께 사용하기
이 모델의 기반이 되는 모델은 텍스트에서 이미지를 생성하는 [Stable Diffusion](https://github.com/CompVis/stable-diffusion)이라는 모델입니다.
이 모델을 이용하여 텍스트에서 이미지를 생성하고 그 이미지로 영상을 생성하는 과정을 진행하겠습니다.

### Stable Diffusion 파이프라인 셋업
```python
scheduler = EulerDiscreteScheduler.from_pretrained("/mnt/elice/dataset/stable-diffusion-2-1-base", subfolder="scheduler")
# scheduler : diffusion모델에서 노이즈를 점진적으로 제거하는 과정을 제어하는 핵심 컴포넌트
# from_pretrained : 사전 학습된 모델 로딩
# 첫번째 인자 : 모델이 저장된 디렉터리 경로
# subfolder : 해당 이름의 하위 스케줄러 설정/가중치 파일을 로드합니다.

image_pipe = StableDiffusionPipeline.from_pretrained("/mnt/elice/dataset/stable-diffusion-2-1-base", scheduler=scheduler, torch_dtype=torch.float16)
#torch_dtype : 모델연산에 사용할 데이터
image_pipe = image_pipe.to("cuda")
# gpu 연산
image_pipe.enable_attention_slicing()
# 어텐션 연산을 여러번에 나눠서 실행하여 GPU 메모리 사용량을 줄임 (추론 시간 10% 늘어남)
```

### 텍스트에서 이미지를 생성하는 도우미 함수
이미지를 생성한 다음 그 이미지로 영상을 생성하는 도우미 함수입니다.

```python
def generate_text2video(prompt: str, video_file: str, seed=-1):
    if seed == -1:
        seed = random.randint(0, 2**32)

    generator = torch.manual_seed(seed)
    # 이미지 생성
    image = image_pipe(
        prompt, guidance_scale=7.5, generator=generator, width=1024, height=576
    ).images[0]
    # guidance_scale : 텍스트 조건에 얼마나 강하게 맞출지 조절하는 하이퍼파라미터
    print("생성된 이미지")
    display(image)

    # 영상 생성
    frames = pipe(
        image,
        decode_chunk_size=2,
        generator=generator,
        num_frames=12,
        num_inference_steps=10,
    ).frames[0]

    export_to_video(frames, "tmp.mp4", fps=7)
    !ffmpeg -y -hide_banner -loglevel error -i tmp.mp4 {video_file}

    print("seed", seed)
    return Video(video_file)
```

이제 이 도우미 함수를 이용해 텍스트에서 영상을 생성해 보겠습니다.

### 우주 비행사가 말을 타고 있는 영상
```python
generate_text2video(
    "a photograph of an astronaut riding a horse", "horse.mp4", seed=1972088094
)
```

### 도시의 야경
```python
generate_text2video(
    "An aerial view of a city at night, long exposure, instagram contest",
    "city.mp4",
    seed=3815670667,
)
```
