# Image Captioning
- ViT for image encoder
- GPT-2 for caption decoder
- Seq2SeqTrainer을 이용한 finetuning
- Flickr8k dataset

```python
import os

import datasets
import pandas as pd
from PIL import Image
import multiprocessing as mp
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor
from transformers import AutoTokenizer, default_data_collator

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
```

```python
os.environ["WANDB_DISABLED"] = "true"

# 설정값을 저장하는 CFG 클래스 정의
class CFG: 
    # 이미지 인코더로 사용할 ViT 모델 이름 (Vision Transformer)
    ENCODER = "google/vit-base-patch16-224"
    # 텍스트 디코더로 사용할 GPT-2 모델 이름
    DECODER = "gpt2"
    # 캡션(주석) 파일 경로 (텍스트 파일)
    ANNOTATION_FILE = "/mnt/elice/dataset/captions.txt"
    # 이미지가 저장된 디렉토리 경로
    IMAGE_DIR = "/mnt/elice/dataset/Images"
    # 학습 배치 사이즈
    TRAIN_BATCH_SIZE = 8
    # 검증 배치 사이즈
    VAL_BATCH_SIZE = 8
    # 검증 에폭 수 (추가 평가 횟수)
    VAL_EPOCHS = 1
    # 학습률(Learning Rate)
    LR = 5e-5
    # 랜덤 시드 (재현성을 위해)
    SEED = 42
    # 최대 시퀀스 길이 (입력 토큰 수)
    MAX_LEN = 128
    # 요약문 길이 (생성 시 사용할 길이)
    SUMMARY_LEN = 20
    # 가중치 감소 (Weight Decay) 계수
    WEIGHT_DECAY = 0.01
    # 이미지 정규화를 위한 평균값 (ImageNet 기준)
    MEAN = (0.485, 0.456, 0.406)
    # 이미지 정규화를 위한 표준편차 (ImageNet 기준)
    STD = (0.229, 0.224, 0.225)
    # 전체 데이터 중 학습에 사용할 비율
    TRAIN_PCT = 0.95
    # 데이터 로딩에 사용할 프로세스(worker) 수 (현재 CPU 코어 수 사용)
    NUM_WORKERS = mp.cpu_count()
    # 전체 에폭 수
    EPOCHS = 3
    # 이미지 크기 (너비, 높이)
    IMG_SIZE = (224, 224)
    # 라벨 마스킹에 사용할 값 (토큰화 시 패딩에 해당하는 값)
    LABEL_MASK = -100
    # 생성 시 고려할 Top-K 후보 수
    TOP_K = 1000
    # 생성 시 고려할 Top-P (누적 확률) 값
    TOP_P = 0.95
```

### Caption 생성을 위한 Special Tokens

```python
# GPT-2의 토큰화를 위한 스페셜 토큰 추가 함수 정의
# 입력 토큰 리스트 앞뒤에 시작(BOS) 및 종료(EOS) 토큰을 추가합니다.
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs

# AutoTokenizer에 위에서 정의한 build_inputs_with_special_tokens 함수 할당
AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
```
![image](https://github.com/user-attachments/assets/14527c19-6309-4780-8a9c-365af895c798)

### Annotation dataframe

```python
df=  pd.read_csv(CFG.ANNOTATION_FILE)
train_df , val_df = train_test_split(df , test_size = 0.2)
df.head()
```

```python
# ViT 모델을 위한 feature extractor 초기화 (이미지 전처리 및 특징 추출)
feature_extractor = ViTFeatureExtractor.from_pretrained(CFG.ENCODER)

# GPT-2를 위한 AutoTokenizer 초기화 (텍스트 토큰화)
tokenizer = AutoTokenizer.from_pretrained(CFG.DECODER)

# 패딩 토큰을 알 수 없는 토큰(unk_token)으로 설정 (GPT-2는 기본적으로 pad_token이 없음)
tokenizer.pad_token = tokenizer.unk_token
```

### Dataset

```python
# 이미지와 캡션 데이터를 다루기 위한 커스텀 데이터셋 클래스 정의
class ImgDataset(Dataset):
    def __init__(self, df, root_dir, tokenizer, feature_extractor, transform=None):
        self.df = df                         # 캡션과 이미지 정보가 담긴 데이터프레임
        self.transform = transform           # 추가적인 이미지 변환(transforms) 적용 (옵션)
        self.root_dir = root_dir             # 이미지 파일들이 저장된 디렉토리
        self.tokenizer = tokenizer           # 텍스트 토큰화를 위한 토크나이저
        self.feature_extractor = feature_extractor   # 이미지 전처리 및 특징 추출 기능
        self.max_length = 50                 # 캡션 토큰 시퀀스의 최대 길이
        
    def __len__(self):
        # 데이터셋 내 샘플 개수 반환
        return len(self.df)
    
    def __getitem__(self, idx):
        # 인덱스에 해당하는 캡션과 이미지 파일명 추출
        caption = self.df.caption.iloc[idx]
        image = self.df.image.iloc[idx]
        # 이미지 파일의 전체 경로 생성
        img_path = os.path.join(self.root_dir, image)
        # 이미지 파일 열기 및 RGB 모드로 변환
        img = Image.open(img_path).convert("RGB")
        
        # 이미지 전처리: feature_extractor를 이용하여 픽셀 값을 텐서 형태로 변환
        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values
        
        # 캡션을 토크나이저로 인코딩: 최대 길이까지 패딩 적용
        captions = self.tokenizer(
            caption,
            padding='max_length',
            max_length=self.max_length
        ).input_ids
        
        # 패딩 토큰 부분은 학습 시 무시할 수 있도록 LABEL_MASK(-100) 값으로 대체
        captions = [token if token != self.tokenizer.pad_token_id else -100 for token in captions]
        
        # 최종 입력 딕셔너리 구성: 이미지 픽셀 값과 캡션 레이블(토큰) 반환
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(captions)}
        
        return encoding
```

## Model

### ViT Encoder  

![](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png)

### GPT-2 Decoder

![](https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Full_GPT_architecture.png/498px-Full_GPT_architecture.png)

```python
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(CFG.ENCODER, CFG.DECODER)
```

```python
# 모델의 디코더 관련 설정 값 업데이트
# 아래 설정들은 모델 생성 시 자동으로 생성된 config에 접근하여 값을 재설정하는 부분입니다.
model.config.decoder_start_token_id = tokenizer.cls_token_id  # 디코더 시작 토큰 ID 설정 (여기서는 CLS 토큰 사용)
model.config.pad_token_id = tokenizer.pad_token_id              # 패딩 토큰 ID 설정
model.config.vocab_size = model.config.decoder.vocab_size       # 디코더의 어휘 크기를 config에 반영
model.config.eos_token_id = tokenizer.sep_token_id              # 종료 토큰 ID 설정 (여기서는 SEP 토큰 사용)
model.config.decoder_start_token_id = tokenizer.bos_token_id     # 디코더 시작 토큰 ID를 BOS 토큰으로 재설정
model.config.max_length = 50                                   # 생성 최대 길이 설정
model.config.early_stopping = True                             # 조기 종료 옵션 활성화
model.config.no_repeat_ngram_size = 3                          # 반복된 n-gram 생성을 방지하기 위한 설정
model.config.length_penalty = 2.0                              # 길이 페널티 설정 (긴 문장 생성 억제)
model.config.num_beams = 4                                     # 빔 서치 시 탐색할 빔 수 설정
```

## Training

### Training Arguments

```python
# Seq2SeqTrainingArguments: 훈련 관련 하이퍼파라미터 및 설정값 정의
training_args = Seq2SeqTrainingArguments(
    output_dir='VIT_large_gpt2',                              # 모델 및 결과가 저장될 출력 디렉토리
    per_device_train_batch_size=CFG.TRAIN_BATCH_SIZE,         # 디바이스별 학습 배치 사이즈
    per_device_eval_batch_size=CFG.VAL_BATCH_SIZE,            # 디바이스별 검증 배치 사이즈
    predict_with_generate=True,                               # 생성 기반 평가 활성화
    evaluation_strategy="epoch",                              # 평가 전략을 에폭 단위로 설정
    do_train=True,                                            # 학습 수행 여부
    do_eval=True,                                             # 평가 수행 여부
    logging_steps=1024,                                       # 로깅 간격 (스텝 단위)
    save_steps=2048,                                          # 모델 저장 간격 (스텝 단위)
    warmup_steps=1024,                                        # warmup 단계의 스텝 수
    learning_rate=CFG.LR,                                     # 학습률 설정
    max_steps=1500,  # 전체 학습 스텝 수 (전체 학습 시 삭제 가능)
    num_train_epochs=CFG.EPOCHS,  # 학습 에폭 수
    overwrite_output_dir=True,                                # 출력 디렉토리 덮어쓰기 여부
    save_total_limit=1,                                       # 저장할 체크포인트 최대 개수
)
```

### Seq2Seq Trainer
```python
# Seq2SeqTrainer 인스턴스 생성: 모델, 토크나이저, 학습 인자, 데이터셋, 데이터 콜레이터 등을 지정
trainer = Seq2SeqTrainer(
    tokenizer=feature_extractor,         # 입력 데이터를 처리하기 위한 feature_extractor 사용
    model=model,                         # 사용할 모델
    args=training_args,                  # 학습 관련 설정 인자
    train_dataset=train_dataset,         # 학습 데이터셋
    eval_dataset=val_dataset,            # 검증 데이터셋
    data_collator=default_data_collator, # 데이터 배치(collation) 함수
)

# 학습 시작
trainer.train()
```

```python
trainer.save_model('VIT_large_gpt2')
```

```python
# 학습 후, 테스트 이미지 하나를 불러와서 열기 (RGB 모드)
img =  Image.open("/mnt/elice/dataset/Images/1001773457_577c3a7d70.jpg").convert("RGB")
img
```
![image](https://github.com/user-attachments/assets/0ac8fe2b-f7e6-495a-ad9c-ddfed418d533)

```python
# 테스트 이미지에 대해 feature_extractor를 적용하여 픽셀 값을 생성하고,
# 이를 모델의 generate 메서드에 전달하여 생성된 캡션을 디코딩 후 출력
generated_caption = tokenizer.decode(model.generate(feature_extractor(img, return_tensors="pt").pixel_values.to("cuda"))[0])

# 생성된 캡션의 앞 85글자를 시각적으로 강조하여 출력 (ANSI escape code 사용)
print('\033[96m' +generated_caption[:85]+ '\033[0m')
```
> <|endoftext|>A black and white dog is running through the grass . . . a white and bro

```python
img =  Image.open("/mnt/elice/dataset/Images/1000268201_693b08cb0e.jpg").convert("RGB")
img  
```
![image](https://github.com/user-attachments/assets/a4686624-b85a-46bf-a084-9023e800d748)

```python
generated_caption = tokenizer.decode(model.generate(feature_extractor(img, return_tensors="pt").pixel_values.to("cuda"))[0])
print('\033[96m' +generated_caption[:120]+ '\033[0m')
```
> <|endoftext|>A little girl in a pink dress stands in front of a house . . . and a little boy in a blue dress stands behi
