# CLIP(Contrastive Language Image Pretraining)

```python
import os
import json
import cv2
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A

import timm
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
```

### Config

```python
class CFG:
    debug = False
    size = 224 # 이미지 크기
    batch_size = 64
    num_workers = 4
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    image_path = "/mnt/elice/dataset/Images"
    annotation_path = "/mnt/elice/dataset/captions.json"

    model_name = 'resnet50'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # 사전학습 가중치
    trainable = True # 파라미터 freezing
    temperature = 1.0
    
    # image/text encoder에 적용되는 projection head 구조
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
```

```python
image_list = glob(os.path.join(CFG.image_path, "*.jpg"))
print(f"Number of images: {len(image_list)}")

with open(CFG.annotation_path, 'r') as f:
    annotations = json.load(f)
print(f"Number of annotations: {len(annotations)}")
```

```python
data_csv = pd.DataFrame(annotations)
data_csv.drop(axis=1, inplace=True, columns=['id'])
data_csv.sort_values(by='image_id', inplace=True)
data_csv.head()
```

```python
len(data_csv['image_id'].unique())
```

```python
image_path_pattern = "/mnt/elice/dataset/Images/COCO_train2014_{:012d}.jpg"
# 각 이미지 ID에 대해 실제 이미지 경로를 생성하여 data_csv에 새로운 'image_path' 열로 추가
data_csv['image_path'] = data_csv['image_id'].apply(lambda x: image_path_pattern.format(x))

# 전체 이미지 ID 중 처음 5000개를 학습 데이터(train)용 ID로 선택
train_ids = data_csv['image_id'].unique()[:5000]

# 그 다음 1000개 (5000 ~ 6000)를 검증 데이터(val)용 ID로 선택
val_ids = data_csv['image_id'].unique()[5000:6000]

# 학습 데이터프레임 생성: image_id가 train_ids에 포함된 행만 선택
train_csv = data_csv[data_csv['image_id'].isin(train_ids)]

# 검증 데이터프레임 생성: image_id가 val_ids에 포함된 행만 선택
val_csv = data_csv[data_csv['image_id'].isin(val_ids)]

# 학습 데이터와 검증 데이터의 크기(행 개수)를 출력
print(train_csv.shape, val_csv.shape)
```

### 기타 함수 및 클래스

```python
class AvgMeter: # 모델 학습과 검증 과정에서 손실 값을 효율적으로 집계하고 평균을 계산하여, 학습 모니터링에 중요한 역할을 수행
    def __init__(self, name="Metric"):
        self.name = name  # 측정할 지표의 이름 (예: "train_loss" 또는 "valid_loss")
        self.reset()     # 초기화 메서드를 호출하여 평균, 총합, 샘플 수를 0으로 설정

    def reset(self):
        # 평균(avg), 총합(sum), 샘플 개수(count)를 모두 0으로 초기화
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        # 업데이트할 때마다 배치의 손실 값(val)과 해당 배치에 포함된 샘플 수(count)를 반영
        # count: 배치에 포함된 샘플 수 (예: 이미지 텐서의 첫번째 차원 크기)
        self.count += count              # 총 샘플 수 업데이트
        self.sum += val * count          # 배치의 총 손실 (손실 값 * 샘플 수)를 누적
        self.avg = self.sum / self.count # 누적 손실을 전체 샘플 수로 나눠 평균 손실 계산

    def __repr__(self):
        # 객체 출력 시 측정 지표의 이름과 현재까지의 평균 손실을 소수점 4자리까지 출력
        text = f"{self.name}: {self.avg:.4f}"
        return text

# 옵티마이저에서 현재 Learning Rate를 반환하는 함수
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
```

### Dataset

```python
# CLIP 모델 학습에 사용할 커스텀 데이터셋 클래스
class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, transforms):
        # 데이터프레임에서 이미지 경로를 리스트로 추출 (각 이미지의 실제 경로)
        self.image_paths = df['image_path'].tolist()
        # 데이터프레임에서 캡션(문자열)들을 리스트로 추출
        self.captions = df['caption'].tolist()
        
        # 텍스트 토크나이저를 사용하여 캡션들을 인코딩
        # - padding: 시퀀스 길이를 맞추기 위해 추가 토큰 삽입
        # - truncation: 최대 길이 초과 시 잘라냄
        # - max_length: 최대 토큰 길이를 CFG.max_length로 제한
        self.encoded_captions = tokenizer(self.captions, 
                                          padding=True, 
                                          truncation=True, 
                                          max_length=CFG.max_length)
        
        # 이미지 전처리(또는 데이터 증강) 함수 저장 (예: 리사이즈, 정규화 등)
        self.transforms = transforms
        
    def __len__(self):
        # 데이터셋의 길이 반환: 캡션의 총 개수와 동일함
        return len(self.captions)

    def __getitem__(self, idx):
        # 인덱스 idx에 해당하는 인코딩된 캡션 데이터를 텐서로 변환하여 딕셔너리 형태로 저장
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        # 이미지 경로를 사용해 해당 이미지를 읽음 (cv2 사용)
        image = cv2.imread(self.image_paths[idx])
        # OpenCV는 기본적으로 BGR 순서를 사용하므로, RGB 순서로 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 미리 정의된 transforms를 적용하여 이미지 전처리 (리사이즈, 정규화 등)
        image = self.transforms(image=image)['image']
        
        # 전처리된 이미지를 텐서로 변환
        # - permute(2, 0, 1): 이미지 텐서의 채널 순서를 (채널, 높이, 너비)로 변경
        # - float(): 데이터 타입을 부동소수점으로 변환
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        # 원본 캡션 문자열도 함께 추가 (추후 필요 시 참고 가능)
        item['caption'] = self.captions[idx]

        # 구성된 데이터 샘플 반환 (이미지 텐서, 인코딩된 캡션, 원본 캡션 등)
        return item


# 이미지 전처리 함수 정의 (모드에 따라 적용할 전처리 방법을 지정)
def get_transforms(mode="train"):
    # train 모드와 그 외 모드 모두 동일한 전처리 적용 (리사이즈와 정규화)
    return A.Compose(
        [
            # CFG에서 정의한 크기로 이미지 리사이즈
            A.Resize(CFG.size, CFG.size, always_apply=True),
            # 이미지의 픽셀 값을 0~1 범위로 정규화
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )
```

```python
train_dataset = CLIPDataset(df=train_csv, 
                            tokenizer=DistilBertTokenizer.from_pretrained(CFG.text_tokenizer), 
                            transforms=get_transforms(mode="train"))

val_dataset = CLIPDataset(df=val_csv, 
                          tokenizer=DistilBertTokenizer.from_pretrained(CFG.text_tokenizer), 
                          transforms=get_transforms(mode="val"))


print(f"Train: {len(train_dataset)} - Validation: {len(val_dataset)}")
```

```python
train_dataset[0]
```

### Dataloader

```python
train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=CFG.batch_size, 
                                               num_workers=CFG.num_workers, 
                                               shuffle=True)

val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=CFG.batch_size, 
                                             num_workers=CFG.num_workers)
```

```python
batch = next(iter(train_dataloader))
batch
```

## Model
CLIP 모델은 아래 세 부분으로 구성
- Text Encoder
- Image Encoder
- Projection Head

### Text Encoder

```python
# TextEncoder 클래스: 텍스트 인코딩을 위한 모듈
class TextEncoder(nn.Module):
    def __init__(
        self, 
        model_name=CFG.text_encoder_model,   # 사용할 텍스트 인코더 모델 이름
        pretrained=CFG.pretrained,           # 사전학습 가중치 사용 여부 (CFG 설정)
        trainable=CFG.trainable              # 모델 파라미터 업데이트 여부 (freeze 여부)
    ):
        super().__init__()
        # 사전학습된 텍스트 모델 사용 여부에 따라 모델 초기화
        if pretrained:
            # Hugging Face의 DistilBertModel을 사전학습된 가중치로 불러옴
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            # 사전학습되지 않은 모델의 경우 기본 구성(config)으로 초기화
            self.model = DistilBertModel(config=DistilBertConfig())
            
        # 모델의 모든 파라미터에 대해 학습 가능 여부 설정 (freeze 또는 fine-tuning)
        for p in self.model.parameters():
            p.requires_grad = trainable

        # 문장 임베딩을 위해 BERT의 CLS 토큰(hidden state)의 인덱스를 지정
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        # 입력 아이디와 어텐션 마스크를 사용해 DistilBERT 모델의 출력을 생성
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        # CLS 토큰의 hidden state를 문장 임베딩으로 반환
        return last_hidden_state[:, self.target_token_idx, :]
```

### Image Encoder

```python
# ImageEncoder 클래스: 이미지 인코딩을 위한 모듈
class ImageEncoder(nn.Module):
    def __init__(
        self, 
        model_name=CFG.model_name,         # 사용할 모델 이름
        pretrained=CFG.pretrained,         # 사전학습 가중치 사용 여부 (CFG 설정)
        trainable=CFG.trainable            # 모델 파라미터 업데이트 여부 (freeze 여부)
    ):
        super().__init__()
        # timm 라이브러리를 사용하여 사전학습된 이미지 모델 생성
        # num_classes=0과 global_pool="avg"를 설정하여 feature extractor로 사용
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        # 모델의 모든 파라미터에 대해 학습 가능 여부 설정 (freeze 또는 fine-tuning)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        # 입력 이미지 텐서를 모델에 통과시켜 특징(feature) 추출
        return self.model(x)
```

### Projection Head

```python
# ProjectionHead 클래스: 임베딩을 투영하는 모듈
class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,  # 입력 임베딩 차원
        projection_dim=CFG.projection_dim,  # 투영할 차원 (CFG에서 정의)
        dropout=CFG.dropout  # 드롭아웃 비율 (CFG에서 정의)
    ):
        super().__init__()
        # 입력 임베딩을 투영 차원으로 변환하는 선형 레이어
        self.projection = nn.Linear(embedding_dim, projection_dim)
        # GELU 활성화 함수
        self.gelu = nn.GELU()
        # 투영 차원 내에서 추가적인 선형 변환을 위한 레이어
        self.fc = nn.Linear(projection_dim, projection_dim)
        # 과적합 방지를 위한 드롭아웃 레이어
        self.dropout = nn.Dropout(dropout)
        # 레이어 정규화 (Residual 연결 후 안정성 확보)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        # 입력 임베딩 x를 선형 변환하여 투영 결과 생성
        projected = self.projection(x)
        # GELU 활성화 적용
        x = self.gelu(projected)
        # 추가 선형 변환
        x = self.fc(x)
        # 드롭아웃 적용
        x = self.dropout(x)
        # Residual connection: 원래 투영된 값과 합산
        x = x + projected
        # 레이어 정규화로 출력 안정화
        x = self.layer_norm(x)
        return x
```

### CLIP 모델

```python
class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,      # Contrastive 학습에서 온도 조절 파라미터 (스케일링 역할)
        image_embedding=CFG.image_embedding,  # 이미지 인코더의 출력 임베딩 차원
        text_embedding=CFG.text_embedding,    # 텍스트 인코더의 출력 임베딩 차원
    ):
        super().__init__()
        # 이미지 특징 추출을 위한 이미지 인코더 생성
        self.image_encoder = ImageEncoder()
        # 텍스트 임베딩 생성을 위한 텍스트 인코더 생성
        self.text_encoder = TextEncoder()
        # 이미지 임베딩을 동일 차원으로 투영하기 위한 Projection Head 생성
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        # 텍스트 임베딩을 동일 차원으로 투영하기 위한 Projection Head 생성
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        # 온도 파라미터 저장 (contrastive loss 계산 시 사용)
        self.temperature = temperature

    def forward(self, batch):
        # 배치에서 이미지 데이터를 받아 이미지 인코더를 통해 특징 추출
        image_features = self.image_encoder(batch["image"])
        # 배치에서 텍스트 데이터를 받아 텍스트 인코더를 통해 임베딩 생성
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        
        # 이미지와 텍스트 특징을 각각 Projection Head를 통과시켜 동일한 차원의 임베딩으로 투영
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # 이미지와 텍스트 임베딩 간의 유사도를 계산하기 위해 내적(dot product) 후 온도 스케일링 적용
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        
        # 이미지와 텍스트 임베딩 사이의 자기 유사도 계산 (이미지-이미지, 텍스트-텍스트)
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        
        # 이미지와 텍스트의 자기 유사도를 평균 내고 온도 스케일링 후 소프트맥스로 정답 분포(target) 생성
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        
        # 텍스트와 이미지 임베딩 간의 내적 결과(logits)에 대해 cross entropy loss 계산 (텍스트 기준)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        # 이미지와 텍스트 임베딩 간의 내적 결과를 전치(transpose)하여 이미지 기준 loss 계산
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        # 두 개의 손실을 평균 내어 최종 loss 계산 (배치 내 각 샘플에 대해)
        loss = (images_loss + texts_loss) / 2.0
        # 배치의 평균 손실을 반환
        return loss.mean()


# Contrastive Loss 계산을 위한 CrossEntropyLoss 함수 정의
def cross_entropy(preds, targets, reduction='none'):
    # 입력 logits에 대해 LogSoftmax 계산 (마지막 차원을 기준으로)
    log_softmax = nn.LogSoftmax(dim=-1)
    # 음의 정답 분포와 LogSoftmax 결과를 곱한 후, 각 샘플별로 합산하여 loss 계산
    loss = (-targets * log_softmax(preds)).sum(1)
    # reduction 옵션에 따라 loss를 그대로 반환하거나 평균값을 반환
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
```
```python
dummy_embeddings = torch.randn(CFG.batch_size, CFG.text_embedding)

out = dummy_embeddings @ dummy_embeddings.mT
print(F.softmax(out, dim=-1))
```

## Training

### 모델 선언

```python
model = CLIPModel().to(CFG.device)

params = [
    {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
    {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
    {"params": itertools.chain(
        model.image_projection.parameters(), model.text_projection.parameters()
    ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
]
```

### Train & validation 함수

```python
def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    # 학습 손실을 기록하기 위한 AvgMeter 인스턴스 생성 (손실 지표 이름: 기본 "Metric")
    loss_meter = AvgMeter()
    
    # tqdm을 사용하여 학습 데이터 로더에 대한 진행률 표시
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    # 학습 데이터 로더를 순회하면서 배치별로 처리
    for batch in tqdm_object:
        # 캡션을 제외한 나머지 항목을 CFG.device (GPU 또는 CPU)로 이동
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        
        # 모델에 배치를 입력하여 손실(loss) 계산
        loss = model(batch)
        
        # 옵티마이저의 기울기 초기화
        optimizer.zero_grad()
        
        # 역전파 수행
        loss.backward()
        
        # 가중치 업데이트
        optimizer.step()
        
        # 배치 단위로 스케줄러 업데이트: step이 "batch"인 경우에만 적용
        if step == "batch":
            lr_scheduler.step()

        # 배치 내 이미지 수 (샘플 개수)를 구함
        count = batch["image"].size(0)
        
        # 현재 배치의 손실 값을 샘플 개수만큼 반영하여 AvgMeter 업데이트
        loss_meter.update(loss.item(), count)

        # 진행률 표시줄에 현재 평균 학습 손실과 learning rate 출력
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    
    # 전체 에폭에 대한 평균 손실 반환
    return loss_meter


def valid_epoch(model, valid_loader):
    # 검증 손실을 기록하기 위한 AvgMeter 인스턴스 생성
    loss_meter = AvgMeter()

    # tqdm을 사용하여 검증 데이터 로더에 대한 진행률 표시
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    
    # 검증 데이터 로더를 순회하면서 배치별로 처리
    for batch in tqdm_object:
        # 캡션을 제외한 나머지 항목을 CFG.device (GPU 또는 CPU)로 이동
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        
        # 모델에 배치를 입력하여 손실(loss) 계산
        loss = model(batch)

        # 배치 내 이미지 수 (샘플 개수)를 구함
        count = batch["image"].size(0)
        
        # 현재 배치의 손실 값을 샘플 개수만큼 반영하여 AvgMeter 업데이트
        loss_meter.update(loss.item(), count)

        # 진행률 표시줄에 현재 평균 검증 손실 출력
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    
    # 전체 검증 에폭에 대한 평균 손실 반환
    return loss_meter
```

```python
step = "epoch"
optimizer = torch.optim.AdamW(params, weight_decay=0.)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                          mode="min", 
                                                          patience=CFG.patience, 
                                                          factor=CFG.factor)

best_loss = float('inf')
```

```python
for epoch in range(CFG.epochs):
    print(f"Epoch: {epoch + 1}")
    model.train()
    train_loss = train_epoch(model, train_dataloader, optimizer, lr_scheduler, step)
    model.eval()
    with torch.no_grad():
        valid_loss = valid_epoch(model, val_dataloader)

    if valid_loss.avg < best_loss:
        best_loss = valid_loss.avg
        torch.save(model.state_dict(), "best.pt")
        print("Saved Best Model!")

    lr_scheduler.step(valid_loss.avg)
```

## Inference

### Image embeddings 가져오기


```python
def get_image_embeddings(dataloader, model_path):
    # 텍스트 토크나이저 초기화: CFG에서 지정한 토크나이저 이름을 사용
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    
    # CLIPModel 인스턴스 생성 후, 지정된 디바이스(CFG.device)로 이동
    model = CLIPModel().to(CFG.device)
    # 저장된 모델 가중치 불러오기 (모델 파일 경로: model_path, CFG.device로 매핑)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    # 평가 모드로 전환 (드롭아웃, 배치 정규화 등 평가 시 동작 변경)
    model.eval()
    
    # 이미지 임베딩 결과를 저장할 리스트 초기화
    valid_image_embeddings = []
    # 기울기 계산 없이 이미지 임베딩 추출 (메모리 절약 및 추론 속도 향상)
    with torch.no_grad():
        # 데이터로더에 포함된 모든 배치를 순회
        for batch in tqdm(dataloader):
            # 배치 내 "image" 데이터를 지정된 디바이스로 이동
            # 이미지 인코더를 통해 이미지 특징 추출
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            # 이미지 특징을 Projection Head를 통해 동일 차원의 임베딩으로 투영
            image_embeddings = model.image_projection(image_features)
            # 추출된 임베딩을 리스트에 추가
            valid_image_embeddings.append(image_embeddings)
    # 모든 배치의 이미지 임베딩을 하나의 텐서로 연결하여 반환하며, 모델도 함께 반환
    return model, torch.cat(valid_image_embeddings)
```

```python
model, image_embeddings = get_image_embeddings(val_dataloader, "best.pt")
```

### Query에 맞는 Image embedding 회수하기

```python
def find_matches(model, image_embeddings, query, image_filenames, n=9):
    # 텍스트 토크나이저 초기화 (CFG에 지정된 토크나이저 사용)
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    # 검색 쿼리(query)를 토큰화하여 인코딩 (배치 형태로 변환)
    encoded_query = tokenizer([query])
    # 인코딩된 결과를 텐서로 변환하고, CFG.device로 이동하여 배치 데이터 구성
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    # 기울기 계산 없이 텍스트 임베딩 추출 (추론 모드)
    with torch.no_grad():
        # 텍스트 인코더를 사용해 쿼리의 텍스트 특징 추출
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # 텍스트 특징을 Projection Head를 통해 동일 차원의 임베딩으로 투영
        text_embeddings = model.text_projection(text_features)
    
    # 이미지 임베딩과 텍스트 임베딩을 정규화 (L2 norm, p=2) 하여 비교하기 쉽게 만듦
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    # 텍스트 임베딩과 모든 이미지 임베딩 간의 내적(dot product) 계산하여 유사도 행렬 생성
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    # 유사도 행렬에서 상위 n*5개의 값을 선택 (여러 후보 중에서 일정 간격으로 샘플링하기 위함)
    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    # 선택된 인덱스 중 매 5번째 값을 선택하여 최종 매칭 이미지 파일명 리스트 구성
    matches = [image_filenames[idx] for idx in indices[::5]]
    
    # 결과를 3x3 형태의 서브플롯으로 시각화 (총 9개의 이미지)
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        # 파일 경로(match)를 통해 이미지 읽기 (cv2 사용)
        image = cv2.imread(match)
        # BGR 색상 순서를 RGB로 변환 (cv2 기본 색상 순서 보정)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 서브플롯에 이미지 표시
        ax.imshow(image)
        ax.axis("off")  # 축 제거
    # 모든 서브플롯 표시
    plt.show()
```

```python
find_matches(model, 
             image_embeddings,
             query="A horse sitting on the grass",
             image_filenames=val_csv['image_path'].values,
             n=9)
```

```python
find_matches(model, 
             image_embeddings,
             query="All kinds of delicious food",
             image_filenames=val_csv['image_path'].values,
             n=9)
```
