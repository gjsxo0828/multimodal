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
![image](https://github.com/user-attachments/assets/b742319a-428e-4dab-bdf4-31b02e9df2c3)


```python
len(data_csv['image_id'].unique())
```
> 18783

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
> (25008, 3) (5001, 3)

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
> Train: 25008 - Validation: 5001

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
```
{'input_ids': tensor([[ 101, 1037, 2158,  ...,    0,    0,    0],
         [ 101, 1037, 2711,  ...,    0,    0,    0],
         [ 101, 2045, 2024,  ...,    0,    0,    0],
         ...,
         [ 101, 2048, 2111,  ...,    0,    0,    0],
         [ 101, 1037, 2317,  ...,    0,    0,    0],
         [ 101, 1037, 3345,  ...,    0,    0,    0]]),
 'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0],
         ...,
         [1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0],
         [1, 1, 1,  ..., 0, 0, 0]]),
 'image': tensor([[[[-1.9980, -1.9980, -1.9809,  ..., -1.7412, -1.9124, -1.8610],
           [-1.9980, -1.9809, -1.9638,  ..., -1.8097, -1.7925, -1.8782],
           [-1.9980, -1.9809, -1.9467,  ..., -1.6898, -1.9124, -1.7754],
           ...,
           [-2.0837, -2.0837, -2.1008,  ..., -2.0837, -2.0837, -2.0837],
           [-2.0837, -2.0837, -2.1008,  ..., -2.0837, -2.0837, -2.0837],
           [-2.0837, -2.1008, -2.0837,  ..., -2.1008, -2.1008, -2.1008]],
 
          [[-1.9132, -1.9132, -1.8957,  ..., -1.6155, -1.7906, -1.7381],
           [-1.9132, -1.8957, -1.8782,  ..., -1.6856, -1.6681, -1.7381],
           [-1.9132, -1.8957, -1.8606,  ..., -1.5630, -1.7906, -1.6506],
           ...,
           [-2.0007, -2.0007, -2.0182,  ..., -2.0007, -2.0007, -2.0007],
           [-2.0007, -2.0007, -2.0182,  ..., -2.0007, -2.0007, -2.0007],
           [-2.0007, -2.0182, -2.0007,  ..., -2.0182, -2.0182, -2.0182]],
 
          [[-1.6824, -1.6824, -1.6650,  ..., -1.4036, -1.5779, -1.5256],
           [-1.6824, -1.6650, -1.6476,  ..., -1.4733, -1.4559, -1.5256],
           [-1.6824, -1.6650, -1.6302,  ..., -1.3513, -1.5779, -1.4384],
           ...,
           [-1.7696, -1.7696, -1.7870,  ..., -1.7696, -1.7696, -1.7696],
           [-1.7696, -1.7696, -1.7870,  ..., -1.7696, -1.7696, -1.7696],
           [-1.7696, -1.7870, -1.7696,  ..., -1.7870, -1.7870, -1.7870]]],
 
 
         [[[ 0.4337,  0.5022,  0.4337,  ...,  0.7419,  0.7762,  0.8276],
           [ 0.4166,  0.4851,  0.4508,  ...,  0.6906,  0.7591,  0.8104],
           [ 0.3652,  0.4508,  0.3652,  ...,  0.7591,  0.8104,  0.7933],
           ...,
           [-1.0219, -1.0219, -0.9363,  ..., -0.6623, -0.8164, -0.8507],
           [-0.8164, -0.7822, -0.7650,  ..., -0.8335, -0.7650, -0.5082],
           [-0.6623, -0.6965, -0.5938,  ..., -0.6623, -0.7822, -0.4911]],
 
          [[ 0.7304,  0.8004,  0.7304,  ...,  0.9580,  1.0105,  1.0455],
           [ 0.7129,  0.7829,  0.7479,  ...,  0.9055,  0.9755,  1.0280],
           [ 0.6429,  0.7304,  0.6604,  ...,  0.9755,  1.0280,  1.0105],
           ...,
           [-1.4055, -1.3529, -1.3179,  ..., -0.9153, -1.0728, -1.0203],
           [-1.2829, -1.1779, -1.1779,  ..., -1.0728, -1.0203, -0.6877],
           [-1.2129, -1.1429, -1.0028,  ..., -0.9153, -1.0378, -0.6702]],
 
          [[ 1.2457,  1.3154,  1.2457,  ...,  1.3851,  1.4200,  1.4722],
           [ 1.2282,  1.2980,  1.2631,  ...,  1.3502,  1.4025,  1.4548],
           [ 1.1934,  1.2805,  1.2108,  ...,  1.4025,  1.4374,  1.4374],
           ...,
           [-1.2990, -1.1944, -1.1073,  ..., -0.5670, -0.7238, -0.7064],
           [-1.1596, -0.9853, -0.9504,  ..., -0.7413, -0.6715, -0.3578],
           [-1.0550, -0.9330, -0.7761,  ..., -0.5670, -0.6890, -0.3404]]],
 
 
         [[[ 1.6324,  0.6906,  0.0227,  ..., -0.0972, -0.1486, -0.1486],
           [ 1.6667,  0.9303,  0.8789,  ..., -0.0116, -0.1657, -0.1486],
           [ 1.5982,  1.0673,  1.5125,  ...,  0.0569, -0.1999, -0.1657],
           ...,
           [-0.5596, -0.6794, -1.4158,  ..., -1.3130, -1.2959, -1.3302],
           [-0.4054, -0.6965, -0.8507,  ..., -1.2959, -1.2788, -1.2274],
           [ 0.3481,  0.3481,  0.0056,  ..., -1.3815, -1.3644, -1.2959]],
 
          [[ 1.7633,  1.2731,  0.7479,  ..., -0.3025, -0.3725, -0.4601],
           [ 1.7983,  1.4482,  1.4132,  ..., -0.3025, -0.3725, -0.4601],
           [ 1.7458,  1.5007,  1.7108,  ..., -0.2675, -0.4951, -0.4601],
           ...,
           [-0.7577, -0.8452, -1.2479,  ..., -1.3880, -1.3529, -1.4230],
           [-0.5826, -0.4601, -0.4251,  ..., -1.3704, -1.3529, -1.3529],
           [ 0.7129,  0.7654,  0.4503,  ..., -1.4580, -1.4405, -1.3704]],
 
          [[ 1.9428,  1.3851,  1.0888,  ..., -0.5147, -0.5844, -0.6018],
           [ 2.0300,  1.5768,  1.9428,  ..., -0.4275, -0.5844, -0.6367],
           [ 1.9428,  1.7511,  1.9951,  ..., -0.4101, -0.6890, -0.6541],
           ...,
           [-0.8633, -1.0550, -1.3687,  ..., -1.3339, -1.3164, -1.3687],
           [-0.6367, -0.6541, -0.4798,  ..., -1.2816, -1.2816, -1.2816],
           [ 0.8797,  0.8622,  0.5659,  ..., -1.3861, -1.3687, -1.2990]]],
 
 
         ...,
 
 
         [[[-0.7822, -0.8335, -0.6965,  ...,  0.2796,  0.1426, -0.0801],
           [-0.7650, -0.7137, -0.6794,  ..., -0.3712, -0.3369, -0.4226],
           [-0.6965, -0.7308, -0.7479,  ..., -0.5767, -0.4911, -0.5938],
           ...,
           [-1.3302, -1.3130, -1.3130,  ..., -1.9124, -1.9295, -1.9638],
           [-1.2788, -1.3987, -1.3302,  ..., -1.9295, -1.9809, -2.0152],
           [-1.3987, -1.3815, -1.4158,  ..., -1.9467, -1.9980, -2.0494]],
 
          [[-0.7402, -0.6877, -0.6527,  ...,  0.6078,  0.4853,  0.3627],
           [-0.6527, -0.7052, -0.7052,  ..., -0.2500, -0.3025, -0.2675],
           [-0.5826, -0.6702, -0.6352,  ..., -0.4251, -0.4776, -0.4601],
           ...,
           [-1.2129, -1.1954, -1.2129,  ..., -1.8256, -1.8431, -1.9132],
           [-1.1604, -1.2829, -1.2304,  ..., -1.8606, -1.8957, -1.9132],
           [-1.2829, -1.2304, -1.2829,  ..., -1.8606, -1.9132, -1.9482]],
 
          [[-0.8284, -0.9330, -0.8807,  ...,  0.2696,  0.1651,  0.1302],
           [-0.7761, -0.7936, -0.8458,  ..., -0.3753, -0.2707, -0.3753],
           [-0.7587, -0.7936, -0.7936,  ..., -0.5321, -0.4275, -0.7064],
           ...,
           [-1.1073, -1.1073, -0.9853,  ..., -1.5604, -1.6127, -1.6476],
           [-1.0201, -1.1596, -1.0027,  ..., -1.5430, -1.6302, -1.6999],
           [-1.1944, -1.1421, -1.1421,  ..., -1.6302, -1.6476, -1.6650]]],
 
 
         [[[ 1.0331,  1.0673,  0.9303,  ...,  1.3584,  1.1872,  1.3413],
           [ 0.7933,  1.0502,  1.0673,  ...,  1.4098,  1.5468,  1.2385],
           [ 0.8104,  0.6734,  0.7762,  ...,  1.3070,  1.3070,  1.5297],
           ...,
           [ 1.1187,  1.1700,  1.1872,  ..., -0.2171, -0.1143,  0.3823],
           [ 1.1872,  1.2214,  1.2557,  ..., -0.0458, -0.3883, -0.3541],
           [ 1.1529,  1.4098,  1.4612,  ..., -0.3027, -0.5082, -0.4054]],
 
          [[ 0.9055,  1.0105,  0.8880,  ...,  0.3803,  0.2227,  0.3978],
           [ 0.5028,  0.8880,  0.9755,  ...,  0.1877,  0.0476,  0.0126],
           [ 0.7829,  0.8529,  0.8529,  ...,  0.4153,  0.0651,  0.2227],
           ...,
           [ 1.1155,  1.1155,  0.9580,  ..., -1.0728, -1.1078, -1.0028],
           [ 0.8704,  1.0105,  1.2206,  ..., -1.1954, -1.2479, -1.2129],
           [ 1.2381,  1.2381,  1.0980,  ..., -1.3179, -1.1779, -1.1078]],
 
          [[-0.5495, -1.1770, -1.5779,  ..., -1.1247, -1.2293, -1.4036],
           [-0.6193, -0.2010, -0.8458,  ..., -1.4907, -1.7173, -1.5779],
           [-0.1661, -0.2707, -0.1835,  ..., -1.6302, -1.7173, -1.3861],
           ...,
           [ 0.9494,  0.7054,  0.7925,  ..., -1.6824, -1.7347, -1.6302],
           [ 0.5834,  0.6879,  0.8797,  ..., -1.7347, -1.7522, -1.7696],
           [ 1.1934,  0.9319,  1.0888,  ..., -1.6999, -1.6999, -1.6999]]],
 
 
         [[[ 2.2318,  2.2147,  2.2147,  ...,  0.0056,  0.0569, -1.7583],
           [ 2.2318,  2.1290,  2.2147,  ...,  0.0912,  0.0398, -1.1589],
           [ 0.5707,  1.7523,  2.2318,  ...,  0.0912,  0.0569, -0.5253],
           ...,
           [ 0.4508, -0.3027, -0.1143,  ..., -1.0733, -1.0219, -1.5185],
           [-0.2684, -0.3198,  0.0741,  ..., -0.9705, -0.9705, -1.5014],
           [-0.2856, -0.5082,  0.1083,  ..., -1.2959, -1.4329, -1.4672]],
 
          [[ 2.4111,  2.4286,  2.3936,  ...,  0.3627,  0.3102, -1.9832],
           [ 2.4111,  2.3585,  2.4286,  ...,  0.4328,  0.3627, -1.5805],
           [ 1.0980,  2.1134,  2.3761,  ...,  0.4678,  0.4328, -0.9853],
           ...,
           [ 0.3627, -0.0574, -0.1450,  ..., -1.0203, -0.9503, -1.4230],
           [-0.1975, -0.1625, -0.0049,  ..., -0.8978, -0.8627, -1.3880],
           [-0.1275, -0.4076, -0.0574,  ..., -1.2829, -1.3354, -1.3004]],
 
          [[ 2.6226,  2.6226,  2.6051,  ...,  0.3045,  0.2696, -1.6650],
           [ 2.6226,  2.5529,  2.5354,  ...,  0.3568,  0.3393, -1.4210],
           [ 1.3851,  2.3611,  2.6051,  ...,  0.3742,  0.3916, -0.9678],
           ...,
           [ 0.3742,  0.1128, -0.0615,  ..., -1.0724, -0.9504, -1.2641],
           [-0.1661, -0.0267, -0.0267,  ..., -0.9330, -0.7761, -1.1596],
           [-0.0441, -0.2358,  0.1651,  ..., -1.2467, -1.2293, -1.1421]]]]),
 'caption': ['A man with a wristband is smiling while talking on a cell phone.',
  'A person airborne with a snowboard that is vertical.',
  'THERE ARE A LOT OF FRIENDS SITTING AROUND A TABLE ',
  'A red one door refrigerator in the corner of a tiled room.',
  'A picture of a bee is next to a bunch of bananas.',
  'A cake covered in globs of frosting on a table.',
  'A boy with a surfboard points at something, while the people around him look where he is pointing. ',
  'a lady that is skiing across a body of water',
  'A white bowl filled with cereal and banana slices.',
  'A person is sitting in a chair and a bird is on the ground. ',
  'Sign for gas pump in front of a large building.',
  'there is a large kite that is on the beach',
  'there is a person sitting and licking their fingers',
  'A boy who has just thrown a purple frisbee',
  'One sheep running through the shrub brush of a field.',
  'An uncooked pizza sits on top of a stove.',
  'A group of bananas that are sitting in a tree.',
  'Two giraffes stand by a post at a feeding station.',
  'Two giraffes eating leaves from a barren tree',
  'a large long kitchen that has a stove and some cabits',
  'A computer next to a cup of black stuff',
  'A cell phone broken into multiple pieces in the street.',
  'Two zebras grazing in grass near one another.',
  'A person with a striped shirt is sitting in a dimly lit room.',
  'A pair of scissors sitting on a red yarn roll.',
  'People flying a kite at the beach .',
  'A man flying through the air while riding a skateboard.',
  'A man standing in the doorway of a bus traveling down the road',
  'Some vegetables and a knife on a cutting board.',
  'A small group of people seated at a table with food.',
  'a couple of horses are running down a hill',
  'Two antique vases are set next to each other on a table.',
  'A food entree with a salad is served in a dish.',
  'A toilet sitting in a bathroom next to a wall under construction.',
  'A large lot with motorcycles parked and lined up along the edge of the lot and various people next to the motorcycles.',
  'A large jetliner sitting on top of an airport tarmac.',
  'Two zebra standing on a tree covered park.',
  'This is a red building with a giant clock in the center.',
  'A person wearing a red beanie and a yellow and red scarf.',
  'a horse that is walking on a ground',
  'A sign directing people by arrow to the nearest police station. ',
  'A baseball player who is holding his glove with his right hand.',
  'A woman holding a tennis racket in her hand.',
  'a street sign for masonic on a city street',
  'A stop sign at the corner of the road.',
  'A bunch of bananas are growing on a tree.',
  'Two men walking on the beach with their surfboards. ',
  'A man swinging a tennis racquet on top of a court.',
  'a sandwich with some veggies and an egg in it ',
  'A couple of women holding up smart phones in their hands.',
  'different images of a clock on a wall.',
  'A group of people standing around animals and small buildings.',
  'a person sanding with a keyboard on his neck',
  'A pan sitting on a table that has a cooked pizza on it. ',
  'A very big blue shiny garbage truck on the road.',
  'A bathroom with many brown fixtures and brown floors',
  'Two plates on a tray of fuzzy looking food.',
  'A clock tower is on the ground behind a fence.',
  'A city with people walking and riding on chariots.',
  'Some hooks on a wall holding some thongs, scissors, and keys.',
  'A smiling man that has long dread locks in his hair.',
  'Two people carrying two huge stuffed animals on their backs.',
  'a white plate with two different donuts and a jar',
  'A train in a subway that has a few passengers.']}
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
![image](https://github.com/user-attachments/assets/57254e4d-c734-4ecc-8a29-c2b34a91bc9a)


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
![image](https://github.com/user-attachments/assets/0a0678b3-c4b5-47c5-8609-b14466bfc3be)


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
![image](https://github.com/user-attachments/assets/e947345e-56c4-44f4-bb9c-d98899337f3e)

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
![image](https://github.com/user-attachments/assets/aa47773c-e109-476d-a1e4-fa6691787238)


```python
find_matches(model, 
             image_embeddings,
             query="All kinds of delicious food",
             image_filenames=val_csv['image_path'].values,
             n=9)
```
![image](https://github.com/user-attachments/assets/1b8b4489-a3d6-42e1-8bcd-d730bed87c59)

