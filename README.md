# multimodal

## 멀티모달 챗봇 개요

1. 멀티모달이란
  - 한 매체에 여러 형식의 정보(multiple literacies)를 적용한 경우를 말한다.
  - 예, 음성, 시각(화살표 기호), 시각(문자열) 등
  - output 뿐만 아니라 input 또한 여러 형식의 정보를 활용할 수 있다.

2. 멀티모달 챗봇이란
  - 텍스트만 해석할 수 있는 챗봇이 아닌, 이미지, 영상, 음성 등 여러 형식의 정보를 활용하고 해석할 수 있는 챗봇을 의미한다.

3. 멀티모달 LLM이란
   - 멀티모달 LLM은 그 목적과 기능에 따라 멀티모달 데이터를 이해하거나, 생성할 수 있다.
   - Image, Videio, Audio -> Modality Encoder (벡터화, embedding) -> Input Projector (통합 embedding 공간으로 정렬, 이미지와 자연어를 처리할 수 있도록 같은 공간으로 정렬하는 과정)
   - Output Projector -> Modality Generator
  
### 멀티모달 챗봇 실습 환경
 - Vision Question Answering (LLAVA)
 - Table Parsing
 - PDF 문서 파싱

### 이미지와 텍스트를 이해하는 챗봇

1. 텍스트를 이해해야함
2. 이미지를 포함된 정보를 이해해야함
3. 이미지의 정보와 텍스트의 정보를 조합하는 기능
4. 조합된 정보를 바탕으로 적합한 답변을 생성할 수 있어야 함


### LLM이 데이터를 이해하는 과정
1. LLM을 비롯 하여 자연어를 처리하는 AI는 자연어를 있는 그대로 입력 받는 것이 아니라 특정 알고리즘에 따라 정해진 크기로 벡터화 된 정보를 입력 받는다.
2. 자연어 데이터를 이해하기 위해 초기에 토큰화 (Tokenization)을 거친다.
   - 토큰화는 텍스트를 작은 단위인 토큰으로 쪼개는 것을 의미함
   - 텍스트를 토큰으로 쪼개는 모델을 토크나이저라고 함(Tokenizer)
   - https://platform.openai.com/tokenizer
   - OpenAI 토크나이저를 시각화 해주는 서비스가 있음 (주소,)
   - 보통 LLM 서비스 과금 방식은 Toekn 단위로 됨
3. 토큰화된 텍스트 데이터를 임베딩 모델을 통해 벡터화된 데이터로 변환함
   - 비슷한 의미를 가진 텍스트는 수학적으로 비슷한 벡터를 가지며 단어 문장이 가지고 있는 정보, 의미를 벡터 연산을 통해 더하고 뺄 수 있다.
  
### 멀티모달 예시(CLIP)
1. CLIP (Contrastive Lanaguage-Image Pretraining)은 OpenAI에서 개발한 이미지-텍스트 멀티모달 모델이다.
   - 이미지와 텍스트를 같은 임베딩 공간에서 학습시켜, 이미지와 텍스트를 연결하는 능력이 뛰어나다.
2. CLIP은 대조 학습 (Contrastive Learining) 방식을 이용해 이미지와 텍스트를 연결한다.
3. CLIP 모델을 할용한 다른 모델인 LAVVA 도 있다.

### Diffusion Model
학습과정 : 원본이미지에 노이즈를 추가하며 학습, 다시 노이즈를 제거하여 학습(디노이징)
사용과정 : 노이즈만 생성한 이미지에 text 정보를 받아서 노이즈를 제거하면서 이미지를 추론하여 생성하는 과정을 진행함 
conditioning으로 주는 매체(input)은 text 뿐만 아니라, Pose, 테이블, 범주형 등 다양한 포맷이 가능함.


---

## 멀티모달
- Multimodal tasks
  - 여러 종류 modality를 동시에 처리하거나 통합하여 문제를 해결하는 작업
- Modality : 데이터의 성격에 의해 결정되는 서로 다른 유형
  - 텍스트, 이미지, 음성, 비디오 등 서로 다른 감각 기반의 데이터 
  - 동일한 감각 기관으로 인지된다 하더라도, 관찰 및 수집 방식이 다를 경우 다른 Modality로 간주
- Text-to-image gengeration, VQA(Visual Question-Answering), Image Captioning, VCR(Visual Commonsense Reasoning)

### 멀티모달을 활용한 모델 종류들 
1. Enc-Dec Models
- 이미지를 처리하는 encoder와, 텍스트 생성 기능을 담당하는 Decoder로 구성
  - Encoder
    - 이미지에 대한 표현을 벡터로 변환
    - CNN 기반의 Encoder를 사용할 경우 Feature map을 Self-attention 기반의 Encoder를 사용할 경우 Patch embedding (작은 크기로 이미지를 분할하여 벡터화)을 계산
  - Decoder
    - Encoder에서 생성된 데이터를 전달받아 Text 생성 시 참조
    - 텍스트 생성 과정은 일반적인 Transformer decoder 활용
    - Decoder에 입력된 텍스트는 Query, Encdoer에서 출력된 벡터는 Key와 Value로 사용되어 Cross-attention 연산을 수행함
  - 관련 모델 (Image Captioning, VLN, VCR 등 활용)
    - Vision Encoder Decoder Models
    - Seq2seq trainer

2. Contrastive Learning Models
- Contrastive Learning 을 활용하여 이미지와 텍스의 사이의 관계를 학습
  - 이미지와 이에 대한 Caption을 비지도 학습을 통해 연결지으며 유사도를 최대화
  - Positive Pair : 이미지와 이에 대한 올바른 설명
  - Negative Pair : 무작위로 짝지어진, 관계없는 이미지와 설명
- 시각적 데이터를 자연어 설명과 연결시키는데 자주 사용
  - Text2Image 생성모델, Image Retrieval, VL-Classification등에서 사용
 

---

### 대조학습

일반적인 Machin Learning 분류
1. 비지도 학습
   - 데이터만으로 모델을 학습 (비용이 매우 저렴)
   - 정답이 없기 때문에 방법이 매우 모호하며 추상적
   - 모델의 학습 방향을 제시하기 어려움 (보통 시각하여 정성적으로 분류)
3. 지도 학습
   - 데이터(X)와 이에 대한 레이블(y_true)를 제공
   - 데이터를 통해 모델이 예측한 값(y_pred)과 정답 간 차이를 계산
   - 차이가 수치화되므로 모델이 어느 방향으로 학습해야할 지 지시할 수 있음
   - 그러나 지도학습 데이터를 만드는 것은 노동력, 비용, 시간이 많이 필요, 텍스트 데이터의 경우 사실상 불가능한 접근 방식
4. 강화 학습 (Agent가 직접 Action을 수행하면서 Reward 보상을 최대화 할 것인가 하는 학습방식)

   
#### SSL(Self-Sufperivsed Learning) (Repersentation Learning 방법 중에 하나)
- 데이터를 레이블 없이 학습하는 방식의 일종 (비지도 학습)
  - 별도로 데이터를 대한 레이블을 생성하지 않음
- 그러나 모델이 데이터를 스스로 학습할 수 있도록 하는 기법
- 일종의 Representation Learining
- 마스킹을 사용하여 자체 데이터를 가지고 정답지와 문제지를 만들어 증강 학습 사용
  - SSL에서 모델이 학습할 수 있도록 만들어진 가짜 목표(인위적인 과제)로 학습 진행
  - 레이블이 없는 데이터로 모델을 학습할 수 있도록 데이터에 인위적인 변형을 부여
  - 그 변형을 예측하거나 특정 문제를 해결하게 끔 유도 (BERT, MLM, NSP)
- 목표가 구체화된 신호로 모델을 지도하는 별도의 학습과정 (fine-tuning)을 거침
  -> 궁극적으로 Downstream task에서 좋은 성능을 낼수 있는 특징을 학습하는 것이 목표
  - QA sentnece generation, medical object detection 등
 
#### Representation Learning
- 데이터에서 의미있는 표현(Representation)을 자동으로 학습하는 방식
  - 작업을 수행하기 위한 특징만을 학습하지 않고 Task-agnostic 한 정보들을 추출하는 것이 목표
  - 분류/탐지를 잘 수행하는 것이 아니라 이미지 자체를 잘 이해하고, 필수적인 정보를 포작하는 과정 (패턴 추출 등)
- 고차원 데이터에서 저차원 벡터로 표현(mapping) 차원축소할 수 있어야 함.
  - 고차원 상의 데이터에서 불필요한 부분을 제거하고, 필수적인 부분을 남겨 요약할 수 있어야 함.
 
#### 대조학습
- 비슷한 데이터 쌍은 임베딩 공간에서 가깝게, 다른 데이터 쌍은 멀게 학습하는 표현 학습(Represenation Learning) 방법 중에 하나
- 이방식은 주로 라벨이 없는 데이터 학습에서 사용

1. Instance discrimination
   - 기준(Anchor) 데이터, 비슷한 데이터(Similar), 전혀다른 데이터(Dissimilar)가 주어졌을 때, 이들 간의 유사도를 학습하는 과정
   - 즉 기준 데이터를 변형하여 서로 다른 표현들을 비교하며 학습
   - Intra-class(클래스 내부) distance, inter-class(클래스간) distance 조정
  
   - 절차
     - 주어진 데이터에서 기준 데이터를 설정
     - 랜덤하게 변형 된 데이터인 Positive Pair 생성
     - Negative piar 학습
     - Positive pair는 가까운 거리로, Negative Pair는 먼 거리로 학습하며 특징을 추출


#### SimCLR
비지도 학습도 지도학습만큼 성능을 낼 수 있는 것을 확인한 의미있는 이론
Augmentation(증강) -> Encoding(Resnet) -> projection(MLP = Linear + ReLU)

- 증강에 사용되는 방법들
  - Crop, resize, flip, color distort(drop), color distort(jitter), rotate, Cutout, Gaussian Noise, Gaussian Blur, Sobel filtering 등


#### CLIP
- 이미지와 텍스트를 같은 임베딩 공간에서 학습시켜, 이미지와 텍스트를 연결
  - 이미지, 텍스트의 embedding vector에 대해 유사도를 계산
  - 같은 pair라면 높은 유사도 값, 다른 pair라면 낮은 유사도 값을 갖도록 학습
- 이미지와 텍스트에 대한 cross-entropy loss 계산
- 이미지가 input되면, 유사도 계산
- 유사도가 가장 큰 텍스트로 분류됨

---

## VQA

### VQA(Visual Question Answering) 정의
- 이미지나 영상을 입력 받아 자연어 질문에 답하는 AI 기술을 의미한다.
- 컴퓨터 비전과 자연어 처리 기술의 융합을 통해 이루어진다.
- 사용자는 텍스트 질문을 입력하고, AI가 이미지 속 정보를 분석해 답변을 추출한다.

### VQA의 목표
- 세밀한 인식 (fine-grained recognition)
- 객체 탐지 (object detection)
- 행동 인식 (activity recognition)
- 지식 기반 추론 (knowledge base reasoning)
- 상식 추론 (common sense reasoning)

### LLAVA with VQA
  
