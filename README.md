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
  - 이 피자는 어떤 종류의 치즈를 사용하였는가
- 객체 탐지 (object detection)
  - 이 사진에 자전거가 몇대 있나요
- 행동 인식 (activity recognition)
  - 이 남자는 울고 있나요?
- 지식 기반 추론 (knowledge base reasoning)
  - 이 피자는 채식주의자가 먹을 수 있나요?
- 상식 추론 (common sense reasoning)
  - 이 사람은 손님을 기다리고 있나요?
  - 이 사람은 시력이 좋은가요?

### LLAVA with VQA
- Instruction Tuning
  - Google의 FLAN(Finetuned Language Models are Zero-shot Learners) 논문에서 제안
  - LLM 모델을 Instruction 데이터셋을 통해 fine-tuning을 진행하여 Zero-shot 성능을 높이는 방법론
  - 방법론의 차이
    - (A) Pretrain-finetune (BERT, T5)
    - (B) Prompting (GPT3) : 모델의 파라미터를 수정하지 않고 지시사항을 정확하게하는 것으로 개선
    - (C) Instruction Tuning (FLAN) : Pretrained model에  : Pair 짝을 통해 학습 ??
  - Instruction Tuning에 Visual 적으로 푼 모델을 LLAVA라고 함
- 기존 Multimodal model
  - OepnFlamingo, LLaMA-Adapter : 오픈소스 LLaMA가 이미지 입력을 사용할 수 있도록 하여 multimodal LLM 구축
  - promising task transfer 일반화 성능을 보임
  - 일반적으로 Language-only task에 비해 multimodal task에서는 성능이 좋지 못함.
  - CC, LAION 등의 기존 multimodal dataset은 단순한 image-text pair data
  - multimodal instruction-followingdata 구축 필요
    - 생성 프로셋 큰 시간 비용, human crowd sourcing 시 데이터가 잘 정의되지 않을 수 있어 ChatGPT/Language-only GPT4를 이용
  - LLAVA dataset 구축
    - symbolic representation 생성을 위해 COCO dataset을 참고해옴
      - visual contetn가 포함된 instruction-following data 생성하기 위함
      - Captions : 이미지에 대한 다양한 시각 개념 추출 및 설명
      - Bounding boxes : 각 물체 및 개념들의 위치와 정보 설명
      - Language-only GPT를 위해 image 자체를 input으로 사용하지 않음
    - 3 types instruction following data
      - instruction-following data 생성하기 위함
      - COCO dataset 사용하여 생성
      - Conversation, Detaield description, Complex reasoning 3가지 타입으로 생성
      - 각 타입에 대해 사람이 직접 몇 개의 예시를 설계하여 사용
        - 데이터 수집 과정 중 유일한 human annotation
        - in-context learining에서 GPT4 쿼리를 주기 위한 seed example로 사용
        - assistnat로 ChatGPT/GPT4로 실험해본 결과, GPT4가 더 좋은 품질의 data 생성함

  #### 3 types instruction following data : Conversation
  - assistnat가 이미지를 보고 인간의 질문에 답하는 듯한 어조로 답변
  - 이미지의 시각적 내용들에 대하여 다양한 질문들이 제기됨
  - 분명한 답변이 가능한 질문만 고려됨
  - 58K 생성
 
  #### 3 types instruction following data : Detailed Description
  - 이미지에 대한 풍부하고 포괄적인 설명을 포함하기 위하여 질문 목록을 직접 작성
  - 각 이미지마다 하나의 질문을 무작위로 골라서 GPT4로 물어보고 자세한 설명을 만들어 내도록 함
  - 23K 생성

   #### 3 types instruction following data : Complex Reasoning
  - 상기 2가지 유형을 기반으로 심층적인 추론 질문 및 답변 생성
  - 일반적으로 엄격한 논리를 따르는 단계별 추론 과정을 거침
  - 77K 생성
 
- LLAVA architecture (ViT 계열 모델)
  ![image](https://github.com/user-attachments/assets/5cbed2d0-beb9-4d48-8da0-d1ba5305e4d3)

  14*14 크기로 이미지를 slicing하여 patch화 하고, instruction은 "Vicuna" Language model을 사용하였음
  image는 패치마다 벡터값이 생성되고, Instruction은 토큰마다 백터값이 생성됨. 2개를 단순 concat하여 입력데이터로 사용함

- LLAVA dataset 구성
  - 각 이미지 Xv에 대해서 T개의 multiturn conversation data
  - t번째 instruction
    ![image](https://github.com/user-attachments/assets/47622086-83ad-44d0-a991-7b3b1a916b33)

- LLAVA Training (이전에 사용한 예측 학습모델을 누적해서 사용)
  - auto-regressive training objective 사용
  - L개의 sequence에 대한 Xa의 확률 계산
    ![image](https://github.com/user-attachments/assets/367ce020-8021-41d9-a6f7-ad47c0cecfdf)

- LLAVA 문제점
  - 검색기능이 부족, 다국어적인 언어 해석 모델의 한계가 있음 -> 추론이나 검색 엔진을 추가하여 보완
  - 브랜드 같이 정보를 특정하지 못함
  - 이미지를 패치화 하다보니 정보의 유실이 발생하는 것 같다. -> 최근 논문에 의하면 이미지를 패치하지않고, 해상도를 auto-regressive 해서 학습을 시키는 것으로 사용
 
 ### LLAVA를 다운스트림 태스크에 이용하기
 - LLaVA-Med
   - stage 1 : 의학 개념 훈련 (의학이미지, 캡션, 질문, 대답)
   - stage 2 : 의학 지시 조정 데이터셋 (Instruction-tuning dataset)
   - A100 8장 노드로 15시간 내에 훈련이 종료함.
   - 훈련된 LLAVA-Med 모델을 여러 다운스트림 태스크에 응용이 가능함
 - LLaVA를 이용한 숫자 인식 (별도 모델 제작 불필요)
 - LLaVA를 이용한 동물 사진 인식 (이미지 분류)


---

## RAG

### LLM 문제점1 - Hallucination
- Hallucination(환각)
  - 사실에 아닌 것을 사실인 듯 얘기하는 현상
  - 발생원인
    - 문장 생성 메커니즘이 확률 기반
    - 모델이 학습하지 못한 데이터
    - 학습된 데이터가 충분하지 않은 경우
    - 사용자가 제공한 정보를 참으로 가정하는 경우
    - 텍스트를 통한 이미지 생성 분야에서도 자주 발생
 - ![image](https://github.com/user-attachments/assets/2d21ee80-1496-4df2-bcee-a458c05d9eae)

### in-context learning
  - LLM이 별도의 파라미터 업데이트나 추가 학습(fine-tuning)업이, 프롬프트에 주어진 예시(입력-출력 쌍)만으로 새로운 작업을 수행하는 능력
  - 인간의 유추(anology)능력과 유사하게 모델이 예시를 보고 새로운 입력에 대해 유사한 방식으로 답변
  - 프롬프트 엔지니어링의 구현 방안
    - 문장의 긍부정 감정 예측
    - ![image](https://github.com/user-attachments/assets/579d5a7c-4067-438e-a6b7-3eed89278d52)
   
### Prompt Engineering
- 프롬프트 엔지니어링
  - LLM이 갖고 있는 In-Context learing 성능에 근거
  - 자연어처리 모델을 효율적으로 활용하기 위해 프롬프트를 설계하고 최적화하는 과정
  - LLM에게 입력되는 프롬프트를 가공하여 모델의 효율을 최대화하기 위한 방법
  - 추가학습(Full/Fine-tuning) 없이, 입력 값을 가공하여 모델의 성능을 높이기 위한 방법
  - LLM 활용시 필수 기술이나, 그 중요도는 갈수록 감소하고 있음
- RAG는 프롬프트 엔지니어링의 연장선

### Why RAG
- 모든 언어 모델은 세상을 확률로 판다.
  - 단어의 등장 순서와 관계 : 확률
  - 문장의 조립 및 사용 여부 : 확률
  - 정보를 학습/답변을 생성하는 과정에서 정확도가 떨어짐
- 확률론적 관점은 사실/등장을 구분하지 못함
  - 모델의 사실적인 정보를 장담할 수 없음
  - Hallucination
  - 현업에서 LLM을 사용하기 어려운 이유 중 가장 직접적
 
### RAG(검색 증강 생성)
- RAG의 대략적인 진행 과정은 다음을 따름 :
  1. 사전에 정보를 담고 있는 문서를 일정 크기(Chunk)로 나눠서 Vector DB로 저장함
  2. 사용자의 입력과 유사한 K개의 문서를 검색 (Retrieve) 한다.
  3. 사용자의 입력에 검색된 문서를 더해 증강된 (Augmented) 프롬프트를 LLM에 입력한다.
  4. LLM은 검색된 문서 정보를 바탕으로 상대적으로 더 정확한 답변을 생성한다.
  ![image](https://github.com/user-attachments/assets/d82ae7e3-cddd-4ffd-9540-00e79c4f0dc8)





