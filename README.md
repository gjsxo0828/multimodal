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

   

