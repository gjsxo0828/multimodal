# multimodal

### 멀티모달 챗봇 개요

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

