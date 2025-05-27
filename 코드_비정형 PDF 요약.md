# [실습] 비정형 PDF 문서를 해석하는 멀티모달 챗봇 기능 구현

## 실습 목표
---
한국어 멀티모달 (이미지 + 텍스트) 기능을 지원하는 `MiniCPM-V` 오픈 소스 모델과 ChatGPT-4o-mini API를 활용하여 PDF 페이지를 이미지로 취급하는 멀티모달 챗봇을 구현합니다.


## 실습 목차
---
1. **환경 설정**: 실습을 진행하기 위해 사전에 준비된 라이브러리가 모두 잘 설치되어 있는지 확인합니다.
2. **VQA 체인 테스트**: MiniCPM-V 모델과 ChatGPT-4o-mini API를 각각 활용하여 이미지와 텍스트를 입력으로 받아 답변을 생성하는 VQA 기능이 잘 작동하는지 테스트를 진행합니다.
3. **Document 라벨 생성**: 멀티모달 LLM을 활용하여 Retriever가 문서를 검색할 때 활용할 수 있는 설명 라벨을 생성합니다.
4. **간단한 RAG 구현**: 생성한 Document 라벨을 활용하여 간단한 RAG를 구현합니다.

## 1. 환경 검증

실습을 위해 사전에 준비된 라이브러리가 모두 잘 설치되어 있는지 확인합니다.


```python
import base64
import io
import os
import uuid

import fitz
from IPython.display import Image as IPImage
from IPython.display import display
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from PIL import Image
from tqdm import tqdm
```

### 1.1. Ollama를 활용한 로컬 LLM 모델 확인

현재 환경에서는 백그라운드에서 `ollama run minicpm-v` 명령어를 통해 모델을 실행하고 있습니다.

여러분이 직접 실행하려면 백그라운드, 혹은 별도의 프로세스에서 아래의 코드를 실행해주세요.

```bash
ollama serve
ollama run minicpm-v
```

Note. 이전 실습에서는 `ollama pull llama3.1` 명령어를 통해 불러왔습니다. 두 방법 모두 유효한 방법이며, 모델의 종류에 따라 변경한 것은 아닙니다.


```python
llm = ChatOllama(model="minicpm-v", temperature=0)
embeddings = OllamaEmbeddings(model="minicpm-v")
```

### 1.2 PDF 페이지 추출 함수 정의

`PyMuPDF` 라이브러리를 활용하여 PDF 파일을 Base64 인코딩된 이미지로 변환하는 함수를 정의합니다.


```python
def pdf_page_to_base64(pdf_path: str, page_number: int):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number - 1)  # input is one-indexed
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")
```

논문 PDF 파일을 불러온 후, 이미지로 잘 인코딩 되었는지 확인합니다.


```python
file_path = "data/Llama3.1_Ch1~3.pdf"

base64_image = pdf_page_to_base64(file_path, 4)
display(IPImage(data=base64.b64decode(base64_image)))
```


    
![image](https://github.com/user-attachments/assets/31391de5-62e6-4d77-b070-8c43f5feec26)



## 2. VQA 체인 테스트

MiniCPM-V 모델과 ChatGPT-4o-mini API를 각각 활용하여 이미지와 텍스트를 입력으로 받아 답변을 생성하는 VQA 기능이 잘 작동하는지 테스트를 진행합니다.

### 2.1 MiniCPM-V 모델 기반 VQA 확인

로컬 LLM 중 하나인 MiniCPM-V 모델을 활용하여 VQA 체인이 잘 작동하는지 확인합니다.

앞서 테스트를 위해 입력한 페이지는 Llama 3 모델의 아키텍쳐와 학습 방법을 서술한 페이지입니다.

해당 페이지를 인코딩한 이미지를 잘 처리하는지 확인하기 위해, Llama 3가 어떤 구조로 이루어져 있고, 어떻게 학습했는지 영어로 물어봅시다.


```python
# 번역: "Llama 3 모델의 전체 아키텍처와 교육 절차를 설명해주세요."
query = "Explain me the overall architecture and training procedure of Llama 3 models."

# OpenAI와 비슷한 형태로 사용할 수 있습니다.
message = HumanMessage(
    content=[
        {"type": "text", "text": query},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        },
    ],
)
response = llm.invoke([message])
print(response.content)
```

    The figure illustrates that Llama 3 is a Transformer language model trained to predict text from a given sequence, using self-supervised learning methods where it reconstructs masked parts of speech inputs into discrete tokens for reconstruction tasks. The model learns the structure of spoken signals through vision adapter training and integrates pre-trained image encoders during adaptation phases. Speech adapter training involves converting audio recordings into textual formats that can be integrated with visual data in a fine-tuned language model, enhancing its ability to understand human speech dynamically.


초기 LLM 설정에서 `temperature` 값을 0으로 설정했기 때문에, 아래 텍스트와 같은 대답이 나올 것입니다:

`The figure illustrates that Llama 3 is a Transformer language model trained to predict text from a given sequence, using self-supervised learning methods where it reconstructs masked parts of speech inputs into discrete tokens for reconstruction tasks. The model learns the structure of spoken signals through vision adapter training and integrates pre-trained image encoders during adaptation phases. Speech adapter training involves converting audio recordings into textual formats that can be integrated with visual data in a fine-tuned language model, enhancing its ability to understand human speech dynamically.`

이 문장을 번역하면 아래와 같습니다:

`그림은 Llama 3가 주어진 시퀀스로부터 텍스트를 예측하도록 훈련된 Transformer 언어 모델임을 보여줍니다. 이 모델은 마스킹된 발화 입력 부분을 재구성 작업을 위해 이산 토큰으로 변환하는 자기 지도 학습 방법을 사용합니다. 모델은 비전 어댑터 훈련을 통해 발화 신호의 구조를 학습하고, 적응 단계에서 사전 훈련된 이미지 인코더를 통합합니다. 음성 어댑터 훈련은 오디오 녹음을 시각적 데이터와 통합할 수 있는 텍스트 형식으로 변환하여, 세밀하게 조정된 언어 모델에서 인간의 발화를 동적으로 이해할 수 있는 능력을 향상시킵니다.`

논문에 있는 설명과 유사한 설명을 조리있게 잘 작성한 것을 확인할 수 있습니다.

비슷한 질문을 한국어로 해봅시다.


```python
query = "Llama 3 모델의 전체 아키텍처와 학습 절차를 설명해 주세요."

# OpenAI와 비슷한 형태로 사용할 수 있습니다.
message = HumanMessage(
    content=[
        {"type": "text", "text": query},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        },
    ],
)
response = llm.invoke([message])
print(response.content)
```

    Llama 3 모델은 Transformer model으로 구축되어 있으며, 이 모델은 텍스트의 다음 토큰을 예측하는 데 사용되는 self-supervised 방법을 통해 학습된다.이 방식은 음성 입력과 시각적 인식을-masked output parts를 텍스트의 다음 토크나에서 추출하여 구체적인 텍스트를 예측한다.모델은 구조를 배우는 데 사용되는 signals를 추출하고, Section 7에 대한 세부 사항은 스펀서 커뮤니티에서 제공된다.


아래와 비슷한 한국어 답변을 생성하긴 하지만, 영어 버전에 비해 훨씬 엉성한 것을 확인할 수 있습니다

`Llama 3 모델은 Transformer model으로 구축되어 있으며, 이 모델은 텍스트를 다음 텍스트로 예측하는 self-supervised 방법을 사용합니다. 이 방법은 음성 입력의 일부 부분과 시각 정보를 추출하여重构masked output parts를 tekst-sequence-representation으로 변환한다.모델은 이러한 구조를 signals에 대한 확률을 학습한다. Vision encoder training model은 pre-trained language model을 적응시켜 사용할 수 있는 vision encoder model을 학습합니다. 이 모델은 시각 데이터와 음성 데이터의 상호작용을 관찰하고, 시각 데이터와 음성 데이터를 동일한 language representation으로 변환한다.`

### 2.2 (선택사항) ChatGPT-4o-mini API 기반 VQA 확인

Note. 이 챕터를 진행하기 위해서는 OpenAI API Key가 필요하며, API Key를 사용할 수 없는 상황이라면 이 부분은 건너뛰어도 무방합니다.

ChatGPT-4o-mini API를 활용하여 VQA 체인이 잘 작동하는지 확인합니다.

같은 질문을 ChatGPT-4o-mini API에 입력해봅시다.


```python
os.environ["OPENAI_API_KEY"] = "api key"
llm = ChatOpenAI(model="gpt-4o-mini")
```


```python
query = "Llama 3 모델의 전체 아키텍처와 학습 절차를 설명해 주세요."

# message는 MiniCPM-V 모델과 동일한 값을 적용합니다.
response = llm.invoke([message])
print(response.content)
```

    Llama 3 모델은 Transformer 구조를 기반으로 하는 언어 모델로, 다음의 단계로 아키텍처와 학습 절차를 설명할 수 있습니다:
    
    ### 1. 아키텍처
    
    - **입력 데이터 처리**: Llama 3는 다양한 입력 데이터를 처리하는 데 중점을 두고 있습니다. 주로 텍스트, 이미지 및 음성 데이터를 활용합니다.
      
    - **사전학습된 언어 모델**: 모델은 사전학습된 언어 모델을 기반으로 하여, 주어진 텍스트 시퀀스의 다음 토큰을 예측합니다. 이 과정에서, 다음의 구성 요소를 사용합니다:
      - **토큰 임베딩**: 입력 텍스트를 벡터로 변환하여 모델에 입력합니다.
      - **자기 주의(Self-Attention)**: 입력의 문맥 정보를 고려하여 각 단어의 중요도를 조정합니다.
      - **피드포워드 네트워크**: 각 토큰의 표현을 업데이트합니다.
    
    - **다중 모달학습**: Llama 3는 비전과 음성을 포함한 다양한 입력을 처리하기 위해 적응기를 훈련합니다. 
      - **비전 적응기**: 이미지 인코더를 사용하여 이미지 데이터를 모델에 통합합니다.
      - **음성 적응기**: 음성 데이터를 텍스트 표현으로 변환하여 모델에 결합합니다.
    
    ### 2. 학습 절차
    
    1. **데이터 수집 및 정제**:
       - 웹에서 다양한 출처를 통해 데이터를 수집하고, 개인 식별 정보(PII)와 안전에 관련된 콘텐츠를 필터링합니다.
    
    2. **사전 학습**:
       - 대규모의 텍스트 데이터셋을 기반으로 언어 모델을 사전 학습합니다. 이 과정에서, 모델은 여러 토큰의 맥락을 이해하고, 예측의 정확성을 높이기 위한 기술들을 개발합니다.
    
    3. **적응기 훈련**:
       - 비전 및 음성 데이터를 강화하기 위해 적응기를 훈련합니다. 이 과정에서는 각 데이터 타입에 적합한 구성 요소를 조정하여 언어 모델의 효율성을 높입니다.
    
    4. **파인 튜닝**:
       - 사전 학습 후, 특정 작업이나 도메인에 맞춰 모델을 조정합니다.
    
    ### 3. 결론
    
    Llama 3의 아키텍처와 학습 절차는 다중 모달 데이터를 처리하는 데 최적화되어 있으며, 이로 인해 텍스트, 이미지 및 음성의 입력을 이해하고 생성하는 능력을 갖추고 있습니다.


앞서 생성한 로컬 LLM 기반 한국어 답변보다 훨씬 퀄리티가 좋음을 확인할 수 있습니다.

이어지는 실습부터는 여러분들의 선택에 맞춰 로컬 LLM과 OpenAI API를 상호 교환하여 사용하셔도 좋습니다.

이제 이 모델을 활용해서 PDF 문서 기반 간단한 RAG 챗봇을 구현해봅시다.

## 3. Document 라벨 생성

로컬 LLM 혹은 OpenAI API를 활용하여 문서를 검색할 때 활용할 수 있는 설명 라벨을 생성합니다.

아래 코드의 주석을 적절하게 해제하여 원하는 LLM을 사용합니다.


```python
# 로컬 LLM을 활용하기 위해 이 코드의 주석을 해제해주세요.
# llm = ChatOllama(model="minicpm-v", temperature=0)

# OpenAI API를 활용하기 위해 이 코드의 주석을 해제해주세요.
# llm = ChatOpenAI(model="gpt-4o-mini")
```

각 페이지 별 이미지를 추출하여 RAG 시스템이 문서를 검색할 때 활용할 수 있는 설명 라벨을 생성합니다.


```python
# 이미지와 관련 질문을 입력 받아 대응하는 답변을 생성하는 함수입니다.
# 인자를 _dict 하나만 사용하는 이유는, RAG 체인을 구성할 때 추가 Adapter 없이 사용하기 위함입니다.
def query_about_image(_dict):
    query = _dict["query"]
    base64_image = _dict["base64_image"]
    if isinstance(base64_image, list):
        base64_image = base64_image[0]

    message = HumanMessage(
        content=[
            {"type": "text", "text": query},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ],
    )
    response = llm.invoke([message])
    return response.content
```

논문 PDF의 특정 페이지를 잘 요약하는지 확인합니다.


```python
base64_image = pdf_page_to_base64(file_path, 4)

query = """You are an assistant tasked with summarizing page. Give a concise summary of the page."""

query_about_image({"query": query, "base64_image": base64_image})
```




    "The document discusses the architecture and training of Llama 3, a Transformer language model designed to predict the next token in a text sequence. It highlights the use of a self-supervised learning approach for speech inputs and describes the integration of vision and speech adapters to enhance the model's capabilities. \n\nKey points include:\n\n1. **Adapter Training**: Integrating pre-trained language models with image and speech encoders to align and refine data representations.\n2. **Pre-Training Data**: Focuses on the curation and filtering of a large training dataset, including methods for ensuring the quality and safety of the data.\n3. **Web Data Curation**: The data comes from various web sources, with measures in place to filter out harmful content and personally identifiable information (PII).\n\nOverall, the document outlines the components and strategies employed to develop a robust multimodal language model."



영어로 요약했을 때는 잘 요약해주는 것 같습니다.

이제 모든 페이지에 대한 요약문을 생성합니다. 변환에는 약 1분 정도 소요됩니다.


```python
PDF_PAGE_COUNT = len(fitz.open(file_path))
print(PDF_PAGE_COUNT)

tqdm_iter = tqdm(range(1, PDF_PAGE_COUNT + 1))

page_summary_texts = []
base64_images = []
for page_idx in tqdm_iter:
    base64_image = pdf_page_to_base64(file_path, page_idx)
    query = """You are an assistant tasked with summarizing page. \
        Give a concise summary of the page."""
    response = query_about_image({"query": query, "base64_image": base64_image})

    page_summary_texts.append(response)
    base64_images.append(base64_image)
```

    15


    100%|██████████| 15/15 [01:19<00:00,  5.27s/it]


## 4. 간단한 RAG 구현

### 4.1. VectorDB 구성

생성한 Document 라벨을 활용하여 간단한 RAG를 구현합니다.

- `InMemoryStore`를 활용하여 텍스트 및 표 데이터를 저장합니다
- `vectorstore`에는 전체 문서를 임베딩하는 대신, 요약된 텍스트만 임베딩하여 저장합니다.


```python
vectorstore = Chroma(
    embedding_function=embeddings, collection_name="summaries"
)

store = InMemoryStore()
id_key = "doc_id"

# VectorDB에서 바로 Retriever를 생성하는 대신, 별도의 Retriever를 생성합니다.
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)
```


```python
# Add Encoded Images
img_ids = [str(uuid.uuid4()) for _ in base64_images]
summary_images = [
    Document(page_content=s, metadata={id_key: img_ids[i]})
    for i, s in enumerate(page_summary_texts)
]
retriever.vectorstore.add_documents(summary_images)
retriever.docstore.mset(list(zip(img_ids, base64_images)))
```

### 4.2. RAG 구성

RAG Chain을 구성하고, 의도대로 잘 작동하는지 테스트합니다.

Note. 로컬 LLM (MiniCPM-V) 모델을 활용하시는 경우, 영어로 프롬프트를 입력해야 정확한 결과를 얻을 수 있습니다.


```python
# RAG pipeline
chain = (
    {"base64_image": retriever, "query": RunnablePassthrough()}
    | RunnableLambda(query_about_image)
    | StrOutputParser()
)
```


```python
query = "Explain me the overall architecture and training procedure of Llama 3 models."

print(chain.invoke(query))
```

    The architecture and training procedure of Llama 3 models involve several advanced techniques designed to optimize performance and scalability. Here's an overview of key components:
    
    ### Architecture
    
    1. **Pipeline Parallelism**: 
       - Llama 3 employs a pipeline parallelism strategy divided into multiple stages (0 to 3). Each stage processes different parts of the model across a number of GPUs, allowing simultaneous operation to improve throughput.
       - The model is designed to maintain efficiency with micro-batch processing, where each stage operates on several small batches serialized over a shared sequence length.
    
    2. **Network-aware Parallelism**:
       - The architecture configures parallelism (referred to as TP, CP, DP, etc.) based on the requirements of network bandwidth and latency.
       - The inner layers focus on maximizing network bandwidth, while the outer layers are optimized for minimizing latency. This ensures effective data flow and reduced communication overhead during training.
    
    3. **Collective Communication**:
       - The model utilizes NVIDIA’s NCCL library to manage communication between GPUs effectively.
       - Multiple strategies for data transfer are employed, including chunking data to reduce the number of messages and optimizing point-to-point transfers to improve overall communication time.
    
    ### Training Procedure
    
    1. **Numerical Stability**:
       - Training stability is addressed by utilizing FP32 gradient accumulation. This process helps maintain precision during backpropagation while operating over multiple micro-batches.
    
    2. **Asynchronous Gradient Updates**:
       - The model uses asynchronous updates for its weights to prevent stalling during training, allowing for the flexibility of data flow.
    
    3. **Performance Projections**:
       - The training procedure involves performance estimation and projections based on different configurations to identify the most efficient setups for achieving desired training speeds.
    
    4. **Scaling and Configuration**:
       - Llama 3 is designed to support a large number of micro-batches and can project performance based on various configurations to ensure optimal resource utilization.
    
    ### Summary
    
    Overall, Llama 3’s architecture and training procedures focus on maximizing parallelism, optimizing communication between GPUs, maintaining numerical stability, and leveraging performance projections to address scalability and efficiency in large-scale training scenarios.



```python

```
