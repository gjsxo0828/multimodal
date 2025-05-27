# [실습] Image Captioning을 적용한 RAG 챗봇 구현

## 실습 목표
---
우리가 처리하는 대부분의 문서는 텍스트 외에도 다양한 형식을 가진 정보를 포함하고 있습니다. 이미지가 그 대표적인 예 중 하나입니다. RAG 애플리케이션이 텍스트만을 처리한다면, 이미지가 가진 정보는 모두 유실되게 됩니다. 

이번 실습에서는 Multimodal LLM인 LLaVA를 이용하여 이미지에서 캡션을 자동으로 생성하고, 이 캡션과 텍스트를 동시에 고려하여 RAG 애플리케이션을 만들어 보겠습니다. 실습을 위해 여러 개의 이미지가 들어간 PDF 문서를 활용하여 진행해 보겠습니다. LLaVA 외에도 LLM 모델로 Llama 3.1-8B 를 이용하여 실습을 진행하겠습니다. 두 모델은 모두 Ollama 를 이용해 로컬 GPU에 호스팅 되어 있습니다.

## 실습 목차
---
1. **PDF 로드 및 변환**: 준비된 PDF, 혹은 원하는 PDF를 업로드하여 문서 내에 있는 텍스트와 이미지를 추출합니다. 추출을 위해서 `unstructured` 라이브러리를 사용합니다.
2. **벡터 인덱싱**: PDF 에서 추출한 이미지와 텍스트에서 요약 (summary) 혹은 이미지 캡션을 만듭니다. 이것에 기반해 벡터화와 인덱싱을 진행합니다.
3. **벡터 스토어에 저장**: 전 단계에서 추출한 벡터를 벡터 스토어에 저장합니다. 이번 실습에서는 외부 데이터베이스를 쓰지 않고 in-memory 데이터베이스를 활용합니다.
4. **RAG**: 벡터 스토어를 기반으로 RAG 애플리케이션을 제작합니다.

## 0. 환경 설정

필요한 라이브러리를 불러옵니다.


```python
import base64
import os
import uuid
import io
import re

from langchain_text_splitters import CharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from unstructured.partition.pdf import partition_pdf
from IPython.display import HTML, display, Markdown
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image
```

## 1. PDF 로드 및 변환

준비된 PDF 를 기반으로 문서를 로드하고 문서 내에 있는 텍스트와 이미지를 추출해 보겠습니다. 예시 PDF는 위키피디아 영문판에 업로드 된 삼성전자 문서입니다. 삼성전자의 로고 변천사와 같은 이미지를 RAG 를 통해 제대로 추출하고 답변을 진행할 수 있을지 확인해 봅시다.

![image-4.png](image-4.png)


```python
# PDF 에서 엘리멘트를 추출합니다.
def extract_pdf_elements(filepath):
    return partition_pdf(
        filename=filepath,
        extract_images_in_pdf=True,
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir="./images",
    )


# PDF 에서 텍스트를 추출합니다.
def extract_text(raw_pdf_elements):
    texts = []
    for element in raw_pdf_elements:
        texts.append(str(element))
    return texts

# 준비된 삼성전자 위키피디아 PDF 를 입력합니다.
pdf_elements = extract_pdf_elements("./samsung_electronics_wikipedia.pdf")
texts = extract_text(pdf_elements)

# 1000 개의 토큰을 기준으로 텍스트를 나눕니다.
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)
joined_texts = " ".join(texts)
splitted_texts = text_splitter.split_text(joined_texts)
```

## 2. 벡터 인덱싱

Ollama 에 의해 로컬로 호스팅된 Llama 3.1 LLM을 이용해 이미지와 텍스트에 대해 캡셔닝 혹은 요약본을 생성해보겠습니다. 일반적으로 RAG 애플리케이션을 만들 때 텍스트는 별다른 처리 없이 사용하기도 하지만, LLM을 이용해 요약본을 생성하는 것이 성능을 높이는데 도움이 됩니다.

이번 실습에서는 Anthropic 에서 발표한 [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval) 기술을 적용해 보겠습니다.

![image.png](image.png)


```python
# 텍스트 엘리멘트에서 컨텍스트를 추출합니다.
def generate_text_contexts(texts):
    # Contextual Retrieval 을 진행합니다.
    prompt_text = f"""
    <document>
    {joined_texts.replace('{', '(').replace('}', ')')}
    </document>
    Here is the chunk we want to situate within the whole document:
    <chunk>
    {{element}}
    </chunk>
    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Llama 3.1 모델을 이용해 요약본을 생성합니다.
    model = OllamaLLM(model="llama3.1")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # 동시에 5 개의 쿼리를 진행하며 배치 프로세싱을 진행합니다.
    text_contexts = summarize_chain.batch(texts, {"max_concurrency": 5})
    
    return text_contexts

text_contexts = generate_text_contexts(splitted_texts)
```

컨텍스트 추출이 잘 되었는지 한번 확인해 봅시다.


```python
text_contexts[0][:300]
```


```python
splitted_texts[0][:300]
```

컨텍스트 기반 추출 및 요약이 깔끔하게 된 것을 알 수 있습니다. 다음으로 이미지 캡셔닝을 진행해 보겠습니다. 이전 실습에서 진행했던 프롬프트와 한번 비교해 보시기 바랍니다.


```python
# 이전 실습과 같은 방식으로 base64 인코딩을 진행합니다.
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 주어진 이미지와 프롬프트를 이용해 이미지에 대한 캡션을 생성합니다.
def image_summarize(img_base64, prompt):
    llava = OllamaLLM(model="llava")
    
    llm_with_image_context = llava.bind(images=[img_base64])
    msg = llm_with_image_context.invoke(prompt)

    return msg

# images 디렉토리 내에 저장된 이미지들에 대해 캡셔닝을 수행합니다.
def generate_img_captions(path):
    img_base64_list = []
    image_summaries = []

    # 이미지 캡셔닝을 위한 프롬프트입니다.
    prompt = """You are an assistant tasked with describing images. \
    These description will be embedded and used to retrieve the raw image. \
    Give a detailed description of the image."""

    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))

    return image_summaries

image_captions = generate_img_captions("./images")
image_contexts = generate_text_contexts(image_captions)
```

## 3. 벡터 스토어에 저장

만들어진 텍스트와 이미지, 그리고 각 텍스트와 이미지에 대응하는 요약본과 캡션을 in-memory 벡터 스토어에 저장하겠습니다.

먼저 Contextual Retrieval 을 통해 만든 컨텍스트와 raw text 를 합치고, 이미지 캡션을 뒤이어 저장하겠습니다.


```python
documents = []
for i in range(len(text_contexts)):
    documents.append(Document(page_content=f"{text_contexts[i]}\n\n{splitted_texts[i]}"))

for i in range(len(image_contexts)):
    documents.append(Document(page_content=f"{image_contexts[i]}\n\n{image_captions[i]}"))
```


```python
# Llama 3.1 임베딩 모델을 사용합니다.
embeddings = OllamaEmbeddings(model="llama3.1")

# 가장 기본적인 In-memory 벡터 스토어를 만듭니다.
vector_store = InMemoryVectorStore(embeddings)
_ = vector_store.add_documents(documents)
```


```python
# Retriever를 만듭니다.
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 6}
)
```

## 4. RAG

이제 모든 준비가 완료되었습니다. RAG 애플리케이션을 제작합니다.


```python
# RAG 를 위한 프롬프트 생성 함수입니다.
def prompt_func(data_dict):
    formatted_context = "\n".join([
        d.page_content for d in data_dict["context"]
    ])
    
    text_message = f"""
    <context>
    {formatted_context}
    </context>
    
    Given the context, answer the user's question. Just clearly answer the given question.
    <question>
    {data_dict['question']}
    </question>
    """
    
    return text_message


def multi_modal_rag_chain(retriever):
    # Ollama 기반 llama 3.1 모델을 사용합니다.
    model = OllamaLLM(model="llama3.1")

    # RAG 파이프라인
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(prompt_func)
        | model
        | StrOutputParser()
    )

    return chain

# RAG 체인을 만듭니다.
chain_multimodal_rag = multi_modal_rag_chain(retriever)
```

### 테스트


```python
query = "What products does Samsung make?"
display(Markdown(chain_multimodal_rag.invoke(query)))
```


```python
query = "When was the Samsung logo with red used?"
display(Markdown(chain_multimodal_rag.invoke(query)))
```

### 성능

이미지 하나에 대한 단편적인 답변은 가능하지만, 이미지와 그 이미지가 쓰인 위치, 즉 컨텍스트를 종합적으로 고려한 질문에 대한 성능은 다소 떨어지는 것을 확인할 수 있습니다.

계속된 실습에서는 문서 내의 레이아웃과 테이블 데이터, 그리고 이미지를 종합적으로 고려하여 더 높은 정확도로 대답을 하는 봇을 구현해 보겠습니다.
