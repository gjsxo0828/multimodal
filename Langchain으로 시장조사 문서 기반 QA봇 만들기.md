# [실습2] Langchain으로 시장조사 문서 기반 챗봇 만들기 - PDF

## 실습 목표
---
[실습1] RAG를 위한 Vector Score, Retriever 에서 학습한 내용을 바탕으로 LangChain을 활용해서 입력된 문서를 요약해서 Context로 활용하는 챗봇을 개발합니다.

## 실습 목차
---

1. **시장조사 문서 벡터화:** RAG 챗봇에서 활용하기 위해 시장조사 파일을 읽어서 벡터화하는 과정을 실습합니다.

2. **RAG 체인 구성:** 이전 실습에서 구성한 미니 RAG 체인을 응용해서 간단한 시장 조사 문서 기반 RAG 체인을 구성합니다.

3. **챗봇 구현 및 사용:** 구성한 RAG 체인을 활용해서 시장조사 문서 기반 챗봇을 구현하고 사용해봅니다.

## 실습 개요
---
RAG 체인을 활용해서 시장조사 문서 기반 챗봇을 구현하고 사용해봅니다.

## 0. 환경 설정
- 필요한 라이브러리를 불러옵니다.

```python
from langchain.document_loaders import PyPDFLoader

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
```

- Ollama를 통해 Mistral 7B 모델을 불러옵니다.


```python
!ollama pull mistral:7b
```

## 1. 시장조사 문서 벡터화
- RAG 챗봇에서 활용하기 위해 시장조사 파일을 읽어서 벡터화하는 과정을 실습합니다.

먼저, mistral:7b 모델을 사용하는 ChatOllama 객체와 OllamaEmbeddings 객체를 생성합니다.
  
```python
llm = ChatOllama(model="mistral:7b") # 채팅할때 쓰겠다.
embeddings = OllamaEmbeddings(model="mistral:7b") # embedding 모델에 사용하겠다.
```

다음으로, 시장조사 PDF 문서를 불러와서 벡터화 해보겠습니다.
- 한국소비자원의 2022년 키오스크(무인정보단말기) 이용 실태조사 보고서를 활용했습니다
  - https://www.kca.go.kr/smartconsumer/sub.do?menukey=7301&mode=view&no=1003409523&page=2&cate=00000057
- 이 실태조사 보고서는 2022년 키오스크의 사용자 경험, 접근성, 후속 조치에 대해 논의하는 보고서입니다. 
- 이를 활용해서 키오스크를 어떻게 세일즈 할 수 있을지 아이디어를 제공하는 챗봇을 만들어야 하는 상황이라고 가정해 봅시다.

먼저, LangChain의 `PyPDFLoader`를 활용해서 시장조사 보고서의 텍스트를 추출하고, 페이지 별로 `Document`를 생성하여 저장합니다.

```python
doc_path = "docs/키오스크(무인정보단말기) 이용실태 조사.pdf"
loader = PyPDFLoader(doc_path) #로더 객체 생성
docs = loader.load() #load : PDF를 로드해서 문서 객체 리스트를 생성
```
생성된 Document의 수를 확인해봅시다.

```python
loader
```
> <langchain_community.document_loaders.pdf.PyPDFLoader at 0x7f106de3f9a0>

```python
docs
```
![image](https://github.com/user-attachments/assets/ff8cca96-f7b8-42ed-994e-f669d013dd1c)

```python
print(len(docs)) #페이지당 document 생성
```
> 59

다음으로, 각 Document의 길이를 확인해봅시다.

```python
#document 길이 = 추출된 문자 수
doc_len = [len(doc.page_content) for doc in docs]
print(doc_len)
```
> ![image](https://github.com/user-attachments/assets/4a8e9b7f-c766-44c3-97c1-2123257aea7a)

1천자 미만의 문서도 있지만, 6천자가 넘는 문서도 있는 것을 확인할 수 있습니다. 이대로 그냥 사용할 경우, Context가 너무 길어져 오히려 성능이 낮아질 수도 있습니다.

우선은 이대로 RAG 체인을 구성해 봅시다.

## 2. RAG 체인 구성
RAG 체인을 구성하기 위해 `Document`를 `OllamaEmbeddings`를 활용해 벡터로 변환하고, FAISS DB를 활용하여 저장합니다.
- 변환 및 저장 과정은 약 3분 정도 소요됩니다.

```python
# FAISS는 facebook에서 개발한 벡터 DB 모델
vectorstore = FAISS.from_documents(
    docs,
    embedding=embeddings
)
# vectorization, 저장
```
```python
#러너블한 retriever로 생성 -> 랭체인의 체인 구성 요소로서 사용할 수 있다. (FAISS는 Langchain에서 호환이 안됨)
# 체인 내 요소의 입출력 포맷이 동일하다.
db_retriever = vectorstore.as_retriever()
# 벡터 DB + 검색 기능 -> 검색기 (db_retriever), FIASS는 
```

이전 실습에서 구성한 미니 RAG Chain과 비슷하게 Chain을 구성해 봅시다.
- 지난 실습과 달리 이번 챗봇의 역할은 마케터를 위한 챗봇으로 고정했으므로, 역할을 별도로 인자로 전달할 필요가 없습니다.
- `RunnablePassthrough()`는 Chain의 이전 구성 요소에서 전달된 값을 그대로 전달하는 역할을 수행합니다.

```python
def get_retrieved_text(docs): #각 페이지에 존재하는 문자열 -> 하나의 문자열로 생성
    result = "\n".join([doc.page_content for doc in docs])
    return result

def init_chain():  #문맥 기반 질문 응답 체인을 생성
    #메세지 구조 생성 : 대화기록 (role : 내용)
    messages_with_contexts = [
        # system 은 답변의 기조, 가이드라인을 의미함, 단 확률적 선택에 의한 답변이므로 100% system을 준수하지는 않음
        ("system", "당신은 마케터를 위한 친절한 지원 챗봇입니다. 사용자가 입력하는 정보를 바탕으로 질문에 답하세요."),
        #
        ("human", "정보: {context}.\n{question}."),
    ]
    #대화 프롬프트 템플릿 객체 생성
    #ChatPropmtTemplate 전달하는 메세지를 표준화.
    prompt_with_context = ChatPromptTemplate.from_messages(messages_with_contexts)

    # 체인 구성
    # context에는 질문과 가장 비슷한 문서를 반환하는 db_retriever에 get_retrieved_text를 적용한 chain의 결과값이 전달됩니다.
    qa_chain = (
        {"context": db_retriever | get_retrieved_text, "question": RunnablePassthrough()}
        # db_retriever : 유저 쿼리 input -> 유사 문서 검색 k개 return List[Documnet()]
        # get_retrieved_text (문서 전체 내용) + question -> db_retriever (유사 문서 검색)
        #RunnablePassthrough : 랭체인의 러너블 인터페이스 중 하나
            #입력 데이터를 아무런 변경 없이 그대로 다음 체인 단계로 전달하는 역할, 
            #검색 결과는 가공, 질문은 그대로 넘기고 싶을때 사용
        | prompt_with_context #대화 프롬프트 템플릿 (db_retriever가 반환한 내용 context에 추가)
        | llm #답변 + 메타데이터 로 반환 (어떤 모델을 사용했는지, 토큰의 갯수 등)
        | StrOutputParser() #답변만 추출해서 문자열로 파싱
    )
    # LCEA
    # 파이프연산자  왼쪽 output을 오른쪽 input으로 사용

    return qa_chain


#? 그럼 함수 호출될때마다 기존 대화 기록이 저장될까 안될까? -> 안됨 -> 메모리 세션 없음
```
```python
qa_chain = init_chain()
```
> Chain 구성이 완료되었습니다.


## 3. 챗봇 구현 및 사용
- 구성한 RAG 체인을 활용해서 시장조사 문서 기반 챗봇을 구현하고 사용해봅니다.

방금 구현한 RAG Chain을 사용해서 시장조사 문서 기반 챗봇을 구현해볼 것입니다. 

그 전에, 별도로 RAG 기능을 추가하지 않은 LLM과 답변의 퀄리티를 비교해 봅시다.

```python
#rag기능 없는 기본 챗봇 생성 (성능 비교 위해)
messages_with_variables = [ #메세지 구조 생성
    ("system", "당신은 마케터를 위한 친절한 지원 챗봇입니다."),
    ("human", "{question}."),
]

prompt = ChatPromptTemplate.from_messages(messages_with_variables)
parser = StrOutputParser()
chain = prompt | llm | parser
```
```python
#rag 사용 x
print(chain.invoke("키오스크 관련 설문조사 결과를 한국어로 알려줘"))
```
> ![image](https://github.com/user-attachments/assets/723b36d2-aaba-46a4-af52-066c2ad1ccb5)

```python
#rag 사용
print(qa_chain.invoke("키오스크 관련 설문조사 결과를 한국어로 알려줘"))
```
![image](https://github.com/user-attachments/assets/24946f5b-c231-401c-8297-9db66f671e85)

일반 체인은 아무런 출처가 없는 답변을 생성한 반면, RAG 기능을 추가한 챗봇은 데이터를 기반으로 상대적으로 정확한 답변을 하는 것을 확인할 수 있습니다.

이제 챗봇을 한번 사용해 봅시다.


```python
qa_chain = init_chain()
while True:
    question = input("질문을 입력해주세요 (종료를 원하시면 '종료'를 입력해주세요.): ")
    if question == "종료":
        break
    else:
        result = qa_chain.invoke(question)
        print(result)
```

![image](https://github.com/user-attachments/assets/c9535416-0a4a-48cb-92dd-e635b5d6a183)

저희는 이전 챕터에서 구현한 챗봇이 가지고 있는 문제점 중 '문서나 데이터 기반 추론이 불가능하다.'를 완화했습니다.

또한, 지금 구성한 챗봇은 UI가 없고 단순 표준 입출력 만을 사용합니다. 5챕터에서 Streamlit을 활용해 ChatGPT와 비슷한 웹 챗봇 어플리케이션을 제작해 볼 것입니다.

