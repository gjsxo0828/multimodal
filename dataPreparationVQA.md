# [실습] VQA 모델을 활용한 이미지 데이터 전처리

## 실습 목표

다음 실습에서는 VQA 모델을 이용하여 이미지에 자동으로 캡션을 생성하는 방식으로 이미지에 대한 채팅을 진행할 것입니다. 이 실습에서는 VQA 모델이 어떤 질문에 대답할 수 있고 LLaVA 모델이 얼마나 높은 성능을 가지고 있는지 논문 [VQA: Visual Question Answering](https://arxiv.org/pdf/1505.00468) 에 제시된 예시를 동해 실습을 진행해봅니다.


## 실습 목차

1. **이미지 로드 및 변환**: 이미지를 로드하고 base64 형식으로 변환하여 LLaVA 모델에 전달할 수 있도록 준비합니다.
2. **VQA 논문에 제시된 질문 쿼리**: VQA: Visual Question Answering 모델에서 제시된 대로, free-form 형태의 질문들을 LLaVA 모델에 프롬프팅하고 결과를 확인합니다.
3. **이미지 캡셔닝**: 원하는 이미지를 자유롭게 다운로드하여 이미지 캡셔닝을 진행해봅니다.


## 0. 환경 설정

필요한 라이브러리를 불러옵니다.

```python
import base64
import requests
import json
import os

from IPython.display import Image
```

## 1. 이미지 로드 및 변환

디스크에 있는 이미지를 로드하고 base64 형식으로 변환하여 LLaVA 모델에 전달합니다.

```python
def image_to_base64str(image_path):
    with open(image_path, "rb") as image_file:
    #img > rb(바이너리 읽기 0, 1) 모드로 load > image_file
        return base64.b64encode(image_file.read()).decode('utf-8')
        # 이미지는 바이너리 데이터 형식인데, 그대로 전송하거나 저장하면 시스템 플랫폼에 따라 깨질 수 있음
        # b64encode : Base64(아스키 코드)로 변환 (문자열로 바꿔서, HTML, JSON, HTTP 등 텍스트 기반 시스템에도 안전하게 전송할 수 있음)
        #utf-8 : 전세계의 모든 문자를 컴퓨터가 이해할 수 이쓴 이진 데이터로 변환하는 가장 널리 쓰이는 문자 인코딩 방식
```

LLaVA 모델은 ollama 프레임워크를 통해 구동되고 있습니다. 사용자의 질문을 전송하고, 모델의 응답을 보기 좋게 변환합니다.

```python
def parse_response(response_text):
    response_jsons = response_text.split("\n") #줄바꿈 마다 문자열 분해 > list
    # 생성한 각 단어마다 json 형식으로 metadata(생성 모델, 날짜, 예측 종료 여부, 종료 이유 등) 도 함께 존재
    
    all_response = ""
    for response_json in response_jsons:
        try:
            all_response += json.loads(response_json)['response']
            #jsong.loads(response_json)
            #{'model': 'llava', 'created_At' : '2025-05-25T19:49:01.9232323', 'response': ' The', 'done': False}
        except:
            pass
        
    return all_response
```

```python
def llava_call(prompt, image_path):
    url = "http://localhost:11434/api/generate" #LLAVA 모델이 실행 중인 로컬 서버의 API 주소
    payload = { #API에 전송할 데이터 딕셔너리
        "model": "llava",
        "prompt": prompt,
        "images": [image_to_base64str(image_path)]
        #image_to_base64str : 이미지 파일을 Base64 문자열로 인코딩 > 리스트 (여러 img 처리 가능)
    }

    headers = { #HTTP(LLaVA 모델 서버(로컬에서 실행 중인 API)에게 요청할 헤더, JSON 형식임을 명시
        "Content-Type": "application/json"
    }
    #HTTP POST(데이터 전송) 요청
    response = requests.post(url, data=json.dumps(payload), headers=headers)
        #json.dumps : py dict > json
    return parse_response(response.text)
```

## 2. VQA 논문에 제시된 질문 쿼리

VQA 시스템은 단순히 이미지 안의 물체를 인식하는 것을 넘어, 다양한 종류에 대한 질문에 대답할 수 있어야 합니다. 이번 실습에서 진행할 질문들은 다음들과 같이 다양한 종류를 포함합니다.

* 세밀 인식 (Fine-grained recognition)
    * “이 피자에 어떤 종류의 치즈가 사용되었는가?”
* 객체 탐지 (Object detection)
    * “이 사진에 자전거가 몇 대 있나요?”
* 행동 인식 (Activity recognition)
    * “이 남자는 울고 있나요?”
* 지식 기반 추론 (Knowledge base reasoning)
    * “이 피자는 채식주의자가 먹을 수 있는 건가요?”
* 상식 추론 (Common sense reasoning)
    * “이 사람은 손님을 기다리고 있나요?”
    * “이 사람은 시력이 좋은 사람인가요?”

```python
Image(filename='./images/confusing.jpg')
```
![image](https://github.com/user-attachments/assets/004264b5-852c-4811-9176-64f5f9062d79)


```python
print(llava_call("What is unusual about this image?", "./images/confusing.jpg"))
```
>  The image appears to be digitally altered or a composite of two separate scenes. It shows a person sitting on top of a dryer, seemingly in motion with the vehicle, which is highly unusual as it's not a typical place for someone to sit while a vehicle is moving. The juxtaposition creates a surreal and comical effect, as laundry machines are stationary appliances found indoors, typically in laundromats or homes, rather than on top of cars in traffic. 

```python
print(llava_call("What’s happening in the scene?", "./images/confusing.jpg"))
```
>  The image shows a person standing on the back of a yellow taxi with their arms outstretched. They appear to be balancing or performing some action that requires stability. The person is dressed in casual clothes, and there's no clear context provided by the background, which includes a city street with other vehicles and buildings that suggest an urban environment. This scenario seems unusual and possibly staged for entertainment or artistic purposes. 


```python
Image(filename='./images/mustache.png')
```
![image](https://github.com/user-attachments/assets/d0157b08-f75d-4e64-a12f-7b95349ccb6e)


```python
print(llava_call("What color are her eyes?", "./images/mustache.png"))
```
> The individual in the image has blue eyes. 

```python
print(llava_call("What is the mustache made of?", "./images/mustache.png"))
```
>  The "mustache" in the image appears to be a banana, with the stem forming the outline and the flesh serving as the color. It's an interesting example of food photography that creatively presents a common fruit. 

```python
Image(filename='./images/pizza.png')
```
![image](https://github.com/user-attachments/assets/e172f24c-ebb9-4f52-8896-11c29af1152c)


```python
print(llava_call("How many slices of pizza are there?", "./images/pizza.png"))
```
>  The image shows a slice of pizza with toppings, but the number of slices is not visible in the photo. There is one slice that can be seen clearly, but the rest of the pizza is out of view or obscured by the top of the slice.

```python
print(llava_call("Is this a vegetarian pizza?", "./images/pizza.png"))
```
>  The image you've provided appears to be a deep-dish pizza with several toppings. It has a layer of what looks like tomato sauce, cheese, meat that could be sausage or ground beef, and vegetables such as onions, olives, and possibly peppers. While the pizza contains vegetables, which are common in vegetarian pizzas, it also includes meat, which would not make it vegetarian according to traditional definitions of a vegetarian diet. 

```python
Image(filename='./images/tree.png')
```
![image](https://github.com/user-attachments/assets/455c9044-7f62-43c6-b49d-c05e176b5cdf)


```python
print(llava_call("Is this person expecting company?", "./images/tree.png"))
```
>  The image shows an individual sitting under a tree with a picnic spread out. There's food on the table, and it looks like it could be a pleasant outdoor setting for someone to join for a meal or conversation. However, there is no other person in sight, so the expectation of company might not be realized immediately. The setting appears inviting and relaxed, which might imply that the person is waiting for others to arrive. 

```python
print(llava_call("What is just under the tree?", "./images/tree.png"))
```
> 
 Under the tree, there appears to be a person sitting on a bench or lying down on a picnic blanket with their feet up. There are also some items scattered around that look like personal belongings and possibly food or drinks for a picnic. 

```python
Image(filename='./images/eyesight.png')
```
![image](https://github.com/user-attachments/assets/d03d393b-ff3c-4d03-807d-ec4ac0f8c286)


```python
print(llava_call("Does it appear to be rainy?", "./images/eyesight.png"))
```
>  No, the image shows a clear and sunny day. The weather appears to be pleasant with no signs of rain in the image provided. 

```python
print(llava_call("Does this person have 20/20 vision?", "./images/eyesight.png"))
```
>  In the image, you've provided a person holding up what appears to be a banana. The person seems to be posing for a photo and is standing outdoors, likely on a hill or embankment given the slope of the ground beneath them. It's not possible to determine if someone has 20/20 vision just by looking at an image; that would require assessing their visual acuity in person. However, the individual appears to be focused and is holding up something that looks like a banana with care, which could suggest good vision or simply an interest in the item they're holding. 


## 3. 이미지 캡셔닝

이미 주어진 이미지에 대해 캡셔닝을 진행하거나, 원하는 이미지를 URL을 통해 다운로드 받아 캡셔닝을 진행해 봅시다. 캡셔닝을 진행하기 위한 프롬프트를 변경하며 어떤 응답이 나오는지 확인해 봅니다.

```python
def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        # 이미지를 한번에 모두 메모리에 올리지 않고, 청크 단위로 스트리밍하여 메모리 효율이 높음
        response.raise_for_status()
        # HTTP 오류 발생 시 예외를 발생시켜 아래 코드 실행을 중단하고 except로 이동
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # makedirs : 저장 경로의 상위 폴더가 없으면 자동으로 생성
        # exit_ok = True : 근데 이미 폴더가 있었다면 에러가 없이 넘어감
        with open(save_path, 'wb') as file:
            # wb 모드 : 바이너리로 파일을 저장(이미지, 오디오 등은 반드시 바이너리 모드 필요)
            for chunk in response.iter_content(chunk_size=8192): # 바이트 단위
                # iter_content : 요청/응답 데이터를 반복하여 대용량 응답을 위해 conetnet를 청크단위로 읽음
                file.write(chunk) # 읽어온 청크 단위 파일 저장
        
        print("다운로드가 완료되었습니다.")
    except requests.RequestException as e:
        print(f"다운로드 실패: {e}")
```

```python
download_image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Raccoon_in_Central_Park_%2835264%29.jpg/440px-Raccoon_in_Central_Park_%2835264%29.jpg",
    "./images/raccoon.jpg"
)
```


```python
Image(filename='./images/raccoon.jpg')
```

![image](https://github.com/user-attachments/assets/227fd04f-336b-4b85-a665-0b7659509764)

```python
print(llava_call("Explain this image.", "./images/raccoon.jpg"))
```
>  The image shows a raccoon in an outdoor setting that appears to be a park or nature area, as suggested by the presence of trees and what looks like a path or trail. The raccoon is standing on all fours, looking forward, possibly searching for food or exploring its surroundings. It has typical raccoon features such as a bushy tail, a pointed ear, and a mask-like facial pattern with black patches around the eyes and ears.  
> 
>  The background suggests a cooler climate due to the presence of fallen leaves on the ground, which might indicate that the photo was taken in autumn or winter. The raccoon's fur appears wet, which could suggest it has recently been exposed to water or is living in an area with damp conditions.  
>
>  This type of raccoon is common in North America and is known for its adaptability and intelligence. They are often found near human habitation as they can be opportunistic feeders, scavenging through garbage dumps or urban areas. 
