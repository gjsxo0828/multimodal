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
> The image depicts a situation that is unusual because it shows a person standing on the back of a vehicle, seemingly holding onto a dry cleaning service advertisement. This is not a common sight and is likely staged for humorous or promotional purposes. It's an example of people being used in advertisements to draw attention and create a memorable image. The scene also includes elements typical of city life such as a taxi, a pedestrian with luggage, and a busy street, which adds to the unconventional nature of this scene. 

```python
print(llava_call("What’s happening in the scene?", "./images/confusing.jpg"))
```
>   In the scene, there is a person standing in the back of a taxi van. The person appears to be folding or handling some clothes or material, possibly ironing, while the vehicle is moving down what seems to be a city street with traffic and other vehicles around. This activity could suggest that the person may be providing laundry services within the taxi van itself. It's an unusual sight and it looks like a promotional event or perhaps a creative way to offer a service where the car becomes a mobile laundry station. 


```python
Image(filename='./images/mustache.png')
```
![image](https://github.com/user-attachments/assets/d0157b08-f75d-4e64-a12f-7b95349ccb6e)


```python
print(llava_call("What color are her eyes?", "./images/mustache.png"))
```
> The person in the image has blue eyes. 

```python
print(llava_call("What is the mustache made of?", "./images/mustache.png"))
```
>  The "mustache" in the image is made from bananas, not actual hair. It's a playful, creative way to incorporate fruits as a part of a costume or statement. 

```python
Image(filename='./images/pizza.png')
```
![image](https://github.com/user-attachments/assets/e172f24c-ebb9-4f52-8896-11c29af1152c)


```python
print(llava_call("How many slices of pizza are there?", "./images/pizza.png"))
```
>  There are eight slices of pizza on the plate. 

```python
print(llava_call("Is this a vegetarian pizza?", "./images/pizza.png"))
```
>  This pizza is not a vegetarian pizza. It appears to have meat on it, which makes it non-vegetarian. Specifically, the toppings include chunks of meat, suggesting that it is not suitable for those who follow a vegetarian diet. 

```python
Image(filename='./images/tree.png')
```
![image](https://github.com/user-attachments/assets/455c9044-7f62-43c6-b49d-c05e176b5cdf)


```python
print(llava_call("Is this person expecting company?", "./images/tree.png"))
```
>  Based on the image, it seems that the person is set up for a picnic, which usually suggests they are expecting company. There's food laid out and other items like cups and a frisbee, which could be used during social activities. However, there's only one person visible in the image. Without additional context, it's not possible to definitively determine if they are indeed expecting someone else or if this is just a solitary picnic setup.

```python
print(llava_call("What is just under the tree?", "./images/tree.png"))
```
>  The image shows a person sitting on the ground near a picnic table, with what appears to be a sunny day and some personal items around. There's also a sports ball visible near the person. Under the tree, there are several bags that might contain belongings or food for the picnic. 

```python
Image(filename='./images/eyesight.png')
```
![image](https://github.com/user-attachments/assets/d03d393b-ff3c-4d03-807d-ec4ac0f8c286)


```python
print(llava_call("Does it appear to be rainy?", "./images/eyesight.png"))
```
>  No, the image shows a sunny day with clear skies. The man is holding a banana up to his ear as if he's talking into it, which is not possible, suggesting a playful or humorous situation rather than any indication of rain.  

```python
print(llava_call("Does this person have 20/20 vision?", "./images/eyesight.png"))
```
>   The image you've provided shows a person holding a banana up to their face, seemingly taking a picture or examining it. However, the image quality is quite low and blurry, making it difficult to discern any fine details of the person's eyesight, as they are not visible in this photo.  
>  
> To determine if someone has 20/20 vision, a thorough eye examination would be necessary to check for refractive errors such as nearsightedness or farsightedness, and to assess visual acuity by reading letters on an eye chart. The image does not provide sufficient detail to make any accurate assessment about the person's vision. 


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
>   The image shows a raccoon standing on what appears to be the ground of a forest or wooded area. The raccoon is facing towards the left side of the frame, and it has a somewhat surprised or curious expression. It's walking in front of a small rock outcropping or embankment, which could indicate that it might be navigating an uneven path through its habitat. There are also some leaves on the ground, suggesting that the photo was taken during a time when trees shed their leaves, possibly autumn. The background is slightly blurred, but it looks like a natural environment with forest floor and some vegetation. 
