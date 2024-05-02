# KDT5_OpenCV_Project

![title](image.png)

## 주제 : 피자 낫피자

![주제](image-2.png)

![주제2](image-3.png)

## 팀 내용 (클릭 시 상세 내용 확인 가능)
<details>
<summary> 이윤서 <a href="https://github.com/voo0o08" height="5" width="10" target="_blank">
	<img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=white"/><a> : VGG16 / 직접 만든 모델</summary>
<div markdown="1">
    
## **0. 전처리 및 조건**


>✏️ **The inference transforms are available at `VGG16_Weights.IMAGENET1K_V1.transforms` and perform the following preprocessing operations: Accepts `PIL.Image`, batched `(B, C, H, W)` and >single `(C, H, W)` image `torch.Tensor` objects. The images are resized to `resize_size=[256]` using `interpolation=InterpolationMode.BILINEAR`, followed by a central crop of `crop_size=[224]`. Finally the values are first rescaled >to `[0.0, 1.0]` and then normalized using `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`.**


1. Bilinear 보간법 사용으로 크기 256으로 만들기
2. crop_size = 224 중앙크롭으로 자르기
3. 값의 크기를 0.0, 1.0으로 조정
4. mean=[0.485, 0.456, 0.406] 및 std=[0.229, 0.224, 0.225]으로 normalized

---

![image](https://github.com/voo0o08/KDT5_OpenCV_Project/assets/155411941/3cd47435-cd1d-41e5-90b3-32169fd63740)


---

![image](https://github.com/voo0o08/KDT5_OpenCV_Project/assets/155411941/b5b849f8-313f-48c0-a6d8-1e42192349b8)


---

## **1. 직접 만든 CNN**

![image](https://github.com/voo0o08/KDT5_OpenCV_Project/assets/155411941/8b849d77-7c56-43c1-a2ec-5c1d3c6f6626)


모델 구성

이진 분류를 위해 마지막에 Sigmoid()

**결과**

![image](https://github.com/voo0o08/KDT5_OpenCV_Project/assets/155411941/c5409452-5782-433a-8b47-6b94e1f999a9)
![image](https://github.com/voo0o08/KDT5_OpenCV_Project/assets/155411941/df47bc3e-4eae-4bea-ab2f-35458f20ebad)


test dataset acc : 71.68

## **2. 50 X 50 img VGG16**

![image](https://github.com/voo0o08/KDT5_OpenCV_Project/assets/155411941/5654a33c-1be5-4eb3-8969-fcb902c9ab49)


```python
# 사전 학습된 모델 로딩
import torchvision.models as models # 다양한모델패키지
model = models.vgg16(pretrained=True)

# 사전 훈련된 모델의 파라미터 학습 유무 설정 함수
def set_parameter_requires_grad(model, feature_extract = True):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False # 학습하는 것을 방지

set_parameter_requires_grad(model) # 함수 호출
```

$$
out= \frac{input+2 \ast padding-kernel}{stride} +1
$$

층을 지나 지나면 해당 공식에 따라 output의 크기가 나옴 

```python
# 분류기 부분을 이진 분류기로 수정하는 클래스 정의
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        # VGG16의 특성 추출기 부분만 가져오기
        self.features = model.features
        # VGG16의 특성 추출기의 출력 크기 계산
        self.num_features = 512 * 1 * 1  # VGG16은 입력 이미지를 224x224 크기로 처리하므로, 50x50으로 하면 위 공식에 따라 1x1로 출력됩니다.
        
        # 이진 분류를 위한 새로운 fully connected layer 정의
        self.fc = nn.Sequential(
            nn.Linear(self.num_features, 4096),  # 특성 추출기의 출력 크기를 입력으로 받음
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1),  # 이진 분류를 위한 출력 뉴런 수
            nn.Sigmoid()  # 이진 분류를 위한 시그모이드 활성화 함수
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x

# 모델 생성
model = BinaryClassifier()

# 특성 추출기 부분의 파라미터를 고정시킴
set_parameter_requires_grad(model)

# 모델 구조 확인
print(model)
```

```python
for param in model.fc.parameters(): # 완전연결층은학습
    param.requires_grad = True
```

완전 연결층은 내 데이터로 학습하도록 해줌 

**결과**

![image](https://github.com/voo0o08/KDT5_OpenCV_Project/assets/155411941/472aa5d9-094d-47a2-9552-615c7dd634f7)

![image](https://github.com/voo0o08/KDT5_OpenCV_Project/assets/155411941/f1e95e19-3282-41a5-8093-d993b5189e9e)


test dataset acc : 85.54

## **3.  224 X 224 img VGG16**

test dataset acc : 94.00

## 4. 서비스 구현(model_test_mini_VGG16.py)

```python
# 모델 클래스 생성
model = CNN()
model.load_state_dict(torch.load('model_VGG16.pth')) # 학습된 가중치가 저장된 파
model.eval() 
```

```python
# 이미지 보기(반복문 내의 일부)
            rgb_image = cv2.imread(pred_DS.imgs[cnt][0])
            (h, w) = rgb_image.shape[:2]
            center = (0, h - 5)

            if prediction.item() == 1:
                food = "pizza"
                cv2.putText(rgb_image, food, center, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

            else:
                food = pred_DS.imgs[cnt][0].split("\\")[-1][:-4]
                cv2.putText(rgb_image, "not pizza->"+food, center, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

            cv2.imshow(food, rgb_image)
            cv2.waitKey(0)
```

pred_DS.imgs에는 이미지의 순서대로 (경로, class), (경로, class)…가 있어 cnt가 증가할 때마다 구현을 위한 이미지의 경로에서 사진을 받아오게 된다. 해당 이미지의 예측 결과를 받아 pizza면 이미지 위에 피자라는 글자를 쓰고, pizza가 아니라면 not pizza와 파일명(음식명) 글자를 작성한다.

- 김치전 → 실패
- 오코노미야끼 → 성공
- 파전 → 성공
- 사과 → 성공
- 라자냐 → 실패
- 피자들 → 성공


# 결론

- 배치의 가중치와, 학습 전 가중치 초기화 단계, 스케줄러 등 학습에 영향을 주는 모듈들의 중요성을 알 수 있었음
- VGG의 경우 지정된 크기보다 작아도 상관없지만 지정 크기를 지킬 경우 가장 결과가 잘 나오는 것을 알 수 있음

</div>
</details>
   

<details>
<summary> 전진우 <a href="https://github.com/zeeenoo11" height="5" width="10" target="_blank">
	<img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=white"/><a> : Resnet / 직접 만든 모델 </summary>
<div markdown="1">
내용
</div>
</details>

<details>
<summary> 고우석 <a href="https://github.com/Gowooseo" height="5" width="10" target="_blank">
	<img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=white"/><a> : AlexNet / 직접 만든 모델 </summary>
<div markdown="1">
내용
</div>
</details>

<details>
<summary> 옥영신 <a href="https://github.com/YeongshinOk" height="5" width="10" target="_blank">
	<img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=white"/><a> : Resnet / 직접 만든 모델 </summary>
<div markdown="1">
내용
</div>
</details>

