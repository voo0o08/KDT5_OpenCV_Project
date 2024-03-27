# 모듈 로딩
import pandas as pd
import numpy as np

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
import cv2

import cv2

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # in rgb, 커널은 8개 준비고했고 사이즈는 3*3짜리얌
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)  # 커널이 8개면 결과도 8개
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        # 풀링을 줘서 줄임
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(12 * 12 * 16, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

        # 가중치 초기화 함수
        def initialize_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

        # 모델에 가중치 초기화 적용
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.conv1(x)  # 패딩이랑 커널사이즈 때문에 row, col의 크기는 변화 X => 50 50 8
        x = F.relu(x)
        x = self.pool(x)  # pooling 때문에 반갈죽 => 25 25 8
        x = self.conv2(x)  # 25 25 16
        x = F.relu(x)
        x = self.pool(x)  # pooling 또 반갈죽 => 12 12 16

        x = x.view(-1, 12 * 12 * 16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)  # sigmoid
        return x

# 모델 클래스를 초기화하고 모델 가중치를 로드합니다.
model = CNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 모델을 추론 모드로 설정합니다.

# 이미지 가져오기
from torchvision.datasets import ImageFolder

# 전처리
my_img_root = "." # 해당 경로 내에 있는 파일명이 곧 label이 되는 것

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

preprocessing = transforms.Compose([
    transforms.Resize((50, 50), interpolation=transforms.InterpolationMode.BILINEAR), # 1. resize
    # transforms.CenterCrop(224), # 2. 중앙크롭
    transforms.ToTensor(),  # 3. 값의 크기를 0~1로
    transforms.Normalize(mean=mean, std=std) # 4. normalized
])

pred_DS = ImageFolder(root=my_img_root, transform=preprocessing)
original_loader = DataLoader(pred_DS)


# GPU 연결 -> 물론 없음
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

# 예측
def predict(model, val_loader):
    model.eval()
    correct = 0
    cnt = 0
    with torch.no_grad():
        for image, label in val_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            label = label.float()
            output = model(image)
            # print(output.max(1, keepdim = True))
            prediction = output.round()  # 이진 분류에서는 반올림하여 0 또는 1로 변환
            # print(prediction)

            # 이미지 보기
            rgb_image = cv2.imread(pred_DS.imgs[cnt][0])
            if prediction.item() == 1:
                food = "pizza"
            else:
                food = pred_DS.imgs[cnt][0].split("\\")[-1][:-4]
            (h, w) = rgb_image.shape[:2]
            center = (0, h-5)
            cv2.putText(rgb_image, "not pizza->"+food, center, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            cv2.imshow(food, rgb_image)
            cv2.waitKey(0)


            correct += prediction.eq(label.view_as(prediction)).sum().item()
            cnt += 1
    test_accuracy = 100. * correct / len(val_loader.dataset)
    return test_accuracy

predict(model, original_loader)

# print(pred_DS.imgs)