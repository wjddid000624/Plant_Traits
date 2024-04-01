import os
import csv
import torch
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class PlantDataset(Dataset):
    def __init__(
        self, 
        root: str, 
        split: str,
        transforms=None
    ):
        self.root = root
        self.transforms = transforms
        self.split = split
        self.totensor = T.ToTensor()
        self.data = self.prepare_dataset()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        ## 이미지 텐서로 변환 후 리턴
        image = Image.open(self.data[index][0])
        if self.transforms:
            image = self.transforms(image)
        else:
            image = self.totensor(image)
        ## 식물에 대한 지리적 정보
        info = self.data[index][1]
        ## 식물 특성값
        label = self.data[index][2]
        return {
            'image': image,
            'info': info,
            'label': label 
        }
    
    def prepare_dataset(self):
        ## root는 "./dataset", split은 "test", "train"으로 지정.
        split_base = os.path.join(self.root, self.split)
        ## 이미지 이름 순으로 정렬한 리스트
        image_path = os.path.join(split_base, "image")
        images = sorted(os.listdir(image_path))
        ## csv 파일에서 훈련에 사용할 정보와 타겟값들 불러오기
        csv = pd.read_csv(os.path.join(split_base, "info.csv"))
        ## 이미지 이름과 순서 맞추기 위해 id 순으로 정렬
        csv = csv.sort_values(by="id")
        ## 타겟 값에 대한 표준편차 열 제거.
        csv.drop(columns=["X4_sd","X11_sd","X18_sd","X26_sd","X50_sd","X3112_sd"], inplace=True)
        ## csv 데이터프레임에서 정보와 라벨 분리.
        label_col = ["X4_mean","X11_mean","X18_mean","X26_mean","X50_mean","X3112_mean"]
        info_col = [column for column in csv.columns if column not in label_col]
        ## 데이터 저장할 빈 리스트 생성.
        data = []
        ## [이미지 경로, 정보, 라벨]
        for idx, image_name in enumerate(images):
            data.append([os.path.join(image_path, image_name),
                         list(csv[info_col].iloc[idx]),
                         list(csv[label_col].iloc[idx])])
        return data