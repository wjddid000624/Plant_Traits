import os
import torch
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import Dataset
from torchvision import transforms as T
from preprocess import NumericPreprocessor

class PlantDataset(Dataset):
    def __init__(self, root, mode, transforms=None):

        """
        식물 데이터를 위한 커스텀 데이터셋 클래스.

        Parameters:
        - root: 루트 디렉토리. 누군가가 깃허브를 조져놓지 않는 이상 ./dataset으로 사용하면 됩니다.
        - mode: "train" / "test"
        - transforms: 이미지 변환 옵션. main에서 알아서 구현해서 쓰세요.

        Variables:
        - totensor: 이미지 텐서 변환용.
        - train_data: 훈련 데이터프레임.
        - test_data: 테스트 데이터프테임.
        - processor: NumericPreprocessor
        - items: 전처리후 NumericPreprocessor가 내뱉는 딕셔너리.
        - image: items 내부의 이미지 패스.
        - data: items 내부의 식물 보조 데이터.
        - label: items 내부의 라벨 데이터.

        Information:
        PCA 가중치 공유를 위해서 NumericPreprocessor가 train, test 데이터를 전부 필요로 한다.
        그래서 데이터셋 클래스도 test, train 전부 함께 넣어주고 모드를 따로 설정해줘야 한다는 점...
        좀 불편해도 이 방식이 그나마 제일 편한거니까 다들 참고하도록. 참으라고 ^^
        """

        self.root = root
        self.mode = mode
        self.transforms = transforms
        self.totensor = T.ToTensor()
        self.train_data = pd.read_csv(os.path.join(root, "train/train.csv"))
        self.test_data = pd.read_csv(os.path.join(root, "test/test.csv"))
        self.processor = NumericPreprocessor(root=self.root, train_data=self.train_data, test_data=self.test_data, mode=self.mode)
        self.items = self.processor.preprocess()
        self.image = self.items["image"]
        self.data = self.items["data"]
        self.label = self.items["label"]

    def __len__(self):

        """
        Returns the length of the dataset.

        Returns:
        - Length of the dataset based on the mode (train or test).
        """

        if self.mode == "train":
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    def __getitem__(self, index):

        """
        Fetches an item from the dataset.

        Parameters:
        - index: Index of the item to retrieve.

        Returns:
        - Dictionary containing image, data, and optionally label.
        """

        data = torch.tensor(self.data.iloc[index,:], dtype=torch.float32)
        
        image = Image.open(self.image[index])
        if self.transforms:
            image = self.transforms(image)
        else:
            image = self.totensor(image)

        if self.mode == "train":
            label = torch.tensor(self.label.iloc[index,:], dtype=torch.float32)
            return {'image': image, 'data': data, 'label': label}
        
        return {'image': image, 'data': data}