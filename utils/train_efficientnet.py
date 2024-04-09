import os
import sys
import torch
import logging
import argparse

from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

sys.path.append('../models')
from model_efficientnet import EfficientPlant
from dataset import PlantDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=50, help='the number of train epochs')
    parser.add_argument('-b', '--batch', type=int, default=48, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.002, help='learning rate')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    ## 로그 세팅
    logging.basicConfig(filename='efficient.log',
                    filemode='w',
                    format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO
                    )
    
    ## 모델 파라미터 저장을 위한 코드.
    os.makedirs('./save', exist_ok=True)
    os.makedirs(f'./save/efficient_{args.epoch}_{args.batch}_{args.learning_rate}', exist_ok=True)
    
    ## 데이터 어그멘테이션을 validation set에 적용하지 않기 위해 따로 정의한다.
    transforms = {'train': T.Compose([T.Resize((384,384)),
                                      T.RandomRotation(30),
                                      T.RandomHorizontalFlip(),
                                      T.ToTensor(),
                                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ]),
                    'val': T.Compose([T.Resize((384,384)),
                                      T.ToTensor(),
                                      T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])}
    
    ## Train, Val 스플릿이 제공되지 않아서 같은 데이터셋을 불러온 뒤, transforms만 따로 정의.
    train_dataset = PlantDataset("../dataset", "train", transforms=transforms["train"])
    val_dataset = PlantDataset("../dataset", "train", transforms=transforms["val"])

    ## validation set의 사이즈는 전체 데이터셋의 0.2로 맞춰준다.
    indices = range(len(train_dataset))
    val_size = len(train_dataset)//5

    ## Subset을 사용해서 분할. train, val 데이터셋은 같은 데이터를 열람할 수 없다.
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_size:])

    ## 데이터로더 생성.
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True, num_workers=6)

    ## 디바이스 정의.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    ## 모델 정의
    model = EfficientPlant()
    model.to(device)

    ## 옵티마이저, 손실 함수 정의.
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()

    ## 모델 훈련
    for epoch in range(args.epoch):
        
        model.train()
        step_index = 0
        train_loss = []

        for batch, tensor in enumerate(tqdm(train_loader)):
            image = tensor['image'].to(device)
            data = tensor['data'].to(device)
            label = tensor['label'].to(device)
            
            pred = model(image, data)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        train_loss_avg = sum(train_loss)/len(train_loss)
        logging.info(f"Epoch {epoch} train_loss : {train_loss_avg}")

        ## 모델 이밸류에이션.
        model.eval()
        val_loss = []
        val_r2_score = 0

        with torch.no_grad():
            for batch, tensor in enumerate(tqdm(val_loader)):
                image = tensor['image'].to(device)
                data = tensor['data'].to(device)
                label = tensor['label'].to(device)

                pred = model(image, data)
                loss = criterion(pred, label)
                val_r2_score += r2_score(label.cpu().numpy(), pred.cpu().numpy())
                val_loss.append(loss.item())
                
            val_loss_avg = sum(val_loss)/len(val_loss)
            val_r2_score /= len(val_loader)

            logging.info(f"Epoch {epoch} val_loss : {val_loss_avg}")
            logging.info(f"Epoch {epoch} r2_score : {val_r2_score}")
        
        if not os.path.exists(f"./save/efficient_{args.epoch}_{args.batch}_{args.learning_rate}"):
            os.mkdir(f"./save/efficient_{args.epoch}_{args.batch}_{args.learning_rate}")

        torch.save(model.state_dict(),
                   f"./save/efficient_{args.epoch}_{args.batch}_{args.learning_rate}/{epoch}_val_loss:{round(val_loss_avg,3)}.pth")