import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

from dataset import PlantDataset
from evaluation import evaluate
from model1 import Model1
from model2 import Model2
from model3 import Model3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=int, choices=[1, 2, 3], required=True, help='model architecture to train')
    parser.add_argument('-e', '--epoch', type=int, default=0, help='the number of train epochs')
    parser.add_argument('-b', '--batch', type=int, default=0, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    os.makedirs('./save', exist_ok=True)
    os.makedirs(f'./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}', exist_ok=True)
    
    img_size = 256
    crop_size = 224
    max_rotation = 30
    transforms = T.Compose(
        [T.Resize(img_size),
         T.RandomHorizontalFlip(),
         T.RandomRotation(max_rotation),
         T.RandomResizedCrop(crop_size, scale=(0.5, 1.0)),
         T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
         T.ToTensor(),
         T.RandomErasing(),
         ])

    dataset = PlantDataset("./dataset", "train/", transforms=transforms)
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(num_train * 0.8)

    # 데이터셋을 분할하기 위해 인덱스를 무작위로 섞음
    np.random.shuffle(indices)

    # train set과 validation set을 나눔
    train_idx, val_idx = indices[:split], indices[split:]

    # SubsetRandomSampler를 사용하여 데이터로더에 전달할 데이터를 선별
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # 데이터로더 생성
    train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=64, sampler=val_sampler)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    if args.model == 1:
        model = Model1()
    elif args.model == 2:
        model = Model2()
    elif args.model == 3: 
        model = Model3()
    else:
        raise ValueError("model not supported")
    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(args.epoch):
        
        model.train()
        step_index = 0
        train_loss = []

        for batch, tensor in enumerate(tqdm(train_loader)):
            images = tensor['image'].to(device)
            infos = torch.stack(tensor['info'], dim = 1)
            infos = infos.to(device).float()
            labels = torch.stack(tensor['label'], dim = 1)
            labels = labels.to(device).float()
            
            pred = model(images, infos)
            loss = criterion(pred, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.append(loss.item())


        model.eval()
        total = 0
        correct = 0
        val_score = 0

        with torch.no_grad():
            for batch, tensor in enumerate(tqdm(val_loader)):
                images = tensor['image'].to(device)
                infos = torch.stack(tensor['info'], dim = 1)
                infos = infos.to(device).float()
                labels = torch.stack(tensor['label'], dim = 1)
                labels = labels.to(device).float()
                total += len(images)

                outputs = model(images, infos)
                val_score = evaluate(outputs, labels)
        
        if not os.path.exists(f"./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}"):
            os.mkdir(f"./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}")

        torch.save(model.state_dict(),
                   f"./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}/{epoch}_score:{round(val_score,3)}.pth")