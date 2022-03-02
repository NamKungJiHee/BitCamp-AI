import random, os
import warnings
warnings.filterwarnings('ignore')

from glob import glob
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd

from PIL import Image

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations import Compose, OneOf, Resize, Normalize
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Device = {DEVICE}')

def seed_everything(seed = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Seed set = {seed}')
    
seed_everything()

path1 = '../_data/dacon/computer/'
submission = pd.read_csv("../_data/dacon/computer/sample_submission.csv")
train_imgs = glob('train/*/*')
train_labels = [path.split('\\')[1] for path in train_imgs]
test_imgs = glob('test/*')


label_map = {
    'airplane' : 0, 
    'automobile': 1, 
    'bird': 2,
    'cat': 3, 
    'deer': 4, 
    'dog': 5, 
    'frog': 6,
    'horse': 7, 
    'ship': 8, 
    'truck': 9,
}

class train_dataset(Dataset):
    def __init__(self, imgs, labels, transform = None):
        super(train_dataset, self).__init__()
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        X = np.array(Image.open(self.imgs[idx]))
        y = np.eye(10)[label_map[self.labels[idx]]]
        
        if self.transform:
            img = self.transform(image = X)['image']
        else:
            img = X
        y = torch.tensor(y, dtype = torch.float32)
        return img, y
    
class test_dataset(Dataset):
    def __init__(self, imgs, transform = None, n_tta = None):
        super(test_dataset, self).__init__()
        self.imgs = imgs
        self.transform = transform
        self.n_tta = n_tta
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        X = np.array(Image.open(self.imgs[idx]))
        if self.transform:
            if self.n_tta:
                imgs = [self.transform(image = X)['image'] for _ in range(self.n_tta)]
                return imgs
            else:
                img = self.transform(image = X)['image']
                return img
        else:
            return X

train_imgs, val_imgs, train_labels, val_labels = train_test_split(train_imgs, train_labels, test_size = 0.2, 
                                                                  stratify = train_labels, random_state = 42)
#len(train_imgs), len(val_imgs), len(train_labels), len(val_labels),


class Conv_Block_x2(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv_Block_x2, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels, out_channels, (3, 3), stride, padding = (1, 1), bias = False)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        
        self.conv_2 = nn.Conv2d(out_channels, out_channels, (3, 3), 1, padding = (1, 1), bias = False)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, (1, 1), stride, bias = False)
        self.bn_shortcut = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.relu(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        
        shortcut = self.conv_shortcut(inputs)
        shortcut = self.bn_shortcut(shortcut)
        
        x += shortcut
        outputs = self.relu(x)
        return outputs
    
class Identity_Block_x2(nn.Module):
    def __init__(self, channels):
        super(Identity_Block_x2, self).__init__()

        self.conv_1 = nn.Conv2d(channels, channels, (1, 1), bias = False)
        self.bn_1 = nn.BatchNorm2d(channels)
        
        self.conv_2 = nn.Conv2d(channels, channels, (3, 3), 1, padding = (1, 1), bias = False)
        self.bn_2 = nn.BatchNorm2d(channels)
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.relu(x)
        
        x = self.conv_2(x)
        x = self.bn_2(x)
        
        x += inputs
        outputs = self.relu(x)
        return outputs
    
class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 64, (7, 7), 2, padding = (3, 3), bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3), 2, padding = (1, 1)),
        )
        
        self.block_2 = nn.Sequential(
            Identity_Block_x2(64),
            Identity_Block_x2(64),
            Identity_Block_x2(64),
        )
        
        self.block_3 = nn.Sequential(
            Conv_Block_x2(64, 128, 2),
            Identity_Block_x2(128),
            Identity_Block_x2(128),
            Identity_Block_x2(128),
        )
        
        self.block_4 = nn.Sequential(
            Conv_Block_x2(128, 256, 2),
            Identity_Block_x2(256),
            Identity_Block_x2(256),
            Identity_Block_x2(256),
            Identity_Block_x2(256),
            Identity_Block_x2(256),
        )
        
        self.block_5 = nn.Sequential(
            Conv_Block_x2(256, 512, 2),
            Identity_Block_x2(512),
            Identity_Block_x2(512),
        )
        
        self.classifier = nn.Linear(512, 10)
        
    def forward(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = torch.mean(x, axis = [2, 3])
        outputs = self.classifier(x)
        return outputs


N_EPOCH = 50
LR = 1e-3
BATCH_SIZE = 100
N_TTA = 5

train_transform = Compose([
    Resize(224, 224),
    A.HorizontalFlip(),
    OneOf([
        A.IAAAdditiveGaussianNoise(),
        A.GaussNoise(),
        ], p = 0.2),
    OneOf([
        A.MotionBlur(blur_limit = 3, p = 0.2),
        A.MedianBlur(blur_limit = 3, p = 0.1),
        A.Blur(blur_limit = 3, p = 0.1),
        ], p = 0.2),
    A.ShiftScaleRotate(rotate_limit = 15),
    OneOf([
        A.OpticalDistortion(p = 0.3),
        A.GridDistortion(p = 0.1),
        A.IAAPiecewiseAffine(p = 0.3),
        ], p = 0.2),
    OneOf([
        A.CLAHE(clip_limit = 2),
        A.IAASharpen(),
        A.IAAEmboss(),
        A.RandomBrightnessContrast(),
        ], p = 0.3),
    A.HueSaturationValue(p = 0.3),
    Normalize(),
    ToTensorV2(),
])

val_transform = Compose([
    Resize(224, 224),
    Normalize(),
    ToTensorV2(),
])

def display_aug(imgs, transform, labels = None, n_aug = 5, cols = 5):
    idx = random.randint(0, len(imgs) - 1)
    
    plt.imshow(np.array(Image.open(imgs[idx])))
    
    if labels:
        label = labels[idx]
        plt.title(label)
    plt.show()
    
    rows = int(np.ceil(n_aug / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize = (cols * 5, rows * 5))

    for i in range(n_aug):
        img = np.array(Image.open(imgs[idx]))
        img = transform(image = img)['image']
        img = np.clip(img.numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
        axes.flat[i].imshow(img)
    plt.show()

display_aug(train_imgs, train_transform, labels = train_labels)

train_loader = DataLoader(train_dataset(train_imgs, train_labels, transform = train_transform),
                          shuffle = True, batch_size = BATCH_SIZE)
val_loader = DataLoader(train_dataset(val_imgs, val_labels, transform = val_transform),
                        shuffle = False, batch_size = BATCH_SIZE)

model = ResNet34().to(DEVICE)

criterion = nn.BCEWithLogitsLoss().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr = LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', patience = 2, factor = 0.5, min_lr = 5e-5)

total_batch = len(train_loader)

val_acc = -1

for epoch in range(N_EPOCH):
    avg_cost = 0
    pbar = tqdm(train_loader)

    model.train()
    for X, y in pbar:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, y)
        cost.backward()
        optimizer.step()
        
        avg_cost += cost / total_batch
        
    model.eval()
    with torch.no_grad():
        mean_acc = 0
        for X, y in val_loader:
            X = X.to(DEVICE)
            pred = np.argmax(model(X).cpu().numpy(), axis = 1)
            y = np.argmax(y.numpy(), axis = 1)
            acc = accuracy_score(y, pred)
            mean_acc += acc / len(val_loader)
    
    scheduler.step(mean_acc)
    
    if val_acc < mean_acc:
        print(f'val_acc improved {val_acc:.4f} to {mean_acc:.4f}')
        val_acc = mean_acc
        torch.save(model.state_dict(), 'ResNet34_best_val_acc.pt')
    
    print(f'[Epoch {epoch + 1} / {N_EPOCH}] cost = {avg_cost:.4f}, val_acc = {mean_acc:.4f}')


model.load_state_dict(torch.load('ResNet34_best_val_acc.pt'))

test_loader = DataLoader(test_dataset(test_imgs, transform = train_transform, n_tta = N_TTA),
                         shuffle=False, batch_size = BATCH_SIZE)

model.eval()
preds = []
with torch.no_grad():
    for imgs in tqdm(test_loader):
        pred = [torch.sigmoid(model(X.to(DEVICE))).cpu().numpy() for X in imgs]
        pred = np.mean(np.array(pred), axis = 0)
        preds.append(pred)

pred = np.argmax(np.concatenate(np.array(preds)), axis = 1)
pred.shape

path1 = '../_data/dacon/computer/'

submission['target'] = [list(label_map.keys())[i] for i in pred]
save_path = path1 + "sample_submission.csv"
submission.to_csv(save_path, index = False)
pd.read_csv(save_path)

display_aug(test_imgs, train_transform, list(submission['target']))