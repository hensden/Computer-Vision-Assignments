import os
import sys
import cv2
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import vgg16
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# DEFINE DATASET CLASS
class KittiDataSet(Dataset):
    def __init__(self, img_fnames, lab_fnames):
        self.imgs = img_fnames
        self.labs = lab_fnames

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        X_file, y_file = self.imgs[idx], self.labs[idx]
        X = self.read_img(X_file)
        y = self.read_img(y_file)
        return X.transpose(2,0,1), y

    def read_img(self, img):
        image = Image.open(img)
        image = image.resize((1242,375), Image.ANTIALIAS)
        return np.asarray(image)

# ORDER FILENAMES
datapath, labelpath = './training/image_2/','./training/semantic/'
fnames = [datapath + i for i in sorted(os.listdir(datapath))]
training_imgs, validation_imgs, test_imgs =\
fnames[:140], fnames[140:170], fnames[170:]
fnames2 = [labelpath + i for i in sorted(os.listdir(labelpath))]
training_labs, validation_labs, test_labs =\
fnames2[:140], fnames2[140:170], fnames2[170:]

# DEFINE DATASET OBJECTS
training_set = KittiDataSet(training_imgs, training_labs)
validation_set = KittiDataSet(validation_imgs, validation_labs)
test_set = KittiDataSet(test_imgs, test_labs)

# FEED DATASET PARAMS TO DATALOADER
params = {'batch_size': 1,'shuffle': True,'num_workers': 1}
train_loader = DataLoader(training_set, **params)
val_loader   = DataLoader(validation_set, **params)
test_loader  = DataLoader(test_set, **params)


# FCN32:
class FCN32(nn.Module):
    def __init__(self):
        super(FCN32, self).__init__()
        vgg_model = vgg16(pretrained=True)
        self.features = vgg_model.features
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(7,7)),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=(1,1)),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(4096, 34, kernel_size=(1,1))
        )
        self.deconv9 = nn.ConvTranspose2d(34, 34, (247, 250), 32)
        
    def forward(self, batch):
        y = self.features(batch)
        y = self.conv6(y)
        y = self.conv7(y)
        y = self.conv8(y)
        y = self.deconv9(y)
        return y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FCN32()
model = model.to(device)


def get_ce_loss(y, gt):
    return F.cross_entropy(y, gt)

def get_pixel_accuracy(y, gt):
    ious = []
    y_hat = y.argmax(axis=1)
    for c in range(34):
        TP = float(((y_hat==c)&(gt==c)).type(torch.int).sum())
        FN = float(((y_hat!=c)&(gt==c)).type(torch.int).sum())
        FP = float(((y_hat==c)&(gt!=c)).type(torch.int).sum())
        ious.append(TP/(1+TP+FP+FN))
    return np.asarray(ious)



epochs = 140
lr = 0.001
rms = 0.99
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=rms)


val_history = []
train_history = []
best_val = +float('inf')
for epoch in range(epochs):
    trl = []
    tra = []
    trp = []
    for chunk in train_loader:
        # unpack data to device
        data, label = chunk
        data = data.type(torch.FloatTensor)
        label = label.type(torch.LongTensor)
        data = data.to(device)
        label = label.to(device)
        # feed forward
        output = model(data)
        # loss backprop
        loss = get_ce_loss(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # accuracy functions
        ious = get_pixel_accuracy(output, label)
        miou = ious.mean()
        trl.append(float(loss))
        tra.append(miou)
        trp.append(ious)
    tr_loss = np.asarray(trl).mean()
    tr_miou = np.asarray(tra).mean()
    tr_ious = np.asarray(trp).mean(axis=0)
    tr_dict = {'loss':tr_loss, 'miou':tr_miou, 'ious':tr_ious}
    train_history.append(tr_dict)
    
    trl = []
    tra = []
    trp = []
    for chunk in val_loader:
        data, label = chunk
        data = data.type(torch.FloatTensor)
        label = label.type(torch.LongTensor)
        data = data.to(device)
        label = label.to(device)
        # feed forward
        output = model(data)
        ious = get_pixel_accuracy(output, label)
        miou = ious.mean()
        trl.append(float(loss))
        tra.append(miou)
        trp.append(ious)
    tr_loss = np.asarray(trl).mean()
    tr_miou = np.asarray(tra).mean()
    tr_ious = np.asarray(trp).mean(axis=0)
    val_dict = {'loss':tr_loss, 'miou':tr_miou, 'ious':tr_ious}
    val_history.append(val_dict)
    
    if tr_loss < best_val:
        best_val = tr_loss
        torch.save(model.state_dict(), 'best.pth')
    
    print("Epoch: {},\nTraining: Loss={}, mIOU= {},\
          \nValidation: Loss={}, mIOU={}".format(
        epoch,tr_dict['loss'], tr_dict['miou'],
        val_dict['loss'], val_dict['miou']))
    
f1 = open('training_log.pkl', 'wb')
pickle.dump(train_history, f1)
f1.close()

f2 = open('validation_log.pkl', 'wb')
pickle.dump(val_history, f2)



model = FCN32()
model.load_state_dict(torch.load('best.pth'))
model.to(device)
model.eval()

def visualize(output, label):
  op = output.argmax(axis=1).cpu()[0].numpy().astype(np.uint8)*7
  gt = label.cpu()[0].numpy().astype(np.uint8)*7
  fig = plt.figure(figsize=(15,7))
  plt.subplot(121)
  plt.imshow(gt)

  plt.subplot(122)
  plt.imshow(op)

  plt.show()


trl, tra, trp = [], [], []
test_history = []
for chunk in test_loader:
    data, label = chunk
    data = data.type(torch.FloatTensor)
    label = label.type(torch.LongTensor)
    data = data.to(device)
    label = label.to(device)
    # feed forward
    output = model(data)
    visualize(output, label)
    ious = get_pixel_accuracy(output, label)
    loss = get_ce_loss(output, label)
    miou = ious.mean()
    trl.append(float(loss))
    tra.append(miou)
    trp.append(ious)

tr_loss = np.asarray(trl).mean()
tr_miou = np.asarray(tra).mean()
tr_ious = np.asarray(trp).mean(axis=0)
test_dict = {'loss':tr_loss, 'miou':tr_miou, 'ious':tr_ious}
test_history.append(test_dict)
    
    
print("Test: Loss={}, mIOU={}".format(test_dict['loss'], test_dict['miou']))