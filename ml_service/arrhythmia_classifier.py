import numpy as np
import wfdb
import torch
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
# from torchvision import transforms
# from torchvision.utils import make_grid
import os
import base64
import math
import random

from PIL import Image, ImageOps, ImageEnhance
import numbers

import wfdb
import numpy as np




is_cuda = False
num_epochs = 100
batch_size = 1
lr = 0.0003
torch.manual_seed(46)
log_interval = 10
in_channels_ = 1
num_segments_in_record = 100
segment_len = 3600
num_records = 48
num_classes = 16
allow_label_leakage = True

class CustomDatasetFromCSV(Dataset):
    def __init__(self, data_path, transforms_=None):
        self.df = pd.read_pickle(data_path)
        self.transforms = transforms_

    def __getitem__(self, index):
        row = self.df.iloc[index]
        signal = row['signal']
        target = row['target']
        if self.transforms is not None:
            signal = self.transforms(signal)
        signal = signal.reshape(1, signal.shape[0])
        return signal, target

    def __len__(self):
        return self.df.shape[0]

pdData = pd.read_pickle("data/Arrhythmia_dataset.pkl")  

normalData = pdData[pdData["target"] ==0]
trainData = pd.DataFrame(normalData.iloc[0:198])
validData = pd.DataFrame(normalData.iloc[198:242])
testData = pd.DataFrame(normalData.iloc[242:283])

aData=pdData[pdData["target"] ==1]
trainData=pd.concat([trainData, aData[0:46]])
validData=pd.concat([validData, aData[47:59]])
testData=pd.concat([testData, aData[60:66]])


# aflData=pdData[pdData["target"] ==2]
# trainData=pd.concat([trainData, aflData[0:14]])
# validData=pd.concat([validData, aflData[14:15]])
# testData=pd.concat([testData, aflData[15:20]])


afibData=pdData[pdData["target"] ==3]
trainData=pd.concat([trainData, afibData[0:95]])
validData=pd.concat([validData, afibData[95:118]])
testData=pd.concat([testData, afibData[118:135]])

# prexData=pdData[pdData["target"] ==4]
# trainData=pd.concat([trainData, prexData[0:15]])
# validData=pd.concat([validData, prexData[15:19]])
# testData=pd.concat([testData, prexData[19:21]])


# bData=pdData[pdData["target"] ==5]
# trainData=pd.concat([trainData, bData[0:39]])
# validData=pd.concat([validData, bData[39:46]])
# testData=pd.concat([testData, bData[46:56]])

# tData=pdData[pdData["target"] ==6]
# trainData=pd.concat([trainData, tData[0:10]])
# validData=pd.concat([validData, tData[10:12]])
# testData=pd.concat([testData, tData[12:14]])


# ivrData=pdData[pdData["target"] ==7]
# trainData=pd.concat([trainData, ivrData[0:7]])
# validData=pd.concat([validData, ivrData[7:9]])
# testData=pd.concat([testData, ivrData[9:12]])


# vflData=pdData[pdData["target"] ==8]
# trainData=pd.concat([trainData, vflData[0:6]])
# validData=pd.concat([validData, vflData[6:8]])
# testData=pd.concat([testData, vflData[8:10]])


lData=pdData[pdData["target"] ==9]
trainData=pd.concat([trainData, lData[0:73]])
validData=pd.concat([validData, lData[73:85]])
testData=pd.concat([testData, lData[85:103]])


rData=pdData[pdData["target"] ==10]
trainData=pd.concat([trainData, rData[0:43]])
validData=pd.concat([validData, rData[43:51]])
testData=pd.concat([testData, rData[51:62]])

# pData=pdData[pdData["target"] ==11]
# trainData=pd.concat([trainData, pData[0:31]])
# validData=pd.concat([validData, pData[31:37]])
# testData=pd.concat([testData, pData[37:45]])

trainData.to_pickle('data/train.pkl')
validData.to_pickle('data/validData.pkl')
testData.to_pickle('data/testData.pkl')



# print(trainData.shape)
# print(validData.shape)
# print(testData.shape)

        

train_dataset = CustomDatasetFromCSV('data/train.pkl')
train_size = len(train_dataset)
print(train_size)
valid_dataset = CustomDatasetFromCSV('data/validData.pkl')
test_dataset = CustomDatasetFromCSV('data/testData.pkl')



train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

def basic_layer(in_channels, out_channels, kernel_size, batch_norm=False, max_pool=True, conv_stride=1, padding=0
                , pool_stride=2, pool_size=2):
    layer = nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=conv_stride,
                  padding=padding),
        nn.ReLU())
    if batch_norm:
        layer = nn.Sequential(
            layer,
            nn.BatchNorm1d(num_features=out_channels))
    if max_pool:
        layer = nn.Sequential(
            layer,
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride))

    return layer


class arrhythmia_classifier(nn.Module):
    def __init__(self, in_channels=in_channels_):
        super(arrhythmia_classifier, self).__init__()
        self.cnn = nn.Sequential(
            basic_layer(in_channels=in_channels, out_channels=128, kernel_size=50, batch_norm=True, max_pool=True,
                        conv_stride=3, pool_stride=3),
            basic_layer(in_channels=128, out_channels=32, kernel_size=7, batch_norm=True, max_pool=True,
                        conv_stride=1, pool_stride=2),
            basic_layer(in_channels=32, out_channels=32, kernel_size=10, batch_norm=False, max_pool=False,
                        conv_stride=1),
            basic_layer(in_channels=32, out_channels=128, kernel_size=5, batch_norm=False, max_pool=True,
                        conv_stride=2, pool_stride=2),
            basic_layer(in_channels=128, out_channels=256, kernel_size=15, batch_norm=False, max_pool=True,
                        conv_stride=1, pool_stride=2),
            basic_layer(in_channels=256, out_channels=512, kernel_size=5, batch_norm=False, max_pool=False,
                        conv_stride=1),
            basic_layer(in_channels=512, out_channels=128, kernel_size=3, batch_norm=False, max_pool=False,
                        conv_stride=1),
            Flatten(),
            nn.Linear(in_features=1152, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=.1),
            nn.Linear(in_features=512, out_features=num_classes),
           # nn.Softmax()
        )

    def forward(self, x, ex_features=None):
        return self.cnn(x)


def calc_next_len_conv1d(current_len=112500, kernel_size=16, stride=8, padding=0, dilation=1):
    return int(np.floor((current_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))


device = torch.device("cuda:2" if is_cuda else "cpu")

model = arrhythmia_classifier().to(device).double()
num_of_iteration = len(train_dataset) // batch_size

def acc_batch(oupt, targets):
    # oupt is logits, targets is ordinals
    largests = torch.argmax(oupt, dim=1)  # largest idx each row

    n_correct = torch.sum(targets == largests)
    return n_correct / len(targets)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

criterion = nn.CrossEntropyLoss()
#criterion = nn.NLLLoss()



def train(epoch):
    model.train()
    train_loss = 0
    train_epoch_acc = 0
    accuracy = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        accuracy = acc_batch(output,target)    

    print('====> Epoch: {} Average train loss: {:.4f}   Accuracy on train data = {:.4f} '.format(
        epoch, train_loss / len(train_loader.dataset), accuracy))


def validation(epoch):
    model.eval()
    test_loss = 0
    
    accuracy = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(validation_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            accuracy = acc_batch(output,target)
        

            if batch_idx == 0:
                n = min(data.size(0), 4)
                

    test_loss /= len(validation_loader.dataset)
    print('Accuracy on test data = {:.4f}'.format(accuracy))
    print('====> Validation set loss: {:.5f}'.format(test_loss))
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')


def test(epoch):
    model.eval()
    test_loss = 0
    
    accuracy = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            accuracy = acc_batch(output,target)
        

            if batch_idx == 0:
                n = min(data.size(0), 4)
                

    test_loss /= len(test_loader.dataset)

    print('====> Test set loss: {:.5f}     Accuracy on test data = {:.4f} '.format(test_loss, accuracy))



model.load_state_dict(torch.load("ml_service/trained_model1"))

def arrythmia_name(class_name):
    switcher = {
        0: "Normal_beat",
        1: "Atrial_Premature Beat ",
        2: "Atrial_flutter ",
        3: "Atrial_fibrillation",
        4: "Pre_excitation_(WPW)",
        5: "Ventricular_bigeminy",
        6: "Ventricular_trigeminy",
        7: "Idioventricular_rhythm",
        8: "Ventricular_flutter",
        9: "Left_bundle_branch_block_beat",
        10: "Right_bundle_branch_block_beat",
        11: "Pacemaker_rhythm"
    }
    return switcher.get(class_name, "not_classified")


#------------------------------------------------------------------------------------------

def predict(data):
    output = model(data)
    o = torch.argmax(output, dim=1)
    name = arrythmia_name(o.data[0].item())
    print(data)
    return name
    

def predict_probabilities(data):
    output = model(data)
    softmax_output = F.softmax(output)
    print("DATAAAAAAAAA", data)
    p = torch.argsort(output, dim=1, descending = True)
    print(output[0])
    names = []
    probabilities = []
    first = p[0,:3].data[0].item()
    second = p[0, :3].data[1].item()
    third =  p[0, :3].data[2].item()
    names.append(arrythmia_name(first))
    names.append(arrythmia_name(second))
    names.append(arrythmia_name(third))

    probabilities.append(softmax_output[0][first].item() * 100)
    probabilities.append(softmax_output[0][second].item() * 100)
    probabilities.append(softmax_output[0][third].item() *100)


    # probabilities.append(F.softmax(output[0][first]).item() )
    # probabilities.append(F.softmax(output[0][second]).item())
    # probabilities.append(F.softmax(output[0][third]).item() )
    return names, probabilities
 

def visualize(data, name,  sampto):
    arr = data.cpu().detach().numpy()
    n_arr = np.array(arr).reshape(-1,1)
    wfdb.wrsamp(name, fs = 360, units=['mV'], sig_name= ['MLII'], p_signal=n_arr, fmt=['80'], write_dir = 'ml_service/visualizations/')
    rec = wfdb.rdrecord(f'ml_service/visualizations/{name}', sampto = sampto)
    fig = wfdb.plot_wfdb(record=rec,  plot_sym=True,  time_units='seconds',  figsize=(20,8), ecg_grids='all',return_fig = True )
    fig.savefig(f'ml_service/visualizations/{name}.png')
    with open(f"ml_service/visualizations/{name}.png", "rb") as imageFile:
        encodedString = base64.b64encode(imageFile.read())
    return encodedString;
    


def create_list_of_fragments(list_of_records):
    full_list = []
    for  record in range(len(list_of_records)):
        for item in list_of_records[record]["list_items"]:
            full_list.append(item)
    return full_list




