#Filename:	mlp.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 08 Des 2020 10:08:12  WIB

import warnings
warnings.filterwarnings("ignore")

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score 
from sklearn.preprocessing import MinMaxScaler

from utils.heloc_dataset import *
from utils.Dataset import Dataset
from utils.helper import load_heloc, data_loader_torch

def create_model(input_len):

    model = nn.Sequential(
            nn.Linear(input_len, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
            )

    return model

def train_heloc(filename, epoches, save_name):
    
    heloc_df = load_heloc(filename)
    d = Dataset(dataframe = heloc_df, continuous_features = 'all', outcome_name = 'RiskPerformance', scaler = MinMaxScaler())
    X_train, X_test, y_train, y_test = d.train_scaled_x, d.test_scaled_x, d.train_y.astype(np.float32), d.test_y.astype(np.float32)
    train_loader, test_loader = data_loader_torch(X_train, X_test, y_train, y_test)
    model = create_model(X_train.shape[1])
    #optimizer = optim.SGD(model.parameters(), lr = 0.26, momentum = 0.2)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    best_f1 = -float('inf')
    best_model = None
    best_acc = -float('inf')

    for epoch in range(epoches):

        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc, val_f1 = evaluate(model, test_loader, criterion)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = copy.deepcopy(model)
            best_acc = valid_acc

        print(f'Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}')
        print(f'Epoch: {epoch} | Val. Loss: {valid_loss:.4f} |  Val. Acc: {valid_acc:.4f} |  Val. F1: {val_f1:.4f}')

    print("Best Val. F1: {:.4f}, Best Val. Accuarcy: {:.4f}".format(best_f1, best_acc))

    torch.save(best_model, save_name)

def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    model.train()
    prediction = []
    label = []
    
    for batch_idx, (data, target) in enumerate(iterator):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        #_, preds = torch.max(output, 1)
        preds = torch.round(output)
        loss = criterion(output, target)
        #print(batch_idx, loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(target)
        label.extend(target.tolist())
        prediction.extend(preds.reshape(-1).tolist())
        
    acc = accuracy_score(label, prediction)
    f1 = f1_score(label, prediction)

    return epoch_loss / len(iterator.dataset), acc, f1

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    model.eval()
    prediction = []
    label = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(iterator):
            
            data, target = data.cuda(), target.cuda()
            output = model(data)
            _, preds = torch.max(output, 1)
            preds = torch.round(output)
            loss = criterion(output, target)
            epoch_loss += loss.item() * len(target)
            label.extend(target.tolist())
            prediction.extend(preds.reshape(-1).tolist())
            
    acc = accuracy_score(label, prediction)
    f1 = f1_score(label, prediction, average = 'weighted')

    return epoch_loss / len(iterator.dataset), acc, f1

if __name__ == "__main__":
    epoch = 200
    saved_name = "weights/heloc.pth"
    filename = "data/heloc/heloc_dataset_v1.csv"
    train_heloc(filename, epoch, saved_name)

