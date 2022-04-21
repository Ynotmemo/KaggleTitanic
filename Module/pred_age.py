import numpy as np
import os
import pandas as pd
import random
import sys
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from tqdm import tqdm

class AgeNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.125),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, X):
        y = self.model(X)
        return y

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def standardlization(ser):
    return (ser - ser.mean()) / ser.std()

def pred_age(all_data, batch_size, leraning_rate, train_size_rate):
    seed=0
    epochs = 500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:{0}'.format(device))

    all_data4age = pd.get_dummies(all_data, columns=['Sex', 'Embarked'], drop_first=True)

    #split train data and test data
    train_df4age = all_data4age[~(all_data4age['Age'].isna())].copy()
    test_df4age = all_data4age[all_data4age['Age'].isna()].copy()

    #standardlize "Age"
    age_mean = train_df4age['Age'].mean()
    age_std = train_df4age['Age'].std()
    train_df4age['Age'] = standardlization(train_df4age['Age'])

    dataset = TensorDataset(torch.FloatTensor(train_df4age.drop(columns=['Survived', 'Age', 'PassengerId']).values), torch.FloatTensor(train_df4age['Age'].values).view(-1,1))

    #訓練データと検証データ
    train_size = int(train_size_rate * len(dataset))
    valid_size = len(dataset) - train_size
    torch.manual_seed(seed)
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    #訓練データセットの作成
    g = torch.Generator()
    g.manual_seed(seed)
    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True,
        num_workers=2, drop_last=True,
        worker_init_fn=worker_init_fn, generator=g)

    #検証データセットの作成
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=len(valid_dataset), shuffle=False,
        num_workers=2)

    #ネットワークの定義
    input_dim = 8
    agenet = AgeNet(input_dim).to(device)
    #損失関数の定義
    criterion = nn.MSELoss()
    #最適化手法の定義
    optimizer = optim.Adam(agenet.parameters(), lr=leraning_rate)

    best_score = 1e10
    avg_train_loss = 0
    avg_valid_loss = 0

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for epoch in tqdm(range(epochs)):
        train_loss = 0
        valid_loss = 0

        agenet.train()

        for batch_X, batch_true in train_dataloader:
            batch_X = batch_X.to(device)
            batch_true = batch_true.to(device)

            optimizer.zero_grad()

            out = agenet.forward(batch_X)

            batch_loss = criterion(out, batch_true)
            train_loss += batch_loss


            batch_loss.backward()

            optimizer.step()

        avg_train_loss = train_loss / len(train_dataloader)
        agenet.eval()
        with torch.no_grad():
            for batch_X, batch_true in valid_dataloader:
                batch_X = batch_X.to(device)
                batch_true = batch_true.to(device)
                out = agenet(batch_X)
                batch_loss = criterion(out, batch_true)
                valid_loss += batch_loss

        avg_valid_loss = valid_loss / len(valid_dataloader)

        if best_score > avg_valid_loss:
            best_score = avg_valid_loss
            best_model = agenet.state_dict()
            best_epoch = epoch


    os.makedirs('./Model_Params/', exist_ok=True)
    torch.save(best_model, './Model_Params/netparams4age.pth')

    agenet.load_state_dict(best_model)
    agenet.to('cpu')

    agenet.eval()
    pred_age = agenet.forward(torch.Tensor(test_df4age.drop(columns=['PassengerId', 'Age', 'Survived']).values))
    pred_age_df = pd.DataFrame({'Age':pred_age.view(-1).detach().numpy()}, index = test_df4age['PassengerId'].values)


    age_missing = test_df4age.drop(columns='Age')
    age_complete = train_df4age

    age_missing2complete = pd.merge(age_missing, pred_age_df, left_on='PassengerId', right_index=True)

    all_age_df = pd.concat([age_complete, age_missing2complete])

    all_age_df["Age"] = all_age_df["Age"].map(lambda x:x*age_std + age_mean)

    return all_age_df
