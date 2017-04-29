# coding: utf-8

"""titanic の生存予測を、ニューラルネットワークで行う

ニューラルネットワークのサイズは MultiLayerNet の引数で決まる。
"""

import numpy as np
from common.multi_layer_net import MultiLayerNet
import pandas as pd
import matplotlib.pyplot as plt
from common.optimizer import *
import data_wrangle

def change_one_hot_label(X):
    """書籍を参考。正解データを one_hot_label 形式にする"""
    T = np.zeros((X.size, 2))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T

# 前処理されたデータの取得
train_df, test_df = data_wrangle.wrangle_data ()

# データを学習物と答えに分ける
x_train = train_df.drop (['Survived'], axis=1).values
t_train = train_df['Survived'].values
t_train = change_one_hot_label (t_train)

# ニューラルネットワークの生成。
# 隠れ層は適当に 100, 100, 100 とした。
network = MultiLayerNet(input_size=6, hidden_size_list=[100, 100, 100], output_size=2, weight_decay_lambda=0.01)
# Optimizer の選択。AdaGrad() は common.optimizer 以下にある。
optimizer = AdaGrad()

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 99

train_loss_list = []
train_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

# 学習
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 勾配
    #grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    # 更新
    optimizer.update(network.params, grad)
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        train_acc_list.append(train_acc)

# train データでどのように学習されたか図示
plt.plot (train_acc_list)

# テストデータの PassengerId を退避させる
test_df_id = test_df['PassengerId']
test_df = test_df.drop ('PassengerId', axis=1)
x_test = test_df.values

# テストデータの生存予測
output = network.predict (x_test)
output = np.argmax (output, axis=1)

# 生存予測を Kaggle に投稿する形に修正する
output_df = pd.DataFrame (output, columns=['Survived'])
output_df['PassengerId'] = test_df_id
output_df = output_df.ix[:, ['PassengerId', 'Survived']]
output_df.to_csv('test_ans.csv', index=False, encoding='utf-8')
