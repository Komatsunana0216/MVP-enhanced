import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math


def label_norm(y):
    std = torch.std(y)
    mean = torch.mean(y)
    y = (y-mean)/std
    return y, mean, std

# 预测结果反标准化
def pred_unnorm(pred,mean,std):
    pred = pred*std + mean
    return pred

def next_batch_index(ds, bs, shuffle=True):
    num_batches = math.ceil(ds / bs)

    index = np.arange(ds)
    if shuffle:
        index = np.random.permutation(index)

    for i in range(num_batches):
        if i == num_batches - 1:
            batch_index = index[bs * i:]
        else:
            batch_index = index[bs * i: bs * (i + 1)]
        yield batch_index

class MLPReg(nn.Module):
    def __init__(self, input_size, num_layers, activation):
        super(MLPReg, self).__init__()

        self.num_layers = num_layers
        self.activation = activation

        self.layers = []
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(input_size, input_size))
        self.layers.append(nn.Linear(input_size, 1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x).squeeze(1)



def evaluation(seq_embedding, label_data,  fold=5):
    batch_size = 64
    y=label_data.squeeze()

    x = seq_embedding
    print("x形状",x.shape)
    print("y形状",y.shape)
    split = x.shape[0] // fold
    print(split, x.shape[0], fold)

    device_flag = True

    fold_preds = []
    fold_trues = []
    for i in range(fold):  #
        eval_idx = list(range(i * split, (i + 1) * split, 1))
        train_idx = list(set(list(range(x.shape[0]))) - set(eval_idx))

        x_train, x_eval = x[train_idx], x[eval_idx]
        y_train, y_eval = y[train_idx], y[eval_idx]

        fold_trues.append(y_eval)

        y_train, mean, std = label_norm(y_train) # 标准化
        model = MLPReg(x.shape[1], 3, nn.ReLU()).cuda()

        if device_flag:
            print('device: ', next(model.parameters()).device)
            device_flag = False

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        # one experiment
        patience = 3
        epoch_threshold = 10
        epoch_num = 50
        best_epoch = 0
        best_mae = 1e9
        best_rmse = 1e9
        best_preds = None
        for epoch in range(1, epoch_num + 1):
            model.train()
            for batch_index in next_batch_index(x_train.shape[0], batch_size):
                opt.zero_grad()
                x_batch = x_train[batch_index]
                y_batch = y_train[batch_index]
                # print("x_batch形状",x_batch.shape)
                # print("y_batch形状", y_batch.shape)

                loss = nn.MSELoss()(model(x_batch), y_batch.cuda())
                loss.backward()
                opt.step()

            model.eval()
            y_preds = model(x_eval).detach().cpu()

            mean=mean.cpu()
            std=std.cpu()
            y_preds = pred_unnorm(y_preds, mean, std)
            y_eval=y_eval.cpu()
            mae = mean_absolute_error(y_eval, y_preds)
            rmse = mean_squared_error(y_eval, y_preds) ** 0.5
            # print(f'Epoch: {epoch}, MAE: {mae.item():.4f}, RMSE: {rmse.item():.4f}')
            # print(epoch)
            if epoch == epoch_num:
                # print('situation 1')
                fold_preds.append(best_preds)

            if mae < best_mae and rmse < best_rmse: # 有性能改善 重置 patience
                best_preds = y_preds
                best_mae = mae
                best_rmse = rmse
                best_epoch = epoch
                patience = 3
            else:
                if epoch > epoch_threshold:
                    patience -= 1
                if patience < 0:
                    # print('situation 2')
                    fold_preds.append(best_preds)
                    # print(f'Best epoch: {best_epoch}, MAE: {best_mae.item():.4f}, RMSE: {best_rmse.item():.4f}')
                    break

    y_preds = torch.cat(fold_preds, dim=0)
    y_trues = torch.cat(fold_trues, dim=0)
    y_preds=y_preds.cpu()
    y_trues=y_trues.cpu()
    mae = mean_absolute_error(y_trues, y_preds)
    rmse = mean_squared_error(y_trues, y_preds) ** 0.5
    print(f'travel time estimation  | MAE: {mae:.4f}, RMSE: {rmse:.4f}')

    return best_epoch, best_mae, best_rmse

