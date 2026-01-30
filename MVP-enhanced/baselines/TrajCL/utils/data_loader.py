import ast
import os
import time
import logging
import pickle
import pandas as pd
from torch.utils.data import Dataset

# 1) read raw pd, 2) split into 3 partitions
def read_traj_dataset(file_path,test_file_path):
    logging.info('[Load traj dataset] START.')
    _time = time.time()
    trajs = pd.read_pickle(file_path)
    test_trajs=pd.read_pickle(test_file_path)

    l = trajs.shape[0]
    train_idx = (int(l*0), 200000)
    eval_idx = (int(l*0.7), int(l*0.8))
    test_idx = (int(l*0.8), int(l*1.0))

    train_dataset = TrajDataset(trajs[train_idx[0]: train_idx[1]])
    # _eval,val_time_list = TrajDataset(trajs[eval_idx[0]: eval_idx[1]])
    test_dataset = TrajDataset(test_trajs)
    test_time_list = []
    _test=[]
    train_time_list = []
    _train=[]
    for i in range(len(train_dataset)):
        traj, ptime = train_dataset[i]
        train_time_list.append(ast.literal_eval(ptime)[-2]-ast.literal_eval(ptime)[0])
        _train.append(traj)
    for i in range(len(test_dataset)):
        traj, ptime = test_dataset[i]
        test_time_list.append(ast.literal_eval(ptime)[-2]-ast.literal_eval(ptime)[0])
        _test.append(traj)
    return _train,  _test,test_time_list

    # logging.info('[Load traj dataset] END. @={:.0f}, #={}({}/{}/{})' \
    #             .format(time.time() - _time, l, len(_train), len(_eval), len(_test)))
    # return _train, _eval, _test,test_time_list



class TrajDataset(Dataset):
    def __init__(self, data):
        # data: DataFrame
        self.data = data

    def __getitem__(self, index):
        return self.data.loc[index].merc_seq,self.data['ptime'].iloc[index]

    def __len__(self):
        return self.data.shape[0]


# Load traj dataset for trajsimi learning
def read_trajsimi_traj_dataset(file_path):
    logging.info('[Load trajsimi traj dataset] START.')
    _time = time.time()

    df_trajs = pd.read_pickle(file_path)
    offset_idx = int(df_trajs.shape[0] * 0.7) # use eval dataset
    df_trajs = df_trajs.iloc[offset_idx : offset_idx + 10000]
    assert df_trajs.shape[0] == 10000
    l = 10000

    train_idx = (int(l*0), int(l*0.7))
    eval_idx = (int(l*0.7), int(l*0.8))
    test_idx = (int(l*0.8), int(l*1.0))
    trains = df_trajs.iloc[train_idx[0] : train_idx[1]]
    evals = df_trajs.iloc[eval_idx[0] : eval_idx[1]]
    tests = df_trajs.iloc[test_idx[0] : test_idx[1]]

    logging.info("trajsimi traj dataset sizes. traj: #total={} (trains/evals/tests={}/{}/{})" \
                .format(l, trains.shape[0], evals.shape[0], tests.shape[0]))
    return trains, evals, tests


# Load simi dataset for trajsimi learning
def read_trajsimi_simi_dataset(file_path):
    logging.info('[Load trajsimi simi dataset] START.')
    _time = time.time()
    if not os.path.exists(file_path):
        logging.error('trajsimi simi dataset does not exist')
        exit(200)

    with open(file_path, 'rb') as fh:
        trains_simi, evals_simi, tests_simi, max_distance = pickle.load(fh)
        logging.info("[trajsimi simi dataset loaded] @={}, trains/evals/tests={}/{}/{}" \
                .format(time.time() - _time, len(trains_simi), len(evals_simi), len(tests_simi)))
        return trains_simi, evals_simi, tests_simi, max_distance
