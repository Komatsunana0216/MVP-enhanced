import json
import sys
sys.path.append("")

import numpy as np
import pandas as pd
import time
import time_est
import sim_srh
import pickle
import torch.nn.functional as F
from libcity.data.dataset import ETADataset,ContrastiveDataset
from libcity.model.trajectory_embedding import BERTDownstream
import argparse
# from JTMR import JMTRModel



import torch
import os
torch.set_num_threads(5)
torch.multiprocessing.set_sharing_strategy('file_system')
dev_id = 6
torch.cuda.set_device('cuda:6')

def evaluation(city,  start_time):

    route_min_len, route_max_len, gps_min_len, gps_max_len = 10, 100, 10, 256


    # pretrain_path='./libcity/cache/665403/model_cache/665403_BERTContrastiveLM_chengdu.pt'
    #
    #
    #
    # seq_model = torch.load(pretrain_path, map_location="cuda:{}".format(dev_id))['model']





    # seq_model.eval()
    # # print("seq_model",seq_model.vocab_size)
    # print(seq_model)


    # # load task 1 & task2 label
    # feature_df = pd.read_csv("./data/{}/edge.csv".format(city))
    # num_nodes = len(feature_df)
    # print("num_nodes:", num_nodes)
    #
    # # load adj
    # edge_index = np.load("./data/{}/line_graph_edge_idx.npy".format(city))
    # print("edge_index shape:", edge_index.shape)






    print('start time : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("\n=== Evaluation ===")

    # prepare sequence task
    config=json.load(open('{}.json'.format(city), 'r'))
    eta_dataset=ETADataset(config)
    _, _, test_dataloader=eta_dataset.get_data()
    data_feature=eta_dataset.get_data_feature()
    downstream_model=BERTDownstream(config,data_feature).cuda()
    downstream_model.eval()
    device = torch.device("cuda:{}".format(dev_id))
    #
    node_features = eta_dataset.get_data_feature()['node_features'].cuda()
    edge_index=eta_dataset.get_data_feature()['edge_index'].cuda()
    loc_trans_prob=eta_dataset.get_data_feature()['loc_trans_prob'].cuda()

    graph_dict = {
        'node_features': node_features,
        'edge_index': edge_index,
        'loc_trans_prob': loc_trans_prob,
    }

    with torch.no_grad():
        i=0
        for batch in test_dataloader:
            X, targets, padding_masks, batch_temporal_mat = batch
            X = X.to(device)
            targets = targets.to(device)
            padding_masks = padding_masks.to(device)  # 0s: ignore
            batch_temporal_mat = batch_temporal_mat.to(device)
            predictions,_ = downstream_model(x=X, padding_masks=padding_masks, batch_temporal_mat=batch_temporal_mat,
                                     graph_dict=graph_dict)
            if i==0:
                all_pred=predictions
                batch_travel=targets
            else:
                all_pred=torch.concat([all_pred,predictions],dim=0)
                batch_travel = torch.concat([batch_travel, targets], dim=0)
            i+=1
            if i==1563:
                break
    print("i是多少",i)
    seq_embedding=all_pred.to(device)
    batch_travel=batch_travel.to(device)

    # task 3
    time_est.evaluation(seq_embedding, batch_travel)

    contra_dataset = ContrastiveDataset(config)
    _, _, test_dataloader = contra_dataset.get_data()
    data_feature = contra_dataset.get_data_feature()
    downstream_model = BERTDownstream(config, data_feature).cuda()
    downstream_model.eval()


    node_features = contra_dataset.get_data_feature()['node_features'].cuda()
    edge_index = contra_dataset.get_data_feature()['edge_index'].cuda()
    loc_trans_prob = contra_dataset.get_data_feature()['loc_trans_prob'].cuda()

    graph_dict = {
        'node_features': node_features,
        'edge_index': edge_index,
        'loc_trans_prob': loc_trans_prob,
    }

    with torch.no_grad():
        i = 0
        for batch in test_dataloader:
            X, _, padding_masks, batch_temporal_mat = batch
            X = X.to(device)
            if i==0:
                Y = X.clone()


            padding_masks = padding_masks.to(device)  # 0s: ignore
            batch_temporal_mat = batch_temporal_mat.to(device)
            predictions,_ = downstream_model(x=X, padding_masks=padding_masks, batch_temporal_mat=batch_temporal_mat,
                                           graph_dict=graph_dict)


            # 遍历每个 batch 中的数据
            for j in range(X.shape[0]):

                temp = [contra_dataset.get_data_feature()['vocab'].index2loc[index] for index in X[j, :, 0]]
                X[j, :, 0] = torch.LongTensor([
                    -999 if item == '<pad>' else 2 if item == '<sos>' else item
                    for item in temp
                ]).cuda()


            # 将结果转换为张量（如果需要）
            batch_input = X
            if i == 0:
                all_pred = predictions
                input=batch_input






            else:
                all_pred = torch.concat([all_pred, predictions], dim=0)
                input = torch.concat([input, batch_input], dim=0)

            i += 1
            if i == 1563:
                break
    # print("i是多少", i)
    # print("这里看看",input.shape)

    seq_embedding = all_pred.to(device)
    feature_df = pd.read_csv("./raw_data/{}/edge_features.csv".format(city))
    trans_mat = np.load('./raw_data/{}/transition_prob_mat.npy'.format(city))
    trans_mat = torch.tensor(trans_mat)
    geometry_df = pd.read_csv("./raw_data/{}/edge_geometry.csv".format(city))
    sim_srh.evaluation3(seq_embedding,  downstream_model, input, 0, trans_mat, feature_df, geometry_df,contra_dataset.get_data_feature()['vocab'],graph_dict,device,Y,
                        detour_rate=0.3, fold=10)

    end_time=time.time()
    print("cost time : {:.2f} s".format(end_time - start_time))

if __name__ == '__main__':

    # city = 'chengdu'
    city = 'xian'


    start_time = time.time()


    evaluation(city,  start_time)


