import json
import sys

sys.path.append("..")
import numpy as np
import pandas as pd
import time
import pickle


from libcity.data.dataset import ContrastiveDataset
from libcity.model.trajectory_embedding import BERTDownstream
import road_cls,speed_inf

import torch
import os
import road_emb_concat

torch.set_num_threads(5)

dev_id = 6
torch.cuda.set_device('cuda:6')

def evaluation(city,  start_time):

    print('start time : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("\n=== Evaluation ===")
    device = torch.device("cuda:{}".format(dev_id))
    config = json.load(open('{}.json'.format(city), 'r'))
    contra_dataset = ContrastiveDataset(config)
    train_dataloader, _, _ = contra_dataset.get_data()
    data_feature = contra_dataset.get_data_feature()
    downstream_model = BERTDownstream(config, data_feature).cuda()
    downstream_model.eval()
    vocab=contra_dataset.get_data_feature()['vocab']
    node_features = contra_dataset.get_data_feature()['node_features'].cuda()
    edge_index = contra_dataset.get_data_feature()['edge_index'].cuda()
    loc_trans_prob = contra_dataset.get_data_feature()['loc_trans_prob'].cuda()

    graph_dict = {
        'node_features': node_features,
        'edge_index': edge_index,
        'loc_trans_prob': loc_trans_prob,
    }



    emb_path = './raw_data/{}/{}_1101_1115_road_embedding.pkl'.format(
        city, city)

    if os.path.exists(emb_path):
        # load road embedding from inference result
        road_embedding = torch.load(emb_path, map_location='cuda:{}'.format(dev_id))['road_embedding']
        road_list=pd.read_pickle('./raw_data/{}/{}_road_list.pkl'.format(
        city, city))
    else:
        # infer road embedding
        # todo 改了batchsize

        road_embedding,road_list = road_emb_concat.get_road_emb_from_concat(downstream_model, train_dataloader, graph_dict,vocab,device, batch_size=16,
                                                 city=city)

        file_path = './raw_data/{}/{}_road_list.pkl'.format(
        city, city)
        with open(file_path, 'wb') as file:
            pickle.dump(road_list, file)


    feature_df = pd.read_csv("./raw_data/{}_roadmap_edge/edge_features.csv".format(city))

    road_cls.evaluation(road_embedding, feature_df, road_list)
    speed_inf.evaluation(road_embedding, feature_df, road_list)

    end_time = time.time()
    print("cost time : {:.2f} s".format(end_time - start_time))




if __name__ == '__main__':
    # city = 'chengdu'
    city = 'xian'



    start_time = time.time()

    evaluation(city, start_time)


