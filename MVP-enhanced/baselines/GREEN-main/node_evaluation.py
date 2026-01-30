import sys

sys.path.append("..")
import numpy as np
import pandas as pd
import time
import pickle
from Exp import road_cls, speed_inf
from dataset.preprocess import Preprocess
from config.config import Config
from dataset.green_loader import TrajDataLoader as greenloader

import torch
import os
import road_emb_concat

torch.set_num_threads(5)

dev_id = 6
torch.cuda.set_device('cuda:6')


def evaluation(city,  start_time):
    exp_id = f'exp_bs128_road{Config.road_trm_layer}_grid{Config.grid_trm_layer}_inter{Config.inter_trm_layer}_epoch30_2e-4_cls_{Config.mask_length}_{Config.mask_ratio}ratio'
    pretrain_path = f'./log/{Config.dataset}/{exp_id}/pretrain/best_pretrain_model.pth'
    seq_model = torch.load(pretrain_path, map_location="cuda:{}".format(dev_id))['model']

    seq_model.eval()
    # print("seq_model",seq_model.vocab_size)
    print(seq_model)

    print('start time : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("\n=== Evaluation ===")

    device=torch.device("cuda:{}".format(dev_id))
    preprocess_data = Preprocess().run()
    dataloader = greenloader()
    test_loader = dataloader.get_data_loader(preprocess_data['train_traj'])

    def get_road(df):
        road_list = []
        df['road_traj'].apply(lambda row: road_list.extend(row))
        return list(set(road_list))
    road_list=get_road(preprocess_data['train_traj'])
    max_len = 64
    road_feature = torch.FloatTensor(preprocess_data['road_feature']).to(device)
    road_graph = torch.LongTensor(preprocess_data['road_graph']).to(device)
    grid_image=torch.FloatTensor(preprocess_data['grid_image']).cuda()
    emb_path = './data/{}/{}_1101_1115_road_embedding.pkl'.format(
        city, city)

    if os.path.exists(emb_path):
        # load road embedding from inference result
        road_embedding = torch.load(emb_path, map_location='cuda:{}'.format(dev_id))['road_embedding']
    else:
        # infer road embedding
        # todo 改了batchsize
        type_mask=False
        road_embedding = road_emb_concat.get_road_emb_from_concat(seq_model, test_loader, road_feature,road_graph,grid_image,road_list,max_len,type_mask,device, batch_size=16,
                                                 city=city)
        # todo 保存这里先注释掉了
        torch.save({'road_embedding': road_embedding}, emb_path)

    # task 1

    #
    feature_df = pd.read_csv("./data/{}/edge.csv".format(city))
#todo 有prompt的用这个
    # type_mask=True
    # cls_road_embedding = road_emb_concat.get_road_emb_from_concat(seq_model, test_loader, road_feature, road_graph,grid_image,
    #                                                           road_list, max_len, type_mask, device,batch_size=16,
    #                                                           city=city)
    # road_cls.evaluation(cls_road_embedding, feature_df,road_list)
    road_cls.evaluation(road_embedding, feature_df, road_list)



    # task 2


    speed_inf.evaluation(road_embedding, feature_df, road_list)

    end_time = time.time()
    print("cost time : {:.2f} s".format(end_time - start_time))


if __name__ == '__main__':
    # city = 'chengdu'

    # city = 'didi-chengdu'
    city = 'didi-xian'

    start_time = time.time()

    evaluation(city, start_time)


