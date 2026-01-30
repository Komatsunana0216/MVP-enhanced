import sys
sys.path.append("")

import numpy as np
import pandas as pd
import time
import pickle
from model import trajcl
import torch.nn.functional as F
from config import Config
import time_est
from torch.utils.data.dataloader import DataLoader

from utils.data_loader import read_traj_dataset
import math
import argparse
# from JTMR import JMTRModel



import torch
import os
torch.set_num_threads(5)
torch.multiprocessing.set_sharing_strategy('file_system')
# dev_id = 4
# os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
# torch.cuda.set_device(dev_id)

def evaluation(city,  start_time):

    route_min_len, route_max_len, gps_min_len, gps_max_len = 10, 100, 10, 256

    Config.post_value_updates()
    seq_model=trajcl.TrajCL().cuda()
    # checkpoint_file = '{}/{}_TrajCL_best{}.pt'.format(Config.checkpoint_dir, Config.dataset_prefix,
    #                                                   Config.dumpfile_uniqueid)
    # checkpoint = torch.load(checkpoint_file)
    seq_model.load_checkpoint()
    # seq_model = torch.load(checkpoint_file, map_location="cuda:{}".format(dev_id))['model']

    seq_model.eval()
    # print("seq_model",seq_model.vocab_size)
    print(seq_model)

    _,test_dataset, test_timelist = read_traj_dataset(Config.dataset_file,Config.test_dataset_file)

    print("test_datset长度",len(test_dataset))
    embs = pickle.load(open(Config.dataset_embs_file, 'rb')).to('cpu').detach()  # tensor
    cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))

    num_test_data = len(test_dataset)  # 使用实际的测试数据量

    batch_size=64
    with torch.no_grad():
        for i in range(math.ceil(num_test_data / batch_size)):  # 最后不足batch_size的case不要了
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if end_idx > num_test_data:
                end_idx = num_test_data
            batch_trajs1_emb, batch_trajs1_emb_p, batch_trajs1_len = trajcl.collate_for_test(trajs=test_dataset[start_idx:end_idx],
                                                                           cellspace=cellspace, embs=embs)
            batch_trajs1_emb = batch_trajs1_emb.cuda()
            batch_trajs1_emb_p = batch_trajs1_emb_p.cuda()
            batch_trajs1_len = batch_trajs1_len.cuda()
            batch_emb = seq_model.interpret(batch_trajs1_emb, batch_trajs1_emb_p, batch_trajs1_len)
            batch_travel_time=torch.FloatTensor(test_timelist[start_idx:end_idx]).cuda()


            if i==0:
                all_seq_emb=batch_emb
                batch_travel=batch_travel_time
            else:
                all_seq_emb=torch.concat([all_seq_emb,batch_emb],dim=0)
                batch_travel = torch.concat([batch_travel, batch_travel_time], dim=0)

    seq_embedding=all_seq_emb.cuda()
    batch_travel=batch_travel.cuda()

    # task 3
    time_est.evaluation(seq_embedding, batch_travel)
    a


    # load task 1 & task2 label
    feature_df = pd.read_csv("./data/{}/edge.csv".format(city))
    num_nodes = len(feature_df)
    print("num_nodes:", num_nodes)

    # load adj
    edge_index = np.load("./data/{}/line_graph_edge_idx.npy".format(city))
    print("edge_index shape:", edge_index.shape)


    # test_node_data = pd.read_pickle(
    #     open('../dataset/{}/{}_1101_1115_data_sample10w.pkl'.format(city, city), 'rb')) # data_seq_evaluation.pkl
    #
    # road_list = get_road(test_node_data)
    # print('number of road obervased in test data: {}'.format(len(road_list)))
    #
    # # prepare
    # num_samples = 'all'
    # if num_samples == 'all':
    #     pass
    # elif isinstance(num_samples, int):
    #     test_node_data = fair_sampling(test_node_data, num_samples)
    #
    # road_list = get_road(test_node_data)
    # print('number of road obervased after sampling: {}'.format(len(road_list)))



    print('start time : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("\n=== Evaluation ===")

    # prepare sequence task


    device=torch.device("cuda:{}".format(dev_id))
    preprocess_data = Preprocess().run()
    dataloader = tteloader()
    test_loader = dataloader.get_data_loader(preprocess_data['test_traj'])
    with torch.no_grad():
        all_seq_emb_list,travel_time_list=[],[]
        i=0
        for batch_data in test_loader:
            road_data, grid_data, travel_time = batch_data
            for k, v in road_data.items() or k=='road_lens':
                road_data[k] = v.to(device)
            for k, v in grid_data.items():
                grid_data[k] = v.to(device)
            road_data['g_input_feature'] = torch.FloatTensor(preprocess_data['road_feature']).to(device)
            road_data['g_edge_index'] = torch.LongTensor(preprocess_data['road_graph']).to(device)
            grid_data['grid_image'] = torch.FloatTensor(preprocess_data['grid_image']).to(device)
            travel_time = travel_time.to(device)
            _,_,emb = seq_model.test(grid_data, road_data)
            # print("emb的形状",emb.shape)
            if i==0:
                all_seq_emb=emb
                batch_travel=travel_time
            else:
                all_seq_emb=torch.concat([all_seq_emb,emb],dim=0)
                batch_travel = torch.concat([batch_travel, travel_time], dim=0)
            i+=1
            # all_seq_emb_list.append(emb)
            # travel_time_list.append(travel_time)

    # seq_embedding=torch.stack(all_seq_emb_list,dim=0).to(device)
    # travel_time=torch.stack(travel_time_list,dim=0).to(device)
    # print("all_seq_em形状",all_seq_emb.shape)
    seq_embedding=all_seq_emb.to(device)
    batch_travel=batch_travel.to(device)

    # task 3
    time_est.evaluation(seq_embedding, batch_travel)





    # task 4
    # detour_base = pickle.load(
    #     open('/data/mazp/dataset/JMTR/didi_{}/detour_base_max5.pkl'.format(city), 'rb'))
    #
    # sim_srh.evaluation2(seq_embedding, None, seq_model, test_seq_data, num_nodes, detour_base, feature_df,
    #                     detour_rate=0.15, fold=10)  # 当road_embedding为None的时候过模型处理，时间特征为空

    geometry_df = pd.read_csv("./data/{}/edge_geometry.csv".format(city))

    trans_mat = np.load('./data/{}/transition_prob_mat.npy'.format(city))
    trans_mat = torch.tensor(trans_mat)

    dataloader = greenloader()
    test_loader = dataloader.get_data_loader(preprocess_data['test_traj'])

    with torch.no_grad():

        # all_road_data=[]
        # all_road_traj=[]
        # all_road_type = []
        # all_road_weeks = []
        # all_road_minutes = []

        i=0
        for batch_data in test_loader:
            road_data, _= batch_data
            # all_road_traj.append(road_data['road_traj'])
            # all_road_type.append(road_data['road_type'])
            # all_road_weeks.append(road_data['road_weeks'])
            # all_road_minutes.append(road_data['road_minutes'])
            # print("看看tuple", road_data['mask_road_index'])
            # print("看看tuple",road_data['mask_road_index'][1].shape)

            # all_road_data.append(road_data)
            for k, v in road_data.items():
                if k == 'mask_road_index'or k=='road_lens':
                    continue
                road_data[k] = v.to(device)
            # for k, v in grid_data.items():
            #     grid_data[k] = v.to(device)
            road_data['g_input_feature'] = torch.FloatTensor(preprocess_data['road_feature']).to(device)
            road_data['g_edge_index'] = torch.LongTensor(preprocess_data['road_graph']).to(device)
            # grid_data['grid_image'] = torch.FloatTensor(preprocess_data['grid_image']).to(device)

            emb = seq_model.routeonly_test(road_data)
            # print("emb的形状",emb.shape)
            if i==0:
                all_seq_emb=emb

            else:
                all_seq_emb=torch.concat([all_seq_emb,emb],dim=0)

            i+=1


    # seq_embedding=torch.stack(all_seq_emb_list,dim=0).to(device)
    # travel_time=torch.stack(travel_time_list,dim=0).to(device)
    # print("all_seq_em形状",all_seq_emb.shape)
    seq_embedding=all_seq_emb.to(device)

    # def pad_tensors_to_max_length(tensor_list):
    #     """
    #     此函数将列表中的张量填充到相同的最大长度
    #     :param tensor_list: 包含张量的列表
    #     :return: 填充后的张量列表
    #     """
    #     max_len = max([tensor.shape[1] for tensor in tensor_list])
    #     padded_tensor_list = []
    #     for tensor in tensor_list:
    #         pad_size = max_len - tensor.shape[1]
    #         # 对于形状为 [128, x] 的张量，使用 (0, pad_size) 进行填充
    #         padded_tensor = F.pad(tensor, (0, pad_size))
    #         padded_tensor_list.append(padded_tensor)
    #     return padded_tensor_list
    #
    # all_road_traj = pad_tensors_to_max_length(all_road_traj)
    # all_road_traj=torch.concat(all_road_traj,dim=0).cuda()
    # all_road_type = pad_tensors_to_max_length(all_road_type)
    # all_road_type=torch.concat(all_road_type,dim=0).cuda()
    # all_road_weeks = pad_tensors_to_max_length(all_road_weeks)
    # all_road_weeks=torch.concat(all_road_weeks,dim=0).cuda()
    # all_road_minutes = pad_tensors_to_max_length(all_road_minutes)
    # all_road_minutes=torch.concat(all_road_minutes,dim=0).cuda()
    #
    # all_road_data = {
    #     'road_traj': all_road_traj,
    #     'road_type': all_road_type,
    #     'road_weeks': all_road_weeks,
    #     'road_minutes': all_road_minutes,
    # }



    road_graph=torch.LongTensor(preprocess_data['road_graph']).to(device)
    road_feature=torch.FloatTensor(preprocess_data['road_feature']).to(device)
    sim_srh.evaluation3(seq_embedding, None, seq_model, preprocess_data['test_traj'], num_nodes, trans_mat, feature_df, geometry_df,road_feature,road_graph,
                        detour_rate=0.3, fold=10)  # 当road_embedding为None的时候过模型处理，时间特征为空

    end_time = time.time()
    print("cost time : {:.2f} s".format(end_time - start_time))

if __name__ == '__main__':

    # city = 'chengdu'
    # city='didi-chengdu'
    # city = 'xian'
    city = 'porto'

    start_time = time.time()


    evaluation(city,  start_time)






