import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import time
import pickle
from utils import Logger
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import faiss
import argparse
# from JTMR import JMTRModel

from task import road_cls, speed_inf, time_est, sim_srh
from evluation_utils import get_road, fair_sampling, get_seq_emb_from_traj_withRouteOnly, get_seq_emb_from_traj_withALLModel, prepare_data
import torch
import torch.nn as nn
import os
from traj_sim_search import traj_sim
torch.set_num_threads(5)
os.environ['TORCH_USE_CUDA_DSA']='1'





dev_id = 0
# os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
if torch.cuda.is_available():
    torch.cuda.set_device('cuda:0')
# device = torch.device('cuda:3')





# def query_prepare(expid, source_data,padding_id,num_queries,odd=True):
#     np.random.seed(expid)
#     route_data=source_data[0]
#     route_assign_mat=source_data[1]
#     gps_data=source_data[2]
#     gps_assign_mat=source_data[3]
#     original_gps_length=source_data[4]
#     num_samples = len(route_data)
#
#
#     def downsample_gps(gps_assign_mat, gps_data,route_assign_mat,route_data,num_queries,odd):
#         """
#         此函数对 gps_assign_mat 中的元素进行下采样，并将结果应用到 gps_data 上
#         :param gps_assign_mat: 输入的 gps_assign_mat 张量，形状为 [50000, 255]
#         :param gps_data: 输入的 gps_data 张量，形状为 [50000, 255, 8]
#         :return: 处理后的 gps_assign_mat 和 gps_data 张量
#         """
#         num_rows=num_queries
#         num_cols = gps_assign_mat.shape[1]
#         route_num_cols=route_assign_mat.shape[1]
#         new_gps_assign_mat_list = []
#         new_gps_data_list = []
#         new_route_assign_mat_list = []
#         new_route_data_list = []
#         new_route_list=[]
#         gps_length_list=[]
#         for i in range(num_rows):
#
#             # 找到起点和终点
#             start_index = 0
#             end_index = num_cols - 1
#             while gps_assign_mat[i, end_index] == padding_id:
#                 end_index -= 1
#             #todo 这里动了
#             # valid_indices = [start_index]
#             # for j in range(start_index + 1, end_index):
#             #     if torch.rand(1).item() >= 0.5:  # 以 0.5 的概率保留元素
#             #         valid_indices.append(j)
#             # valid_indices.append(end_index)
#             valid_indices = [start_index]
#             for j in range(start_index+1, end_index):
#                 if odd:
#                     if j%2==1:
#                         valid_indices.append(j)
#                 else:
#                     if j%2==0:
#                         valid_indices.append(j)
#             valid_indices.append(end_index)
#             to_be_resampled=gps_assign_mat[i, valid_indices]
#             new_route=[]
#             prev=None
#             for k in range(len(to_be_resampled)):
#                 if k==len(to_be_resampled)-1:
#                     if new_route[-1]==to_be_resampled[k]:
#                         break
#                     else:
#                         new_route.append(to_be_resampled[k])
#                         break
#                 if to_be_resampled[k]!=prev:
#                     if prev==None:
#                         new_route.append(to_be_resampled[k])
#                     else:
#                         if torch.rand(1).item() >= 0:
#
#                             if to_be_resampled[k]!=new_route[-1]:
#                                 new_route.append(to_be_resampled[k])
#                     prev = to_be_resampled[k]
#
#             # set_new_route=set([road.item() for road in new_route])
#             # if len(new_route)!= len(set_new_route):
#             #     print(new_route)
#             #     print(route_assign_mat[i])
#             #     a
#             new_route_copy=new_route
#             real_indices=[]
#             while len(new_route_copy)!=0:
#                 item=new_route_copy[0]
#                 new_route_copy=new_route_copy[1:]
#                 for p in range(len(to_be_resampled)):
#                     if to_be_resampled[p]==item:
#                         if len(real_indices) != 0 and valid_indices[p] <= real_indices[-1]:
#                             continue
#                         real_indices.append(valid_indices[p])
#                         if p+1<len(to_be_resampled) and to_be_resampled[p+1]!=item:
#                             break
#             final_sampled=gps_assign_mat[i, real_indices]
#
#             gps_length = []
#             prev_value = None
#             count = 0
#             for value in final_sampled:
#                 if prev_value is None:
#                     prev_value = value
#                     count = 1
#                 elif value == prev_value:
#                     count += 1
#                 else:
#                     gps_length.append(count)
#                     prev_value = value
#                     count = 1
#             gps_length.append(count)  # 最后一组相同点的长度
#             gps_length_list.append(torch.tensor(gps_length, dtype=torch.int))
#
#             new_gps_assign_mat_list.append(torch.tensor(gps_assign_mat[i, real_indices], dtype=torch.float32))
#             new_gps_data_list.append(gps_data[i, real_indices])
#             new_route_list.append(new_route)
#
#             # start_index = 0
#             # end_index = route_num_cols - 1
#             # while route_assign_mat[i, end_index] == padding_id:
#             #     end_index -= 1
#             # valid_indices = [start_index]
#             route_index=[]
#             for element in new_route:
#
#                 # for m in range(start_index + 1, end_index):
#                 for m in range(route_num_cols):
#                     if route_assign_mat[i, m]==element:
#                         if len(route_index)!=0 and m<=route_index[-1]:
#                             continue
#                         route_index.append(m)
#                         break
#
#             if len(route_index)!= len(set(route_index)):
#                 print(route_index)
#                 print(new_route)
#                 print(route_assign_mat[i])
#                 print("下面这俩")
#                 print(len(new_route))
#                 print(len(set(new_route)))
#                 b
#             new_route_assign_mat_list.append(torch.tensor(route_assign_mat[i, route_index], dtype=torch.float32))
#             new_route_data_list.append(route_data[i, route_index])
#
#
#         return new_gps_assign_mat_list, new_gps_data_list, new_route_list, new_route_assign_mat_list, new_route_data_list, gps_length_list
#
#
#     # 调用函数进行下采样
#
#     new_gps_assign_mat, new_gps_data, new_route_list, new_route_assign_mat, new_route_data, gps_length = downsample_gps(gps_assign_mat, gps_data,route_assign_mat, route_data,num_queries,odd)
#
#     def pad_sequences_to_length(sequence_list, target_length, padding_value=0):
#         """
#         此函数将序列列表中的张量填充到指定长度
#         :param sequence_list: 输入的张量列表，列表中的每个元素是一个张量
#         :param target_length: 要填充到的目标长度
#         :param padding_value: 填充值，默认为 0
#         :return: 填充后的张量
#         """
#         padded_sequence = pad_sequence(sequence_list, batch_first=True, padding_value=padding_value)
#         if padded_sequence.size(1) < target_length:
#             padding_size = target_length - padded_sequence.size(1)
#             padding = torch.full((padded_sequence.size(0), padding_size, *padded_sequence.size()[2:]), padding_value)
#             padded_sequence = torch.cat([padded_sequence, padding], dim=1)
#         elif padded_sequence.size(1) > target_length:
#             padded_sequence = padded_sequence[:, :target_length, ...]
#
#         return padded_sequence
#
#
#     route_target_length = route_assign_mat.shape[1]
#     gps_target_length = gps_assign_mat.shape[1]
#
#     new_route_assign_mat = pad_sequences_to_length(new_route_assign_mat, route_target_length, padding_value=padding_id)
#     new_route_data=pad_sequences_to_length(new_route_data, route_target_length, padding_value=-100)
#     new_route_data[:, :, :-1] = torch.where(new_route_data[:, :, :-1] == -100,
#                                            torch.full_like(new_route_data[:, :, :-1], 0), new_route_data[:, :, :-1])
#     new_route_data[:, :, -1] = torch.where(new_route_data[:, :, -1] == -100, torch.full_like(new_route_data[:, :, -1], -1), new_route_data[:, :, -1])
#     new_gps_assign_mat = pad_sequences_to_length(new_gps_assign_mat, gps_target_length, padding_value=padding_id)
#     new_gps_data= pad_sequences_to_length(new_gps_data, gps_target_length, padding_value=0)
#     new_gps_data[:, :, 0] = torch.ones_like(new_gps_data[:, :, 0])
#
#     gps_length = rnn_utils.pad_sequence(gps_length, padding_value=0, batch_first=True)
#
#
#     # y = random_index[:num_queries]
#
#     # if not torch.equal(new_route_data, route_data[random_index[:num_queries]]):
#     #     print(1)
#     #     print(new_route_data[:3])
#     #     print(route_data[:3])
#     #     c1
#     # if not torch.equal(new_gps_assign_mat, gps_assign_mat[random_index[:num_queries]]):
#     #     print(2)
#     #     # for i in range(new_gps_assign_mat.shape[0]):
#     #     #     if not torch.equal(new_gps_assign_mat[i], masked_gps_assign_mat[i]):
#     #     #         print(new_gps_assign_mat[i])
#     #     #         print(masked_gps_assign_mat[i])
#     #     #         c_2
#     #     c2
#     # if not torch.equal(new_gps_data, gps_data[random_index[:num_queries]]):
#     #     print(3)
#     #     # print(new_gps_data[:3])
#     #     # print(gps_data[:3])
#     #
#     #     # if not torch.equal(new_route_assign_mat, route_assign_mat[:5000]) or
#     #     # print(new_route_assign_mat[:3])
#     #     # print(route_assign_mat[:3])
#     #     print("输入并不相同")
#     #     c
#     # if not torch.equal(gps_length, original_gps_length[random_index[:num_queries]]):
#     #     print(4)
#     #     temp=original_gps_length[random_index[:num_queries]]
#     #     print(gps_length[:3])
#     #     print(temp[:3])
#     #     d
#     return new_route_assign_mat,new_route_data,new_gps_assign_mat,new_gps_data,gps_length
#



































def load_model(model_path, dev_id, city='chengdu'):
    """兼容多种模型保存格式的加载函数，支持prompt finetune模型"""
    import json
    import prompt_class
    checkpoint = torch.load(model_path, map_location="cuda:{}".format(dev_id), weights_only=False)
    
    if 'model' in checkpoint:
        # 旧格式：直接保存了完整模型对象
        model = checkpoint['model']
    elif 'model_state_dict' in checkpoint and 'model_params' in checkpoint:
        # 新格式：保存了model_params和model_state_dict
        from JGRM_bert_prompt_GPSmemory_ablation_route import JGRMModel
        model = JGRMModel(*checkpoint['model_params']).to("cuda:{}".format(dev_id))
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"[OK] 模型加载成功! epoch={checkpoint.get('epoch')}, loss={checkpoint.get('loss')}")
    elif 'model_state_dict' in checkpoint:
        # prompt finetune模型：只有state_dict，需要从配置文件读取参数并添加prompt组件
        from JGRM_bert_prompt_GPSmemory_ablation_route import JGRMModel
        
        # 从配置文件读取参数
        config_path = f"../config/chengdu_csv.json" if city == 'chengdu' else f"../config/{city}.json"
        if not os.path.exists(config_path):
            config_path = f"../config/{city}.json"
        config = json.load(open(config_path, 'r'))
        
        edge_index = np.load(f"../dataset/{city}/line_graph_edge_idx.npy")
        # 修复路径：配置文件路径是相对于项目根目录的，需要转换为相对于evaluation目录
        edge_features_path = config.get('edge_features_path', f"./dataset/{city}/new_edg_features.csv")
        if edge_features_path.startswith('./'):
            edge_features_path = '../' + edge_features_path[2:]
        edge_features = pd.read_csv(edge_features_path)
        
        vocab_size = config['vocab_size']
        route_embed_size = config['route_embed_size']
        length_mean = config.get('length_mean', 184.028281)
        length_std = config.get('length_std', 180.332222)
        
        # 构建基础模型
        model = JGRMModel(
            prompt_ST=1,  # prompt finetune模型需要设置为1
            vocab_size=vocab_size,
            route_max_len=config['route_max_len'],
            road_feat_num=config['road_feat_num'],
            road_embed_size=config['road_embed_size'],
            gps_feat_num=config['gps_feat_num'],
            gps_embed_size=config['gps_embed_size'],
            route_embed_size=route_embed_size,
            hidden_size=config['hidden_size'],
            edge_index=edge_index,
            drop_edge_rate=config['drop_edge_rate'],
            drop_route_rate=config['drop_route_rate'],
            drop_road_rate=config['drop_road_rate'],
            bert_hiden_size=config.get('bert_hidden_size', route_embed_size),
            pad_token_id=vocab_size,
            bert_attention_heads=config.get('bert_attention_heads', 8),
            bert_hidden_layers=config.get('bert_hidden_layers', 4),
            bert_vocab_size=vocab_size + 1,
            mode='x'
        ).to("cuda:{}".format(dev_id))
        
        # 添加prompt组件（与bert_prompt_finetune_route_ablation_shige.py一致）
        model.route_prompt = prompt_class.prompt(vocab_size, edge_features, length_mean, length_std, route_embed_size)
        model.route_prompt.cuda()
        model.route_fuse = nn.Linear(route_embed_size, route_embed_size)
        model.route_fuse.cuda()
        
        model.bert_prompt = prompt_class.prompt(vocab_size, edge_features, length_mean, length_std, route_embed_size)
        model.bert_prompt.cuda()
        model.bert_fuse = nn.Linear(route_embed_size, route_embed_size)
        model.bert_fuse.cuda()
        
        # 加载权重
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"[OK] Prompt Finetune模型加载成功! epoch={checkpoint.get('epoch')}, loss={checkpoint.get('loss')}")
        if missing:
            print(f"[WARN] Missing keys: {len(missing)} 个")
        if unexpected:
            print(f"[WARN] Unexpected keys: {len(unexpected)} 个")
    else:
        raise KeyError(f"无法识别的模型格式，checkpoint keys: {checkpoint.keys()}")
    
    return model

def evaluation(city, exp_path, model_name, start_time):

    route_min_len, route_max_len, gps_min_len, gps_max_len = 10, 100, 10, 256
    model_path = os.path.join(exp_path, 'model', model_name)
    embedding_name = model_name.split('.')[0]
    print(embedding_name)
    # seq_model=JGRMModel()
    seq_model = load_model(model_path, dev_id, city)
    seq_model.eval()
    print("seq_model",seq_model.vocab_size)
    print(seq_model)



    # load task 1 & task2 label
    feature_df = pd.read_csv("../dataset/{}/edge_features.csv".format(city))
    num_nodes = len(feature_df)
    print("num_nodes:", num_nodes)

    # load adj
    edge_index = np.load("../dataset/{}/line_graph_edge_idx.npy".format(city))
    print("edge_index shape:", edge_index.shape)


    test_node_data = pd.read_pickle(
        open('../dataset/{}/{}_1101_1115_data_sample10w.pkl'.format(city, city), 'rb')) # data_seq_evaluation.pkl

    road_list = get_road(test_node_data)
    print('number of road obervased in test data: {}'.format(len(road_list)))

    # prepare
    num_samples = 'all'
    if num_samples == 'all':
        pass
    elif isinstance(num_samples, int):
        test_node_data = fair_sampling(test_node_data, num_samples)

    road_list = get_road(test_node_data)
    print('number of road obervased after sampling: {}'.format(len(road_list)))

    # seq_model = torch.load(model_path, map_location="cuda:{}".format(dev_id))['model'] # model.to()包含inplace操作，不需要对象承接
    # seq_model.eval()
    # print("seq_model",seq_model.vocab_size)

    print('start time : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("\n=== Evaluation ===")

    # prepare sequence task
    test_seq_data = pd.read_pickle(
        open('../dataset/{}/{}_1101_1115_data_seq_evaluation.pkl'.format(city, city),
             'rb'))
    #todo 这动过，改成5000了
    test_seq_data = test_seq_data.sample(50000, random_state=0)

    route_length = test_seq_data['route_length'].values
    #todo TTE任务
    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = prepare_data(
        test_seq_data, route_min_len, route_max_len, gps_min_len, gps_max_len)
    route_data[:, :, 1] = torch.zeros_like(route_data[:, :, 1]).long()
    route_data[:, :, 2] = torch.full_like(route_data[:, :, 2], fill_value=-1)
    #todo 这里改了
    gps_data[:, :, -2] = torch.full_like(gps_data[:, :, -2], fill_value=-1)
    # gps_data[:, :, -5] = torch.full_like(gps_data[:, :, -5], fill_value=-1)
    # gps_data[:, :, -1] = torch.full_like(gps_data[:, :, -1], fill_value=0)
    gps_data[:, :, 0] = torch.ones_like(gps_data[:, :, 0])
    test_data = (route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset)
    # seq_embedding = get_seq_emb_from_traj_withRouteOnly(seq_model, test_data, batch_size=1024)
    #todo 这里动了
    # all_seq_embedding=get_seq_emb_from_traj_withALLModel(seq_model, test_data, batch_size=64)
    all_seq_embedding = get_seq_emb_from_traj_withALLModel(seq_model, test_data, batch_size=16)
    # task 3
    time_est.evaluation(all_seq_embedding, test_seq_data, num_nodes)
    # time_est.evaluation(seq_embedding, test_seq_data, num_nodes)

    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = prepare_data(
        test_seq_data, route_min_len, route_max_len, gps_min_len, gps_max_len)
    # print("gps_length",gps_length)
    test_data = (
        route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset)
    seq_embedding = get_seq_emb_from_traj_withRouteOnly(seq_model, test_data, batch_size=1024)
    # seq_embedding=None

    geometry_df = pd.read_csv("../dataset/{}/edge_geometry.csv".format(city))

    trans_mat = np.load('../dataset/{}/transition_prob_mat.npy'.format(city))
    trans_mat = torch.tensor(trans_mat)

    sim_srh.evaluation3(seq_embedding, None, seq_model, test_seq_data, num_nodes, trans_mat, feature_df, geometry_df,
                        detour_rate=0.3, fold=10)
    # traj_sim(seq_embedding, seq_model, source_data, num_nodes, fold=10)  # 当road_embedding为None的时候过模型处理，时间特征为空

    end_time = time.time()
    print("cost time : {:.2f} s".format(end_time - start_time))


    #
    #
    #
    # num_queries = 5000
    # # random_index = np.random.permutation(len(route_data))
    # # random_index=[i for i in range(len(route_data))]
    # perm = torch.randperm(len(route_data))
    # test_data=(route_data[perm], masked_route_assign_mat[perm], gps_data[perm], masked_gps_assign_mat[perm], route_assign_mat[perm], gps_length[perm], dataset)
    # source_data = (route_data[perm], route_assign_mat[perm], gps_data[perm], masked_gps_assign_mat[perm], gps_length[perm])
    #
    # y=[i for i in range(num_queries)]
    # new_route_assign_mat, new_route_data, new_gps_assign_mat, new_gps_data, gps_length = query_prepare(0,
    #                                                                                                       source_data,
    #                                                                                                       num_nodes,
    #                                                                                                       num_queries,odd=True)
    #
    # queries = (
    # new_route_data, new_route_assign_mat, new_gps_data, new_gps_assign_mat, new_route_assign_mat, gps_length, None)
    #
    # new_route_assign_mat2, new_route_data2, new_gps_assign_mat2, new_gps_data2, gps_length2 = query_prepare(0,
    #                                                                                                       source_data,
    #                                                                                                       num_nodes,
    #                                                                                                       num_queries,odd=False)
    #
    # queries2 = (
    # new_route_data2, new_route_assign_mat2, new_gps_data2, new_gps_assign_mat2, new_route_assign_mat2, gps_length2, None)
    #
    # print(new_route_assign_mat[0])
    # print(new_route_assign_mat2[0])
    # print(new_gps_assign_mat[0])
    # print(new_gps_assign_mat2[0])
    # torch.manual_seed(42)
    # # seq_embedding = get_seq_emb_from_traj_withALLModel(seq_model, test_data, batch_size=16)
    # q2 = get_seq_emb_from_traj_withALLModel(seq_model, queries2, batch_size=16)
    # torch.manual_seed(42)
    # q = get_seq_emb_from_traj_withALLModel(seq_model, queries, batch_size=16)
    #
    # # torch.manual_seed(42)
    # # seq_embedding2 = get_seq_emb_from_traj_withALLModel(seq_model, test_data, batch_size=16)
    # # print("x0",seq_embedding[0])
    # # print("q0",q[0])
    # # print("x1",seq_embedding2[0])
    #
    # # seq_embedding = F.normalize(seq_embedding, dim=1)
    # # x = seq_embedding.cpu()
    # print("q2shape",q2.shape)
    # q2 = F.normalize(q2, dim=1)
    # print("q2shape", q2.shape)
    # x = q2.cpu()
    #
    # index = faiss.IndexFlatL2(x.shape[1])
    # index.add(x)
    #
    # q = F.normalize(q, dim=1)
    # q = q.cpu().numpy()
    #
    # D, I = index.search(q, 1000)  # D是距离,I是index的id
    #
    # hit = 0
    # rank_sum = 0
    # no_hit = 0
    # five_hit = 0
    # topone_hit = 0
    # for i, r in enumerate(I):
    #     if y[i] in r:
    #         rank_sum += np.where(r == y[i])[0][0]
    #         if y[i] in r[:10]:
    #             hit += 1
    #         if y[i] in r[:5]:
    #             five_hit += 1
    #         if y[i] in r[:1]:
    #             topone_hit += 1
    #     else:
    #         no_hit += 1
    #
    #
    #
    # print(
    #     f'exp {0} | Mean Rank: {rank_sum / num_queries:.4f}, HR@10: {hit / (num_queries - no_hit):.4f}, No Hit: {no_hit}, HR@5: {five_hit / (num_queries - no_hit):.4f}, HR@1: {topone_hit / (num_queries - no_hit):.4f}')




















    # if not torch.equal(seq_embedding[:5000],q):
    #     print("结果并不相同")
    #     print("se",seq_embedding[0])
    #     print("q",q[0])
    #     b




if __name__ == '__main__':

    city = 'chengdu'
    # city='xian'
    # xian





    #
    # exp_path = '../research/exp/JTMR_chengdu_240819175807'
    # model_name = 'JTMR_chengdu_v1_20_100000_240819175807_19.pt'

    # exp_path = '../research/exp/JTMR_chengdu_240827161810' #只加了语义BERT
    # model_name = 'JTMR_chengdu_v1_40_100000_240827161810_28.pt'

    #
    # exp_path = '../research/exp/JTMR_chengdu_240905155234'
    # model_name = 'JTMR_chengdu_v1_40_100000_240905155234_32.pt'

    # exp_path = '../research/exp/JTMR_chengdu_240904105659'
    # model_name = 'JTMR_chengdu_v1_40_100000_240904105659_39.pt'

    # exp_path = '../research/exp/JTMR_chengdu_240903150213'
    # model_name = 'JTMR_chengdu_v1_40_100000_240903150213_39.pt'

    # exp_path='../research/exp/JTMR_chengdu_240909232620'
    # model_name='JTMR_chengdu_v1_40_100000_240909232620_39.pt'

    # exp_path = '../research/exp/JTMR_chengdu_240910225018'
    # model_name='JTMR_chengdu_v1_40_100000_240910225018_39.pt'

    # exp_path = '../research/exp/JTMR_chengdu_240912163458'
    # model_name='JTMR_chengdu_v1_40_100000_240912163458_32.pt'

    # exp_path = '../research/exp/JTMR_chengdu_240919164928'
    # model_name='JTMR_chengdu_v1_40_100000_240919164928_30.pt'

    # exp_path = '../research/exp/JTMR_chengdu_240920230720' #2+2mLSTM_noresult
    # model_name = 'JTMR_chengdu_v1_40_100000_240920230720_39.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241010180434'  #双语义
    # model_name='JTMR_chengdu_v1_40_100000_241010180434_50.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241019101632'  #MEC_loss
    # model_name='JTMR_chengdu_v1_40_100000_241019101632_38.pt'


    # exp_path = '../research/exp/JTMR_chengdu_241019104151'  #双语义+单向GRU
    # model_name='JTMR_chengdu_v1_40_100000_241019104151_42.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241020231001'  #MEC_loss改了超参数，无predictor
    # model_name='JTMR_chengdu_v1_50_100000_241020231001_49.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241021153202'    #MEC_loss,有predictor
    # model_name='JTMR_chengdu_v1_50_100000_241021153202_49.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241022142057'  #重写clip_1+1mLSTM_双语义
    # model_name='JTMR_chengdu_v1_40_100000_241022142057_39.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241024231254'     #2MHSA
    # model_name='JTMR_chengdu_v1_40_100000_241024231254_39.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241024230900'  # 重写clip,clip不计算mask
    # model_name='JTMR_chengdu_v1_40_100000_241024230900_39.pt'






    # exp_path = '../research/exp/JTMR_chengdu_241029170701' # 轨迹级别的clip
    # model_name='JTMR_chengdu_v1_40_100000_241029170701_38.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241124191232' # 加入gps的时间嵌入
    # model_name='JTMR_chengdu_v1_40_100000_241124191232_29.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_241119201934' #JGRM_bert的prompt finetune
    # model_name='JTMR_chengdu_prompt_finetune_40_100000_241119201934_19.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241124190341' #gps memory
    # model_name='JTMR_chengdu_v1_40_100000_241124190341_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241124190037'  # 更改了loss组成
    # model_name='JTMR_chengdu_v1_40_100000_241124190037_24.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241125221215'  # gps memory 不maskOD,2*match loss
    # model_name='JTMR_chengdu_v1_40_100000_241125221215_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241125231437'  # gps memory 不maskOD,2*match loss+0.5*clip loss
    # model_name='JTMR_chengdu_v1_40_100000_241125231437_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241126001249'  # gps memory 不maskOD,2*match loss，更改了bert的初始化权重
    # model_name='JTMR_chengdu_v1_40_100000_241126001249_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241126201655'  # gps memory 不maskOD,8*match loss
    # model_name='JTMR_chengdu_v1_40_100000_241126201655_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241127140145' #loss2/loss2/loss1
    # model_name='JTMR_chengdu_v1_40_100000_241127140145_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241127140410' #gps_memory只用轨迹级clip
    # model_name='JTMR_chengdu_v1_40_100000_241127140410_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241127191323' #itc+itm
    # model_name='JTMR_chengdu_v1_40_100000_241127191323_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241128000519' #只有itm
    # model_name='JTMR_chengdu_v1_40_100000_241128000519_29.pt'


    # exp_path = '../research/exp/JTMR_chengdu_241128162817' #四层mode interactor
    # model_name='JTMR_chengdu_v1_40_100000_241128162817_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241129181124' #三路
    # model_name='JTMR_chengdu_v1_40_100000_241129181124_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241129210343' #最后加mec
    # model_name='JTMR_chengdu_v1_40_100000_241129210343_29.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_241203221833' #Gps memory的route prompt finetune
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_241203221833_9.pt'


    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_241204112249' #Gps memory的route+bert prompt finetune
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_241204112249_19.pt'

    # exp_path = '../research/exp/JTMR_xian_241206164604' #没有prompt的few shot
    # model_name='JTMR_xian_finetune_30_5000_241206164604_29.pt'

    # exp_path = '../research/exp/JTMR_xian_241206171335' #有prompt的few shot，冻结了大部分
    # model_name='JTMR_xian_finetune_30_5000_241206171335_29.pt'

    # exp_path = '../research/exp/JTMR_xian_241208143410' #有prompt的few shot，全部可训练
    # model_name='JTMR_xian_finetune_30_5000_241208143410_29.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_241212154611' #Gps memory的route+bert prompt(没有时间相关的prompt) finetune
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_241212154611_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250218165757' #Gps memory的全部prompt(没有语义) finetune
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250218165757_19.pt'

    # exp_path = '../research/xian_exp/JTMR_xian_241216184057' #JGRM基线在西安数据集上
    # model_name='JTMR_xian_v1_20_100000_241216184057_19.pt'

    # exp_path = '../research/xian_exp/prompt_finetune/JTMR_xian_241218165914' #我的模型在西安数据集上
    # model_name='JTMR_xian_prompt_finetune_20_100000_241218165914_19.pt'

    # exp_path = '../research/xian_exp/JTMR_chengdu_250221125437' #有prompt的few shot，冻结了大部分,西安->成都
    # model_name='JTMR_chengdu_finetune_30_5000_250221125437_29.pt'

    # exp_path = '../research/xian_exp/JTMR_chengdu_250220212711' #无prompt的few shot,西安->成都
    # model_name='JTMR_chengdu_finetune_30_5000_250220212711_29.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_241212154611' #Gps memory的route+bert prompt(没有时间相关的prompt) finetune
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_241212154611_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250221104537' #hidden_size=32
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250221104537_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250221105738' #hidden_size=64
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250221105738_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250221113456' #hidden_size=128
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250221113456_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250221113933' #hidden_size=512
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250221113933_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250226102619' #hidden_size=256
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250226102619_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250225164353' #prompt generator 2层
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250225164353_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250225164907' #prompt generator 3层
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250225164907_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250225165248' #prompt generator 4层
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250225165248_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250227203133' #bert_layer=2
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250227203133_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250227203348' #bert_layer=6
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250227203348_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250227203518' #bert_layer=8
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250227203518_19.pt'

    # exp_path = '../research/xian_exp/JTMR_chengdu_250307144242' #基线西安->成都
    # model_name='JTMR_chengdu_finetune_20_5000_250307144242_19.pt'

    # exp_path = '../research/exp/JTMR_xian_250307191600' #基线成都->西安
    # model_name='JTMR_xian_finetune_20_5000_250307191600_19.pt'

    exp_path = '../route_ablation_result_model'  # route_ablation消融实验
    model_name = 'JTMR_chengdu_prompt_finetune_20_100000_251216100232_19.pt'

    start_time = time.time()
    log_path = os.path.join(exp_path, 'evaluation')
    sys.stdout = Logger(log_path, start_time, stream=sys.stdout)  # record log
    # sys.stderr = Logger(log_path, start_time, stream=sys.stderr)  # record error

    evaluation(city, exp_path, model_name, start_time)


# travel time estimation  | MAE: 81.7392, RMSE: 109.5158
# similarity search       | Mean Rank: 1.3256, HR@10: 0.9781, No Hit: 1.5




# xian
# travel time estimation  | MAE: 87.3313, RMSE: 119.3156 # 全置为0
# travel time estimation  | MAE: 87.6380, RMSE: 119.6562 # min 置为0 相似度低
# travel time estimation  | MAE: 87.0120, RMSE: 118.7117 # week置为0 相似度高

# chengdu
# travel time estimation  | MAE: 83.7680, RMSE: 111.8366 # 全置为0
# travel time estimation  | MAE: 84.4726, RMSE: 111.8322 # min 置为0 相似度低
# travel time estimation  | MAE: 84.0630, RMSE: 111.7760 # week 置为0 相似度低

# travel time estimation  | MAE: 83.8837, RMSE: 111.5713
# similarity search       | Mean Rank: 6.9435, HR@10: 0.8548, No Hit: 0.0

# travel time estimation  | MAE: 87.1631, RMSE: 119.0558
# similarity search       | Mean Rank: 9.3613, HR@10: 0.8039, No Hit: 0.0
# similarity search       | Mean Rank: 8.6928, HR@10: 0.8197, No Hit: 0.0 # min和delta为0
# similarity search       | Mean Rank: 9.2603, HR@10: 0.8053, No Hit: 0.0 # 全0

# xian
# exp_path = '/data/mazp/exp/JTMR_xian_230819055718'
# model_name = 'JTMR_xian_v1_20_100000_230819055718_19.pt'
# travel time estimation  | MAE: 87.1591, RMSE: 118.8026 # 间隔和分钟置为mask值
# similarity search       | Mean Rank: 6.0727, HR@10: 0.8828, No Hit: 0.0 # 不做特殊处理


# chengdu
# travel time estimation  | MAE: 83.8969, RMSE: 111.8088
# similarity search       | Mean Rank: 6.4462, HR@10: 0.8665, No Hit: 0.0

# chengdu 对比 调整tau
# travel time estimation  | MAE: 83.7518, RMSE: 111.8767
# similarity search       | Mean Rank: 6.4036, HR@10: 0.8652, No Hit: 0.0

# xian 对比 调整tau
# travel time estimation  | MAE: 87.2686, RMSE: 119.1458
# similarity search       | Mean Rank: 6.0667, HR@10: 0.8811, No Hit: 0.0

# chengdu 对比 三角损失
# travel time estimation  | MAE: 83.7620, RMSE: 111.7484
# similarity search       | Mean Rank: 6.5985, HR@10: 0.8657, No Hit: 0.0
# 加norm后结果好一些
# travel time estimation  | MAE: 84.3132, RMSE: 112.0723
# similarity search       | Mean Rank: 5.9975, HR@10: 0.8704, No Hit: 0.0  ind pair detour detour_rate = 0.15
# similarity search       | Mean Rank: 5.8675, HR@10: 0.8870, No Hit: 0.0  cont sub traj detour detour_rate = 0.25
# similarity search       | Mean Rank: 2.7228, HR@10: 0.9424, No Hit: 0.0  cont sub traj detour detour_rate = 0.1
# similarity search       | Mean Rank: 4.3831, HR@10: 0.9182, No Hit: 0.0  cont sub traj detour detour_rate = 0.2
# similarity search       | Mean Rank: 6.6062, HR@10: 0.8689, No Hit: 0.0  cont sub traj detour detour_rate = 0.3
# similarity search       | Mean Rank: 9.6757, HR@10: 0.8199, No Hit: 0.0  cont sub traj detour detour_rate = 0.4

# new detour
# similarity search       | Mean Rank: 3.7809, HR@10: 0.9140, No Hit: 0.0 # 0.2
# similarity search       | Mean Rank: 2.5484, HR@10: 0.9411, No Hit: 0.0 # 0.3

# xian 对比 三角损失
# travel time estimation  | MAE: 87.1660, RMSE: 119.2541
# similarity search       | Mean Rank: 5.9897, HR@10: 0.8864, No Hit: 0.3
# 加norm后结果好一些
# travel time estimation  | MAE: 87.2475, RMSE: 119.3162
# similarity search       | Mean Rank: 5.6661, HR@10: 0.8904, No Hit: 0.0 ind pair detour detour_rate = 0.15
# similarity search       | Mean Rank: 3.8581, HR@10: 0.9232, No Hit: 0.0 cont sub traj detour detour_rate = 0.2
# similarity search       | Mean Rank: 5.8871, HR@10: 0.8924, No Hit: 0.0 cont sub traj detour detour_rate = 0.25

# new detour
# similarity search       | Mean Rank: 4.2792, HR@10: 0.8964, No Hit: 0.0 0.2
# similarity search       | Mean Rank: 2.7714, HR@10: 0.9294, No Hit: 0.0 0.3








# xian 对比 三角损失 + 1*轨迹级对比 CL的加入会有一些负面作用
# travel time estimation  | MAE: 95.5387, RMSE: 127.6472
# similarity search       | Mean Rank: 54.1119, HR@10: 0.7372, No Hit: 1244.3

# xian 对比 三角损失 + 0.1*轨迹级对比 CL的加入会有一些负面作用
# travel time estimation  | MAE: 89.2120, RMSE: 121.4869
# similarity search       | Mean Rank: 13.1270, HR@10: 0.8667, No Hit: 19.8



