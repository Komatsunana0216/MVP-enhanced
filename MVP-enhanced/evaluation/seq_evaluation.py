import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import time
from utils import Logger
from task import  time_est, sim_srh
from evluation_utils import get_road, fair_sampling, get_seq_emb_from_traj_withRouteOnly, get_seq_emb_from_traj_withALLModel, prepare_data
import torch
import torch.nn as nn
import os
torch.set_num_threads(5)
os.environ['TORCH_USE_CUDA_DSA']='1'


dev_id = 0
if torch.cuda.is_available():
    torch.cuda.set_device('cuda:0')


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
        from MVP import MVPModel
        model = MVPModel(*checkpoint['model_params']).to("cuda:{}".format(dev_id))
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"[OK] 模型加载成功! epoch={checkpoint.get('epoch')}, loss={checkpoint.get('loss')}")
    elif 'model_state_dict' in checkpoint:
        # prompt finetune模型：只有state_dict，需要从配置文件读取参数并添加prompt组件
        from MVP import MVPModel
        
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
        model = MVPModel(
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
        open('../dataset/{}/{}_1101_1115_data_sample10w.pkl'.format(city, city), 'rb'))

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

    print('start time : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("\n=== Evaluation ===")

    # prepare sequence task
    test_seq_data = pd.read_pickle(
        open('../dataset/{}/{}_1101_1115_data_seq_evaluation.pkl'.format(city, city),
             'rb'))

    test_seq_data = test_seq_data.sample(50000, random_state=0)

    route_length = test_seq_data['route_length'].values

    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = prepare_data(
        test_seq_data, route_min_len, route_max_len, gps_min_len, gps_max_len)
    route_data[:, :, 1] = torch.zeros_like(route_data[:, :, 1]).long()
    route_data[:, :, 2] = torch.full_like(route_data[:, :, 2], fill_value=-1)

    gps_data[:, :, -2] = torch.full_like(gps_data[:, :, -2], fill_value=-1)
    gps_data[:, :, 0] = torch.ones_like(gps_data[:, :, 0])
    test_data = (route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset)

    all_seq_embedding = get_seq_emb_from_traj_withALLModel(seq_model, test_data, batch_size=16)
    # task 3
    time_est.evaluation(all_seq_embedding, test_seq_data, num_nodes)

    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = prepare_data(
        test_seq_data, route_min_len, route_max_len, gps_min_len, gps_max_len)
    test_data = (
        route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset)
    seq_embedding = get_seq_emb_from_traj_withRouteOnly(seq_model, test_data, batch_size=1024)

    geometry_df = pd.read_csv("../dataset/{}/edge_geometry.csv".format(city))

    trans_mat = np.load('../dataset/{}/transition_prob_mat.npy'.format(city))
    trans_mat = torch.tensor(trans_mat)

    sim_srh.evaluation3(seq_embedding, None, seq_model, test_seq_data, num_nodes, trans_mat, feature_df, geometry_df,
                        detour_rate=0.3, fold=10)

    end_time = time.time()
    print("cost time : {:.2f} s".format(end_time - start_time))


if __name__ == '__main__':

    city = 'chengdu'

    exp_path = '../route_ablation_result_model'  # route_ablation消融实验
    model_name = 'JTMR_chengdu_prompt_finetune_20_100000_251216100232_19.pt'

    start_time = time.time()
    log_path = os.path.join(exp_path, 'evaluation')
    sys.stdout = Logger(log_path, start_time, stream=sys.stdout)  # record log

    evaluation(city, exp_path, model_name, start_time)
