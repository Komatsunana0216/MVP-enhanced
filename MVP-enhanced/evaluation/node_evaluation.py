import sys

sys.path.append("..")
import numpy as np
import pandas as pd
import time
import pickle
from task import road_cls, speed_inf, time_est, sim_srh
from evluation_utils import get_road, fair_sampling, get_road_emb_from_traj, prepare_data, get_seq_emb_from_node, \
    get_road_emb_from_route_encoder,get_road_emb_from_sanlu
import torch
import torch.nn as nn
import os
import road_emb_concat

torch.set_num_threads(5)

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

    # load task 1 & task2 label
    feature_df = pd.read_csv("../dataset/{}/edge_features.csv".format(city))
    num_nodes = len(feature_df)
    print("num_nodes:", num_nodes)

    # load adj
    edge_index = np.load("../dataset/{}/line_graph_edge_idx.npy".format(city))
    print("edge_index shape:", edge_index.shape)

    # load origin train data
    test_node_data = pd.read_pickle(
        open('../dataset/{}/{}_1101_1115_data_sample10w.pkl'.format(city, city), 'rb'))
    road_list = get_road(test_node_data)
    print('number of road obervased in test data: {}'.format(len(road_list)))

    # sample train data
    num_samples = 'all'  # 'all' or 50000
    if num_samples == 'all':
        pass
    elif isinstance(num_samples, int):
        test_node_data = fair_sampling(test_node_data, num_samples)
    road_list = get_road(test_node_data)
    print('number of road obervased after sampling: {}'.format(len(road_list)))

    # load model
    seq_model = load_model(model_path, dev_id, city)
    print("模型结构", seq_model)
    # print("看看这里是不是有prompt", seq_model.prompt_ST)
    seq_model.eval()

    print('start time : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))))
    print("\n=== Evaluation ===")

    # prepare road task dataset
    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = prepare_data(
        test_node_data, route_min_len, route_max_len, gps_min_len, gps_max_len)
    test_node_data = (
    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset)

    update_road = 'route'
    emb_path = '../dataset/{}/{}_1101_1115_road_embedding_{}_{}_{}.pkl'.format(
        city, city, embedding_name, num_samples, update_road)

    if os.path.exists(emb_path):
        # load road embedding from inference result
        road_embedding = torch.load(emb_path, map_location='cuda:{}'.format(dev_id))['road_embedding']
    else:
        # infer road embedding
        # todo 改了batchsize
        # road_embedding_gpsandroute,road_embedding_gpsandgps,road_list = get_road_emb_from_traj(seq_model, test_node_data, without_gps=False, batch_size=16,
        #                                         update_road=update_road, city=city)
        # road_embedding = get_road_emb_from_sanlu(seq_model, test_node_data, without_gps=False, batch_size=32,
        #                                         update_road=update_road, city=city)
        # road_embedding = get_road_emb_from_route_encoder(seq_model, test_node_data, without_gps=False, batch_size=32,
        #                                         update_road=update_road, city=city)
        road_embedding,road_list = road_emb_concat.get_road_emb_from_concat(seq_model, test_node_data, without_gps=False, batch_size=16,
                                                update_road=update_road, city=city)
        # todo 保存这里先注释掉了
        torch.save({'road_embedding': road_embedding}, emb_path)

    # task 1

    #

#todo 有prompt的用这个（你的模型是prompt finetune的，应该用这个）
    seq_model.route_prompt.mask_cls=True
    seq_model.bert_prompt.mask_cls = True
    cls_mask_embedding,road_list= road_emb_concat.get_road_emb_from_concat(seq_model, test_node_data, without_gps=False, batch_size=16,
                                                update_road=update_road, city=city)
    road_cls.evaluation(cls_mask_embedding, feature_df,road_list)
    seq_model.route_prompt.mask_cls = False
    seq_model.bert_prompt.mask_cls = False

    # cls_mask_embedding,road_list= road_emb_concat.get_road_emb_from_concat(seq_model, test_node_data, without_gps=False, batch_size=16,
    #                                             update_road=update_road, city=city)
    # road_cls.evaluation(cls_mask_embedding, feature_df,road_list)
    #todo 没有prompt用这个
    # road_cls.evaluation(road_embedding, feature_df, road_list)

    # todo 这里改了
    # seq_model.prompt.mask_cls=True
    # cls_mask_embedding= get_road_emb_from_traj(seq_model, test_node_data, without_gps=False, batch_size=16,
    #                                             update_road=update_road, city=city)
    # road_cls.evaluation(cls_mask_embedding, feature_df)
    # seq_model.prompt.mask_cls = False
    # task 2

    # speed_inf.evaluation(road_embedding, feature_df)
    # speed_inf.evaluation(road_embedding_gpsandroute, feature_df)
    # speed_inf.evaluation(road_embedding_gpsandgps, feature_df)
    # speed_inf.evaluation(road_embedding_gpsandroute, feature_df,road_list)
    speed_inf.evaluation(road_embedding, feature_df, road_list)

    end_time = time.time()
    print("cost time : {:.2f} s".format(end_time - start_time))


if __name__ == '__main__':
    city = 'chengdu'
    # city = 'xian'

    # exp_path = 'D:/research/exp/JTMR_now2v_chengdu_230905234531'
    # model_name = 'JTMR_now2v_chengdu_v1_20_100000_230905234531_19.pt'

    # exp_path = 'D:/research/exp/JTMR_xian_230823032506'
    # model_name = 'JTMR_xian_v1_20_100000_230823032506_19.pt'

    # exp_path = 'D:/research/exp/JTMR_chengdu_230911144056'
    # model_name = 'JTMR_chengdu_finetune_20_5000_230911144056_19.pt'

    # exp_path = 'D:/research/exp/JTMR_xian_230911151138'
    # model_name = 'JTMR_xian_finetune_20_5000_230911151138_19.pt'

    # exp_path = 'D:/research/exp/JTMR_chengdu_230911182001'
    # model_name = 'JTMR_chengdu_v1_20_50000_230911182001_19.pt'

    # exp_path = 'D:/research/exp/JTMR_chengdu_230906140616'
    # model_name = 'JTMR_chengdu_v1_20_100000_230906140616_19.pt'

    # 从这里开始
    # exp_path = 'D:/research/exp/JTMR_chengdu_230912215305'
    # model_name = 'JTMR_chengdu_v1_20_100000_230912215305_19.pt'

    # exp_path = 'D:/research/exp/JTMR_chengdu_230912215549'
    # model_name = 'JTMR_chengdu_v1_20_100000_230912215549_19.pt'

    # exp_path = 'D:/research/exp/JTMR_chengdu_230912220101'
    # model_name = 'JTMR_chengdu_v1_20_100000_230912220101_19.pt'

    # exp_path = 'D:/research/exp/JTMR_chengdu_230913163941'
    # model_name = 'JTMR_chengdu_v1_20_100000_230913163941_19.pt'

    # exp_path = 'D:/research/exp/JTMR_chengdu_230913164227'
    # model_name = 'JTMR_chengdu_v1_20_100000_230913164227_19.pt'

    # exp_path = 'D:/research/exp/JTMR_chengdu_230913164332'
    # model_name = 'JTMR_chengdu_v1_20_100000_230913164332_19.pt'

    # exp_path = 'D:/research/exp/JTMR_chengdu_230913165631'
    # model_name = 'JTMR_chengdu_v1_20_100000_230913165631_19.pt'

    # exp_path = 'D:/research/exp/JTMR_chengdu_230913165648'
    # model_name = 'JTMR_chengdu_v1_20_100000_230913165648_19.pt'

    # exp_path = '../research/exp/JTMR_chengdu_240819175807'
    # model_name = 'JTMR_chengdu_v1_20_100000_240819175807_19.pt'

    # exp_path = '../research/exp/JTMR_chengdu_240827161810' #只加了语义BERT
    # model_name = 'JTMR_chengdu_v1_40_100000_240827161810_28.pt'

    # exp_path='../research/exp/JTMR_chengdu_240909232620'  #只用了clip_loss
    # model_name='JTMR_chengdu_v1_40_100000_240909232620_39.pt'

    # exp_path = '../research/exp/JTMR_chengdu_240903150213' #mLSTM 2+2+result_linear128->256升维
    # model_name = 'JTMR_chengdu_v1_40_100000_240903150213_39.pt'

    # exp_path = '../research/exp/JTMR_chengdu_240905155234'    #mLSTM 1+1
    # model_name = 'JTMR_chengdu_v1_40_100000_240905155234_32.pt'

    # exp_path = '../research/exp/JTMR_chengdu_240912163458' #clip+traj不同映射
    # model_name='JTMR_chengdu_v1_40_100000_240912163458_32.pt'

    # exp_path = '../research/exp/JTMR_chengdu_240919164928' #clip后置
    # model_name = 'JTMR_chengdu_v1_40_100000_240919164928_30.pt'

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

    # exp_path = '../research/exp/JTMR_chengdu_241029170701' # 轨迹级别的clip
    # model_name='JTMR_chengdu_v1_40_100000_241029170701_38.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241124191232' # 加入gps的时间嵌入
    # model_name='JTMR_chengdu_v1_40_100000_241124191232_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241124190037'  # 更改了loss组成
    # model_name='JTMR_chengdu_v1_40_100000_241124190037_31.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241124190341' #gps memory
    # model_name='JTMR_chengdu_v1_40_100000_241124190341_29.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_241118174822'  #基线finetune
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_241118174822_19.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241125221215'  # gps memory 不maskOD,2*match loss
    # model_name='JTMR_chengdu_v1_40_100000_241125221215_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241125231437'  # gps memory 不maskOD,2*match loss+0.5*clip loss
    # model_name='JTMR_chengdu_v1_40_100000_241125231437_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241126001249'  # gps memory 不maskOD,2*match loss，更改了bert的初始化权重
    # model_name='JTMR_chengdu_v1_40_100000_241126001249_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241126205105'  # gps memory 不maskOD,4*match loss
    # model_name='JTMR_chengdu_v1_40_100000_241126205105_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241127140145' #loss2/loss2/loss1
    # model_name='JTMR_chengdu_v1_40_100000_241127140145_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241127140410' #gps_memory只用轨迹级clip
    # model_name='JTMR_chengdu_v1_40_100000_241127140410_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241127191323' #itc+itm
    # model_name='JTMR_chengdu_v1_40_100000_241127191323_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241128000519' #只有itm
    # model_name='JTMR_chengdu_v1_40_100000_241128000519_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241129095807' #分层RNN
    # model_name='JTMR_chengdu_v1_40_100000_241129095807_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241129181124' #三路
    # model_name='JTMR_chengdu_v1_40_100000_241129181124_29.pt'

    # exp_path = '../research/exp/JTMR_chengdu_241129210343' #最后加mec
    # model_name='JTMR_chengdu_v1_40_100000_241129210343_29.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_241119201934' #JGRM_bert的prompt finetune
    # model_name='JTMR_chengdu_prompt_finetune_40_100000_241119201934_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250218165757' #Gps memory的全部prompt(没有语义) finetune
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250218165757_19.pt'


    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_241203221833' #Gps memory的route prompt finetune
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_241203221833_9.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_241204112249' #Gps memory的route+bert prompt finetune
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_241204112249_19.pt'

    # exp_path = '../research/exp/JTMR_xian_241206164604' #没有prompt的few shot
    # model_name='JTMR_xian_finetune_30_5000_241206164604_29.pt'

    # exp_path = '../research/exp/JTMR_xian_241206171335' ##有prompt的few shot，冻结了大部分
    # model_name='JTMR_xian_finetune_30_5000_241206171335_29.pt'

    # exp_path = '../research/exp/JTMR_xian_241208143410' #有prompt的few shot，全部可训练
    # model_name='JTMR_xian_finetune_30_5000_241208143410_29.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_241212154611' #Gps memory的route+bert prompt(没有时间相关的prompt) finetune
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_241212154611_19.pt'


    # exp_path = '../research/xian_exp/JTMR_xian_241216184057' #JGRM基线在西安数据集上
    # model_name='JTMR_xian_v1_20_100000_241216184057_19.pt'

    # exp_path = '../research/xian_exp/prompt_finetune/JTMR_xian_241218165914' #我的模型在西安数据集上
    # model_name='JTMR_xian_prompt_finetune_20_100000_241218165914_19.pt'

    # exp_path = '../research/xian_exp/JTMR_chengdu_250221125437' #有prompt的few shot，冻结了大部分,西安->成都
    # model_name='JTMR_chengdu_finetune_30_5000_250221125437_29.pt'

    # exp_path = '../research/xian_exp/JTMR_chengdu_250220212711' #无prompt的few shot,西安->成都
    # model_name='JTMR_chengdu_finetune_30_5000_250220212711_29.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250221104537' #hidden_size=32
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250221104537_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250221105738' #hidden_size=64
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250221105738_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250221113456' #hidden_size=128
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250221113456_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250221113933' #hidden_size=512
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250221113933_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250225164353' #prompt generator 2层
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250225164353_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250225164907' #prompt generator 3层
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250225164907_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250225165248' #prompt generator 4层
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250225165248_19.pt'

    # exp_path = '../research/exp/prompt_finetune/JTMR_chengdu_250226102619' #hidden_size=256
    # model_name='JTMR_chengdu_prompt_finetune_20_100000_250226102619_19.pt'

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

    print(exp_path)

    start_time = time.time()
    log_path = os.path.join(exp_path, 'evaluation')
    # sys.stdout = Logger(log_path, start_time, stream=sys.stdout)  # record log
    # sys.stderr = Logger(log_path, start_time, stream=sys.stderr)  # record error

    evaluation(city, exp_path, model_name, start_time)

# 5w
# road classification     | micro F1: 0.7031, macro F1: 0.7067
# travel speed estimation | MAE: 3.0330, RMSE: 4.0309
# travel time estimation  | MAE: 83.5865, RMSE: 111.6610
# similarity search       | Mean Rank: 2.9798, HR@10: 0.9320, No Hit: 0.0

# road classification     | micro F1: 0.7331, macro F1: 0.7361
# travel speed estimation | MAE: 2.6225, RMSE: 3.5866
