import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup, AdamW
from utils import weight_init
from dataloader import get_train_loader, random_mask
from utils import setup_seed
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from JGRM_bert_prompt_GPSmemory import JGRMModel
# from JGRM_bert_prompt_GPSmemory_ablation_route import JGRMModel
from cl_loss import get_traj_cl_loss, get_road_cl_loss, get_traj_cluster_loss, get_traj_match_loss
from dcl import DCL
import os
import torch



dev_id = 4
torch.cuda.set_device('cuda:4')
torch.set_num_threads(10)

def train(config):

    city = config['city']

    vocab_size = config['vocab_size']
    num_samples = config['num_samples']
    data_path = config['data_path']
    adj_path = config['adj_path']
    retrain = config['retrain']
    save_path = config['save_path']

    num_worker = config['num_worker']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    warmup_step = config['warmup_step']
    weight_decay = config['weight_decay']

    route_min_len = config['route_min_len']
    route_max_len = config['route_max_len']
    gps_min_len = config['gps_min_len']
    gps_max_len = config['gps_max_len']

    road_feat_num = config['road_feat_num']
    road_embed_size = config['road_embed_size']
    gps_feat_num = config['gps_feat_num']
    gps_embed_size = config['gps_embed_size']
    route_embed_size = config['route_embed_size']

    hidden_size = config['hidden_size']
    drop_route_rate = config['drop_route_rate'] # route_encoder
    drop_edge_rate = config['drop_edge_rate']   # gat
    drop_road_rate = config['drop_road_rate']   # sharedtransformer

    verbose = config['verbose']
    version = config['version']
    seed = config['random_seed']

    mask_length = config['mask_length']
    mask_prob = config['mask_prob']

    bert_hidden_size=config['bert_hidden_size']
    bert_attention_heads=config['bert_attention_heads']
    bert_hidden_layers=config['bert_hidden_layers']

    edge_features_path=config['edge_features_path']

    prompt_ST = config['prompt_ST']
    # 设置随机种子
    setup_seed(seed)

    # define model, parmeters and optimizer
    # edge_features=pd.read_csv(edge_features_path)
    edge_index = np.load(adj_path)
    model = JGRMModel(prompt_ST, vocab_size, route_max_len, road_feat_num, road_embed_size, gps_feat_num,
                      gps_embed_size, route_embed_size, hidden_size, edge_index, drop_edge_rate, drop_route_rate, drop_road_rate, bert_hiden_size=bert_hidden_size, pad_token_id=vocab_size, bert_attention_heads=bert_attention_heads, bert_hidden_layers=bert_hidden_layers,bert_vocab_size=vocab_size+1,mode='x').cuda()
    #todo 做parameter sensitivity的时候把初始化参数去掉了
    init_road_emb = torch.load('./dataset/{}/init_w2v_road_emb.pt'.format(city), map_location='cuda:{}'.format(dev_id))
    model.node_embedding.weight = torch.nn.Parameter(init_road_emb['init_road_embd'])
    model.node_embedding.requires_grad_(True)

    # pad_weight = torch.nn.Parameter(torch.ones(bert_hidden_size)).cuda()
    # pad_weight = pad_weight.unsqueeze(0)
    # init_bert_weight=torch.nn.Parameter(init_road_emb['init_road_embd'])
    # model.seg_embedding_learning.bert.embeddings.word_embeddings.weight = torch.nn.Parameter(torch.concat([init_bert_weight,pad_weight],dim=0))
    #
    # model.seg_embedding_learning.bert.embeddings.word_embeddings.requires_grad_(True)
    print('load parameters in device {}'.format(model.node_embedding.weight.device)) # check process device

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # exp information
    nowtime = datetime.now().strftime("%y%m%d%H%M%S")
    # nowtime="241121123309"            #TODO:这里改了
    model_name = 'JTMR_{}_{}_{}_{}_{}'.format(city, version, num_epochs, num_samples, nowtime)  #TODO:这里改了
    # model_name = 'JTMR_{}_{}_{}_{}_{}'.format(city, version, 40, num_samples, nowtime)
    model_path = os.path.join(save_path, 'JTMR_{}_{}'.format(city, nowtime), 'model')
    log_path = os.path.join(save_path, 'JTMR_{}_{}'.format(city, nowtime), 'log')

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # print("model_path是什么",model_path)
    # print("model_name是什么",model_name)
    checkpoints = [f for f in os.listdir(model_path) if f.startswith(model_name)]
    writer = SummaryWriter(log_path)
    # with open(f'{log_path}/loss.txt', 'a') as f:
    print("checkpoints有什么",checkpoints)
    if not retrain and checkpoints:
        print("进到这里来了")
        print("排序后的checkpoint_path是什么", sorted(checkpoints))
        # checkpoint_path = os.path.join(model_path, sorted(checkpoints)[-1])  #TODO:这里改了
        checkpoint_path = os.path.join(model_path, sorted(checkpoints)[-7])
        checkpoint = torch.load(checkpoint_path)
        print("选中的checkpoint_keys是什么",checkpoint.keys())
        print("选中的checkpoint是什么", checkpoint_path)
        # model.load_state_dict(checkpoint['model_state_dict'])  #TODO:这里改了
        model=torch.load(checkpoint_path)['model']

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model.apply(weight_init)

    train_loader = get_train_loader(data_path, batch_size, num_worker, route_min_len, route_max_len, gps_min_len, gps_max_len, num_samples, seed)
    print('dataset is ready.')

    epoch_step = train_loader.dataset.route_data.shape[0] // batch_size
    total_steps = epoch_step * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=total_steps)

    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(train_loader):
            gps_data, gps_assign_mat, route_data, route_assign_mat, gps_length = batch

            masked_route_assign_mat, masked_gps_assign_mat = random_mask(gps_assign_mat, route_assign_mat, gps_length,
                                                                         vocab_size, mask_length, mask_prob)

            route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length =\
                route_data.cuda(), masked_route_assign_mat.cuda(), gps_data.cuda(), masked_gps_assign_mat.cuda(), route_assign_mat.cuda(), gps_length.cuda()

            gps_road_rep, gps_traj_rep, route_road_rep, route_traj_rep, \
            gps_road_joint_rep, gps_traj_joint_rep, route_road_joint_rep, route_traj_joint_rep, loss1 \
                = model(route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length)

            # # project rep into the same space
            # proj_gps_traj_rep = model.clip_gps_proj_head(gps_traj_rep)
            # proj_route_traj_rep = model.clip_route_proj_head(route_traj_rep)
            # # print("proj_gps_traj_rep形状",proj_gps_traj_rep.shape)
            # norm_gps_traj_rep = F.normalize(proj_gps_traj_rep, dim=1)
            # norm_route_traj_rep = F.normalize(proj_route_traj_rep, dim=1)
            # tau = 0.07
            # sim_g2r = norm_gps_traj_rep @ norm_route_traj_rep.t() / tau
            # sim_r2g = norm_route_traj_rep @ norm_gps_traj_rep.t() / tau
            # targets=torch.arange(norm_gps_traj_rep.shape[0]).cuda()
            # loss_i = F.cross_entropy(sim_g2r, targets)
            # loss_t = F.cross_entropy(sim_r2g, targets)
            # clip_loss = (loss_i + loss_t) / 2




            # flatten road_rep
            mat2flatten = {}
            y_label = []
            route_length = (route_assign_mat != model.vocab_size).int().sum(1)
            gps_road_list, route_road_list, gps_road_joint_list, route_road_joint_list = [], [], [], []
            now_flatten_idx = 0
            for i, length in enumerate(route_length):
                y_label.append(route_assign_mat[i, :length]) # route 和 gps mask 位置是一样的
                gps_road_list.append(gps_road_rep[i, :length])
                route_road_list.append(route_road_rep[i, :length])
                gps_road_joint_list.append(gps_road_joint_rep[i, :length])
                route_road_joint_list.append(route_road_joint_rep[i, :length])
                for l in range(length):
                    mat2flatten[(i, l)] = now_flatten_idx
                    now_flatten_idx += 1

            y_label = torch.cat(y_label, dim=0)
            gps_road_rep = torch.cat(gps_road_list, dim=0)
            route_road_rep = torch.cat(route_road_list, dim=0)
            gps_road_joint_rep = torch.cat(gps_road_joint_list, dim=0)

            route_road_joint_rep = torch.cat(route_road_joint_list, dim=0)

            # project rep into the same space
            gps_traj_rep = model.gps_proj_head(gps_traj_rep)
            route_traj_rep = model.route_proj_head(route_traj_rep)

            #todo 这个是加的
            # gps_traj_joint_rep = model.gps_joint_proj_head(gps_traj_joint_rep)
            # route_traj_joint_rep = model.route_joint_proj_head(route_traj_joint_rep)
            #
            # (GRM LOSS) get gps & route rep matching loss
            tau = 0.07
            match_loss = get_traj_match_loss(gps_traj_rep, route_traj_rep, model, batch_size, tau)
            # loss_itc,loss_itm = get_traj_match_loss(gps_traj_rep, route_traj_rep,gps_traj_joint_rep,route_traj_joint_rep, model, batch_size, tau)
            # loss_itm_pre,loss_itm = get_traj_match_loss(gps_traj_rep, route_traj_rep, gps_traj_joint_rep,
            #                                          route_traj_joint_rep, model, batch_size, tau)

            # lamda = gps_road_joint_rep.size(-1) / (gps_road_joint_rep.shape[0] * 512)  # d/(B*yibuxulong**2)
            # # print("lamda是多少",lamda)
            # param_k = 5  # 泰勒展开的k
            #
            # gps_view, route_view = F.normalize(gps_road_joint_rep), F.normalize(route_road_joint_rep)
            # c = torch.mm(gps_view, route_view.transpose(0, 1)) * lamda  # (B, B)
            # power = c
            # sum_p = torch.zeros_like(power)
            # for k in range(1, param_k + 1):
            #     if k > 1:
            #         power = torch.mm(power, c)
            #     if (k + 1) % 2 == 0:
            #         sum_p += power / k
            #     else:
            #         sum_p -= power / k
            # trace1 = torch.trace(sum_p)
            #
            # c2 = torch.mm(route_view, gps_view.transpose(0, 1)) * lamda  # (B, B)
            # power2 = c2
            # sum_p2 = torch.zeros_like(power2)
            # for k in range(1, param_k + 1):
            #     if k > 1:
            #         power2 = torch.mm(power2, c2)
            #     if (k + 1) % 2 == 0:
            #         sum_p2 += power2 / k
            #     else:
            #         sum_p2 -= power2 / k
            # trace2 = torch.trace(sum_p2)
            #
            # mec_loss = (trace1 + trace2) * 0.5 / gps_road_joint_rep.shape[0]
            # mec_loss = -1 * mec_loss / lamda







            # prepare label and mask_pos
            masked_pos = torch.nonzero(route_assign_mat != masked_route_assign_mat)
            masked_pos = [mat2flatten[tuple(pos.tolist())] for pos in masked_pos]
            y_label = y_label[masked_pos].long()

            # (MLM 1 LOSS) get gps rep road loss
            gps_mlm_pred = model.gps_mlm_head(gps_road_joint_rep) # project head 也会被更新
            masked_gps_mlm_pred = gps_mlm_pred[masked_pos]
            gps_mlm_loss = nn.CrossEntropyLoss()(masked_gps_mlm_pred, y_label)

            # (MLM 2 LOSS) get route rep road loss
            route_mlm_pred = model.route_mlm_head(route_road_joint_rep) # project head 也会被更新
            masked_route_mlm_pred = route_mlm_pred[masked_pos]
            route_mlm_loss = nn.CrossEntropyLoss()(masked_route_mlm_pred, y_label)

            # MLM 1 LOSS + MLM 2 LOSS + GRM LOSS
            # loss = (7/17)*route_mlm_loss + (7/17)*gps_mlm_loss + (3/17)*loss1 + match_loss
            # loss = (route_mlm_loss + gps_mlm_loss + loss1 + 2 * match_loss+0.5*clip_loss) / 4
            loss = (route_mlm_loss + gps_mlm_loss + loss1 + 2 * match_loss) / 4
            # loss = (route_mlm_loss + gps_mlm_loss + loss1 + 0.25*clip_loss) / 4
            # loss =( route_mlm_loss+gps_mlm_loss / (gps_mlm_loss/ route_mlm_loss + 1e-4).detach() + loss1 / (loss1/ route_mlm_loss + 1e-4).detach()+match_loss / (match_loss/ route_mlm_loss + 1e-4).detach())/4
            # loss = (route_mlm_loss + gps_mlm_loss + loss1 + 0.25 * loss_itc+2*loss_itm) / 4
            # loss = (route_mlm_loss + gps_mlm_loss + loss1 +2 * loss_itm_pre+loss_itm) / 4
            # loss = (route_mlm_loss + gps_mlm_loss + loss1 + 2 * match_loss+mec_loss) / 4
            # step = epoch_step*(epoch+40) + idx    #TODO:这里改了
            step = epoch_step * epoch + idx
            # f.write(f"match_loss:{match_loss}\n gps_mlm_loss:{gps_mlm_loss}\nroute_mlm_loss:{route_mlm_loss}\nloss:{loss}\nstep:{step}\n")
            writer.add_scalar('match_loss/match_loss', match_loss, step)
            # writer.add_scalar('mec_loss/mec_loss', mec_loss, step)
            # writer.add_scalar('clip_loss/clip_loss', clip_loss, step)
            # writer.add_scalar('itc_loss/itc_loss', loss_itc, step)
            # writer.add_scalar('itm_loss/itm_loss_pre', loss_itm_pre, step)
            # writer.add_scalar('itm_loss/itm_loss', loss_itm, step)
            writer.add_scalar('mlm_loss/gps_mlm_loss', gps_mlm_loss, step)
            writer.add_scalar('mlm_loss/route_mlm_loss', route_mlm_loss, step)
            writer.add_scalar('mlm_loss/semantic_mlm_loss', loss1, step)
            writer.add_scalar('loss', loss, step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not (idx + 1) % verbose:
                t = datetime.now().strftime('%m-%d %H:%M:%S')
                print(f'{t} | (Train) | Epoch={epoch}\tbatch_id={idx + 1}\tloss={loss.item():.4f}')

        scheduler.step()

        torch.save({
            # 'epoch': epoch+40,        #TODO:这里改了
            'epoch': epoch,
            'model': model,
            'optimizer_state_dict': optimizer.state_dict()
        # }, os.path.join(model_path, "_".join([model_name, f'{epoch+40}.pt'])))   #TODO:这里改了
        }, os.path.join(model_path, "_".join([model_name, f'{epoch}.pt'])))
    return model

if __name__ == '__main__':
    config = json.load(open('config/chengdu.json', 'r'))
    train(config)


