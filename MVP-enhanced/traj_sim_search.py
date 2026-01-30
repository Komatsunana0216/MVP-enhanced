import numpy as np
import faiss # https://zhuanlan.zhihu.com/p/320653340
import torch
import math
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence
from evluation_utils import get_seq_emb_from_node,get_seq_emb_from_traj_withRouteOnly
from datetime import datetime
import torch.nn.functional as F
from evluation_utils import get_seq_emb_from_traj_withALLModel
import torch.nn as nn
from tqdm import tqdm
from shapely.geometry import Polygon
def traj_sim(seq_embedding,  seq_model,  source_data,num_nodes,  fold=5):

    print('device: cpu')
    # batch_size = 1024
    num_queries = 5000
    random_index = np.random.permutation(len(seq_embedding))

    print("x", seq_embedding[random_index[0]])

    seq_embedding = F.normalize(seq_embedding, dim=1)
    x = seq_embedding.cpu()

    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)

    hit_list, mean_rank_list, no_hit_list = [], [], []
    five_hit_list,topone_hit_list=[],[]
    for expid in range(fold):
        new_route_assign_mat,new_route_data,new_gps_assign_mat,new_gps_data,gps_length,y= query_prepare(expid, source_data, num_nodes,num_queries,random_index)






        queries = (new_route_data, new_route_assign_mat, new_gps_data, new_gps_assign_mat, new_route_assign_mat, gps_length, None)

        torch.manual_seed(42)
        q = get_seq_emb_from_traj_withALLModel(seq_model, queries, batch_size=32)

        print("q", q[0])
        q = F.normalize(q, dim=1)
        q = q.cpu().numpy()


        D, I = index.search(q, 1000)  # D是距离,I是index的id

        hit = 0
        rank_sum = 0
        no_hit = 0
        five_hit=0
        topone_hit=0
        for i, r in enumerate(I):
            if y[i] in r:
                rank_sum += np.where(r == y[i])[0][0]
                if y[i] in r[:10]:
                    hit += 1
                if y[i] in r[:5]:
                    five_hit += 1
                if y[i] in r[:1]:
                    topone_hit += 1
            else:
                no_hit += 1

        hit_list.append(hit / (num_queries - no_hit))
        five_hit_list.append(five_hit / (num_queries - no_hit))
        topone_hit_list.append(topone_hit / (num_queries - no_hit))

        mean_rank_list.append(rank_sum / num_queries)
        no_hit_list.append(no_hit)

        print(f'exp {expid} | Mean Rank: {rank_sum / num_queries:.4f}, HR@10: {hit / (num_queries - no_hit):.4f}, No Hit: {no_hit}, HR@5: {five_hit / (num_queries - no_hit):.4f}, HR@1: {topone_hit / (num_queries - no_hit):.4f}')
        tingzhi
    print(f'similarity search       | Mean Rank: {np.mean(mean_rank_list):.4f}, HR@10: {np.mean(hit_list):.4f}, No Hit: {np.mean(no_hit_list)},HR@5: {np.mean(five_hit_list):.4f},HR@1: {np.mean(topone_hit_list):.4f}')


def query_prepare(expid, source_data,padding_id,num_queries,random_index):
    np.random.seed(expid)
    route_data=source_data[0]
    route_assign_mat=source_data[1]
    gps_data=source_data[2]
    gps_assign_mat=source_data[3]
    original_gps_length=source_data[4]
    num_samples = len(route_data)


    def downsample_gps(gps_assign_mat, gps_data,route_assign_mat,route_data,num_queries,random_index):
        """
        此函数对 gps_assign_mat 中的元素进行下采样，并将结果应用到 gps_data 上
        :param gps_assign_mat: 输入的 gps_assign_mat 张量，形状为 [50000, 255]
        :param gps_data: 输入的 gps_data 张量，形状为 [50000, 255, 8]
        :return: 处理后的 gps_assign_mat 和 gps_data 张量
        """
        num_rows=num_queries
        num_cols = gps_assign_mat.shape[1]
        route_num_cols=route_assign_mat.shape[1]
        new_gps_assign_mat_list = []
        new_gps_data_list = []
        new_route_assign_mat_list = []
        new_route_data_list = []
        new_route_list=[]
        gps_length_list=[]
        for n in range(num_rows):
            i=random_index[n]

            # 找到起点和终点
            start_index = 0
            end_index = num_cols - 1
            while gps_assign_mat[i, end_index] == padding_id:
                end_index -= 1
            valid_indices = [start_index]
            for j in range(start_index + 1, end_index):
                if torch.rand(1).item() >= 0:  # 以 0.5 的概率保留元素
                    valid_indices.append(j)
            valid_indices.append(end_index)
            to_be_resampled=gps_assign_mat[i, valid_indices]
            new_route=[]
            prev=None
            for k in range(len(to_be_resampled)):
                if k==len(to_be_resampled)-1:
                    if new_route[-1]==to_be_resampled[k]:
                        break
                    else:
                        new_route.append(to_be_resampled[k])
                        break
                if to_be_resampled[k]!=prev:
                    if prev==None:
                        new_route.append(to_be_resampled[k])
                    else:
                        if torch.rand(1).item() >= 0:

                            if to_be_resampled[k]!=new_route[-1]:
                                new_route.append(to_be_resampled[k])
                    prev = to_be_resampled[k]

            # set_new_route=set([road.item() for road in new_route])
            # if len(new_route)!= len(set_new_route):
            #     print(new_route)
            #     print(route_assign_mat[i])
            #     a
            new_route_copy=new_route
            real_indices=[]
            while len(new_route_copy)!=0:
                item=new_route_copy[0]
                new_route_copy=new_route_copy[1:]
                for p in range(len(to_be_resampled)):
                    if to_be_resampled[p]==item:
                        if len(real_indices) != 0 and valid_indices[p] <= real_indices[-1]:
                            continue
                        real_indices.append(valid_indices[p])
                        if p+1<len(to_be_resampled) and to_be_resampled[p+1]!=item:
                            break
            final_sampled=gps_assign_mat[i, real_indices]

            gps_length = []
            prev_value = None
            count = 0
            for value in final_sampled:
                if prev_value is None:
                    prev_value = value
                    count = 1
                elif value == prev_value:
                    count += 1
                else:
                    gps_length.append(count)
                    prev_value = value
                    count = 1
            gps_length.append(count)  # 最后一组相同点的长度
            gps_length_list.append(torch.tensor(gps_length, dtype=torch.int))

            new_gps_assign_mat_list.append(torch.tensor(gps_assign_mat[i, real_indices], dtype=torch.float32))
            new_gps_data_list.append(gps_data[i, real_indices])
            new_route_list.append(new_route)

            # start_index = 0
            # end_index = route_num_cols - 1
            # while route_assign_mat[i, end_index] == padding_id:
            #     end_index -= 1
            # valid_indices = [start_index]
            route_index=[]
            for element in new_route:

                # for m in range(start_index + 1, end_index):
                for m in range(route_num_cols):
                    if route_assign_mat[i, m]==element:
                        if len(route_index)!=0 and m<=route_index[-1]:
                            continue
                        route_index.append(m)
                        break

            if len(route_index)!= len(set(route_index)):
                print(route_index)
                print(new_route)
                print(route_assign_mat[i])
                print("下面这俩")
                print(len(new_route))
                print(len(set(new_route)))
                b
            new_route_assign_mat_list.append(torch.tensor(route_assign_mat[i, route_index], dtype=torch.float32))
            new_route_data_list.append(route_data[i, route_index])


        return new_gps_assign_mat_list, new_gps_data_list, new_route_list, new_route_assign_mat_list, new_route_data_list, gps_length_list


    # 调用函数进行下采样

    new_gps_assign_mat, new_gps_data, new_route_list, new_route_assign_mat, new_route_data, gps_length = downsample_gps(gps_assign_mat, gps_data,route_assign_mat, route_data,num_queries,random_index)

    def pad_sequences_to_length(sequence_list, target_length, padding_value=0):
        """
        此函数将序列列表中的张量填充到指定长度
        :param sequence_list: 输入的张量列表，列表中的每个元素是一个张量
        :param target_length: 要填充到的目标长度
        :param padding_value: 填充值，默认为 0
        :return: 填充后的张量
        """
        padded_sequence = pad_sequence(sequence_list, batch_first=True, padding_value=padding_value)
        if padded_sequence.size(1) < target_length:
            padding_size = target_length - padded_sequence.size(1)
            padding = torch.full((padded_sequence.size(0), padding_size, *padded_sequence.size()[2:]), padding_value)
            padded_sequence = torch.cat([padded_sequence, padding], dim=1)
        elif padded_sequence.size(1) > target_length:
            padded_sequence = padded_sequence[:, :target_length, ...]

        return padded_sequence


    route_target_length = route_assign_mat.shape[1]
    gps_target_length = gps_assign_mat.shape[1]

    new_route_assign_mat = pad_sequences_to_length(new_route_assign_mat, route_target_length, padding_value=padding_id)
    new_route_data=pad_sequences_to_length(new_route_data, route_target_length, padding_value=-100)
    new_route_data[:, :, :-1] = torch.where(new_route_data[:, :, :-1] == -100,
                                           torch.full_like(new_route_data[:, :, :-1], 0), new_route_data[:, :, :-1])
    new_route_data[:, :, -1] = torch.where(new_route_data[:, :, -1] == -100, torch.full_like(new_route_data[:, :, -1], -1), new_route_data[:, :, -1])
    new_gps_assign_mat = pad_sequences_to_length(new_gps_assign_mat, gps_target_length, padding_value=padding_id)
    new_gps_data= pad_sequences_to_length(new_gps_data, gps_target_length, padding_value=0)
    new_gps_data[:, :, 0] = torch.ones_like(new_gps_data[:, :, 0])

    gps_length = rnn_utils.pad_sequence(gps_length, padding_value=0, batch_first=True)


    y = random_index[:num_queries]

    if not torch.equal(new_route_data, route_data[random_index[:num_queries]]):
        print(1)
        print(new_route_data[:3])
        print(route_data[:3])
        c1
    if not torch.equal(new_gps_assign_mat, gps_assign_mat[random_index[:num_queries]]):
        print(2)
        # for i in range(new_gps_assign_mat.shape[0]):
        #     if not torch.equal(new_gps_assign_mat[i], masked_gps_assign_mat[i]):
        #         print(new_gps_assign_mat[i])
        #         print(masked_gps_assign_mat[i])
        #         c_2
        c2
    if not torch.equal(new_gps_data, gps_data[random_index[:num_queries]]):
        print(3)
        # print(new_gps_data[:3])
        # print(gps_data[:3])

        # if not torch.equal(new_route_assign_mat, route_assign_mat[:5000]) or
        # print(new_route_assign_mat[:3])
        # print(route_assign_mat[:3])
        print("输入并不相同")
        c
    if not torch.equal(gps_length, original_gps_length[random_index[:num_queries]]):
        print(4)
        temp=original_gps_length[random_index[:num_queries]]
        print(gps_length[:3])
        print(temp[:3])
        d
    return new_route_assign_mat,new_route_data,new_gps_assign_mat,new_gps_data,gps_length,y
