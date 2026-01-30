import os
import numpy as np
import pandas as pd
import faiss
import torch
from tqdm import tqdm
from shapely.geometry import Polygon

from utils import next_batch_index


def data_loader(data_path, file_path, padding_id, num_queries, trans_mat,geometry_df,detour_rate=0.3):
    min_len, max_len = 10, 100

    # dfs = []
    # for file_name in file_list:
    #     tmp_df = pd.read_csv(os.path.join(data_path, file_name))
    #     dfs.append(tmp_df)
    # df = pd.concat(dfs).reset_index(drop=True)
    # df['path'] = df['path'].map(eval)
    df = pd.read_pickle(open(file_path, 'rb'))
    df['path'] = df['cpath_list']
    df['path_len'] = df['path'].map(len)
    df = df.loc[(df['path_len'] > min_len) & (df['path_len'] < max_len)]
    df=df[:50000]

    num_samples = len(df)
    x_arr = np.full([num_samples, max_len], padding_id, dtype=np.int32)
    for i in tqdm(range(num_samples)):
        row = df.iloc[i]
        path_arr = np.array(row['path'], dtype=np.int32)
        x_arr[i, :row['path_len']] = path_arr

    def dfs_path(start, end, detour_path, origin_path):
        stack = [(start, [start])]
        paths = []
        while stack:
            (vertex, path) = stack.pop()

            if vertex == end and path != detour_path:  # 如果到达终点，则将路径添加到结果列表中
                paths.append(path)
                # after_detour_length = np.sum([road_length_dict[road] for road in path])
                # before_detour_length = np.sum([road_length_dict[road] for road in detour_path])
                # path_length = np.sum([road_length_dict[road] for road in origin_path])

                poly = detour_path[::-1][:-1] + path
                pt_list = []
                for road in poly:
                    pt_list += [float(item) for pair in geometry_df.iloc[road]['geometry'][12:-1].split(', ') for item
                                in pair.split(' ')]
                area = Polygon([[pt_list[i], pt_list[i + 1]] for i in range(0, len(pt_list), 2)]).area

                if area > 1e-6:

                    return path

            if len(path) - 1 == int(1 / 3 * len(origin_path)) + 1:  # 如果长度到达最大长度但不符合标准则去除
                continue


            for neighbor in torch.nonzero(trans_mat[vertex] != 0).reshape(-1, ).numpy().tolist():
                if neighbor not in path:  # 如果邻居节点尚未访问过，则将其添加到路径中，并将其压入栈中
                    stack.append((neighbor, path + [neighbor]))
        if len(paths) == 0:
            return None
        return paths[-1]
    def detour(replace_rate, path,  detour_path, start_pos):
        # 需要重建detour_base
        # 开始和结束的位置不变

        detour_anchor = dfs_path(detour_path[0], detour_path[-1], detour_path, path)
        if detour_anchor is None:
            detour_anchor = detour_path
        end_pos = start_pos + len(detour_path)
        pre_path = path[:start_pos]
        next_path = path[end_pos:]
        p = np.random.random_sample()  # 产生[0,1)之间的随机数
        if p > replace_rate:
            new_path = pre_path + detour_anchor + next_path
        else:
            new_path = pre_path + [padding_id] * len(detour_path) + next_path

        return new_path

    random_index = np.random.permutation(num_samples)
    q_arr = np.full([num_queries, max_len], padding_id, dtype=np.int32)
    for i in tqdm(range(num_queries)):
        row = df.iloc[random_index[i]]
        sample_len = int(row['route_length'] * detour_rate) + 1
        path = row['cpath_list']
        try_count = 0
        sample_pos_list = list(range(1, row['route_length'] - sample_len, 1))
        while path == row['cpath_list'] and try_count < 10 and len(sample_pos_list) != 0:
            try_count += 1
            start_pos = np.random.choice(sample_pos_list, 1)[0]  # OD不参与处理
            sample_pos_list.remove(start_pos)
            detour_path = row['cpath_list'][start_pos:start_pos + sample_len]
            path= detour(0, row['cpath_list'],  detour_path, start_pos)



        q_arr[i, :len(path)] = np.array(path, dtype=np.int32)

    y = random_index[:num_queries]
    return torch.LongTensor(x_arr), torch.LongTensor(q_arr), y


def evaluation(model, data_path, file_list, num_nodes,trans_mat,geometry_df):
    print("\n--- Similarity Search ---")
    batch_size = 64
    num_queries = 5000
    data, queries, y = data_loader(data_path, file_list, num_nodes, num_queries,trans_mat,geometry_df)
    data_size = data.shape[0]
    print("data_size",data_size)
    model.eval()
    x = []
    with torch.no_grad():
        for batch_idx in tqdm(next_batch_index(data_size, batch_size, shuffle=False)):
            data_batch = data[batch_idx].cuda()
            seq_rep = model.encode_sequence(data_batch)
            if isinstance(seq_rep, tuple):
                seq_rep = seq_rep[0]
            x.append(seq_rep.detach().cpu())
    x = torch.cat(x, dim=0).numpy()

    q = []
    with torch.no_grad():
        for batch_idx in tqdm(next_batch_index(num_queries, batch_size, shuffle=False)):
            q_batch = queries[batch_idx].cuda()
            seq_rep = model.encode_sequence(q_batch)
            if isinstance(seq_rep, tuple):
                seq_rep = seq_rep[0]
            q.append(seq_rep.detach().cpu())
    q = torch.cat(q, dim=0).numpy()

    index = faiss.IndexFlatL2(x.shape[1])
    index.add(x)
    D, I = index.search(q, 1000)
    hit = 0
    rank_sum = 0
    no_hit = 0
    five_hit = 0
    topone_hit = 0
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
    print(f'Mean Rank: {rank_sum / num_queries}, HR@10: {hit / (num_queries - no_hit)}, No Hit: {no_hit},HR@5: {five_hit / (num_queries - no_hit):.4f}, HR@1: {topone_hit / (num_queries - no_hit):.4f}')
