"""
Porto 数据集预处理脚本
将 porto_data.pkl 和 porto_data_evaluation.pkl 转换为 TrajCL 需要的格式
"""
import sys
sys.path.append('..')
import os
import math
import time
import random
import logging
import torch
import pickle
import pandas as pd
import numpy as np

from config import Config
from utils import tool_funcs
from utils.cellspace import CellSpace
from utils.tool_funcs import lonlat2meters
from model.node2vec_ import train_node2vec


def inrange(lon, lat):
    """检查点是否在配置的边界范围内"""
    if lon <= Config.min_lon or lon >= Config.max_lon \
            or lat <= Config.min_lat or lat >= Config.max_lat:
        return False
    return True


def convert_porto_data():
    """
    将 porto_data.pkl 转换为 TrajCL 需要的格式
    输入格式: lng_list, lat_list, tm_list 等列
    输出格式: wgs_seq, merc_seq, ptime, trajlen
    """
    _time = time.time()
    
    # 读取原始数据
    input_file = Config.root_dir + '/data/porto/porto_data.pkl'
    dfraw = pd.read_pickle(input_file)
    logging.info('Loaded raw data. #traj={}'.format(dfraw.shape[0]))
    
    # 将 lng_list 和 lat_list 合并为 wgs_seq
    dfraw['wgs_seq'] = dfraw.apply(
        lambda row: [[lng, lat] for lng, lat in zip(row['lng_list'], row['lat_list'])], 
        axis=1
    )
    
    # 保留时间信息
    dfraw['ptime'] = dfraw['tm_list'].apply(str)
    
    # 计算轨迹长度
    dfraw['trajlen'] = dfraw['wgs_seq'].apply(lambda traj: len(traj))
    
    # 长度过滤
    dfraw = dfraw[(dfraw.trajlen >= Config.min_traj_len) & (dfraw.trajlen <= Config.max_traj_len)]
    logging.info('After length filter. #traj={}'.format(dfraw.shape[0]))
    
    # 范围过滤
    dfraw['inrange'] = dfraw.wgs_seq.map(
        lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj)
    )
    dfraw = dfraw[dfraw.inrange == True]
    logging.info('After range filter. #traj={}'.format(dfraw.shape[0]))
    
    # 转换为墨卡托坐标
    dfraw['merc_seq'] = dfraw.wgs_seq.apply(
        lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj]
    )
    
    # 只保留需要的列并重置索引
    dfraw = dfraw[['trajlen', 'wgs_seq', 'merc_seq', 'ptime']].reset_index(drop=True)
    
    # 保存训练数据
    output_file = Config.dataset_file
    dfraw.to_pickle(output_file)
    logging.info('Saved training data to {}. #traj={}'.format(output_file, dfraw.shape[0]))
    logging.info('convert_porto_data done. @={:.0f}s'.format(time.time() - _time))
    return dfraw


def convert_porto_test_data():
    """
    将 porto_data_evaluation.pkl 转换为 TrajCL 需要的测试格式
    """
    _time = time.time()
    
    # 读取测试数据
    input_file = Config.root_dir + '/data/porto/porto_data_evaluation.pkl'
    dfraw = pd.read_pickle(input_file)
    logging.info('Loaded test data. #traj={}'.format(dfraw.shape[0]))
    
    # 将 lng_list 和 lat_list 合并为 wgs_seq
    dfraw['wgs_seq'] = dfraw.apply(
        lambda row: [[lng, lat] for lng, lat in zip(row['lng_list'], row['lat_list'])], 
        axis=1
    )
    
    # 保留时间信息
    dfraw['ptime'] = dfraw['tm_list'].apply(str)
    
    # 计算轨迹长度
    dfraw['trajlen'] = dfraw['wgs_seq'].apply(lambda traj: len(traj))
    
    # 长度过滤
    dfraw = dfraw[(dfraw.trajlen >= Config.min_traj_len) & (dfraw.trajlen <= Config.max_traj_len)]
    logging.info('After length filter. #traj={}'.format(dfraw.shape[0]))
    
    # 范围过滤
    dfraw['inrange'] = dfraw.wgs_seq.map(
        lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj)
    )
    dfraw = dfraw[dfraw.inrange == True]
    logging.info('After range filter. #traj={}'.format(dfraw.shape[0]))
    
    # 转换为墨卡托坐标
    dfraw['merc_seq'] = dfraw.wgs_seq.apply(
        lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj]
    )
    
    # 只保留需要的列并重置索引
    dfraw = dfraw[['trajlen', 'wgs_seq', 'merc_seq', 'ptime']].reset_index(drop=True)
    
    # 保存测试数据
    output_file = Config.test_dataset_file
    dfraw.to_pickle(output_file)
    logging.info('Saved test data to {}. #traj={}'.format(output_file, dfraw.shape[0]))
    logging.info('convert_porto_test_data done. @={:.0f}s'.format(time.time() - _time))
    return dfraw


def init_cellspace():
    """
    创建网格空间并训练 node2vec 嵌入
    """
    _time = time.time()
    
    x_min, y_min = lonlat2meters(Config.min_lon, Config.min_lat)
    x_max, y_max = lonlat2meters(Config.max_lon, Config.max_lat)
    x_min -= Config.cellspace_buffer
    y_min -= Config.cellspace_buffer
    x_max += Config.cellspace_buffer
    y_max += Config.cellspace_buffer

    cell_size = int(Config.cell_size)
    cs = CellSpace(cell_size, cell_size, x_min, y_min, x_max, y_max)
    
    logging.info('CellSpace created: {}'.format(cs))
    logging.info('Total cells: {}'.format(cs.size()))
    
    with open(Config.dataset_cell_file, 'wb') as fh:
        pickle.dump(cs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info('Saved cellspace to {}'.format(Config.dataset_cell_file))

    _, edge_index = cs.all_neighbour_cell_pairs_permutated_optmized()
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=Config.device).T
    train_node2vec(edge_index)
    
    logging.info('init_cellspace done. @={:.0f}s'.format(time.time() - _time))
    return


def generate_newsimi_test_dataset():
    """
    生成用于相似性测试的数据集
    """
    _time = time.time()
    
    trajs = pd.read_pickle(Config.dataset_file)
    l = trajs.shape[0]
    n_query = min(1000, int(l * 0.1))  # 查询数量
    n_db = min(l, 100000)  # 数据库大小
    
    logging.info('Generating newsimi test dataset. total={}, n_query={}, n_db={}'.format(l, n_query, n_db))
    
    # 使用后 80% 的数据作为测试
    test_idx_start = int(l * 0.8)
    test_trajs = trajs[test_idx_start: min(test_idx_start + n_db, l)]
    
    def _raw_dataset():
        query_lst = []
        db_lst = []
        i = 0
        for _, v in test_trajs.merc_seq.items():
            if i < n_query:
                query_lst.append(np.array(v)[::2].tolist())
            db_lst.append(np.array(v)[1::2].tolist())
            i += 1

        output_file_name = Config.dataset_file + '_newsimi_raw.pkl'
        with open(output_file_name, 'wb') as fh:
            pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("_raw_dataset done. saved to {}".format(output_file_name))
        return

    def _downsample_dataset(rate):
        unrate = 1 - rate
        query_lst = []
        db_lst = []
        i = 0
        for _, v in test_trajs.merc_seq.items():
            if i < n_query:
                _q = np.array(v)[::2]
                _q_len = _q.shape[0]
                _idx = np.sort(np.random.choice(_q_len, max(1, math.ceil(_q_len * unrate)), replace=False))
                query_lst.append(_q[_idx].tolist())
            _db = np.array(v)[1::2]
            _db_len = _db.shape[0]
            _idx = np.sort(np.random.choice(_db_len, max(1, math.ceil(_db_len * unrate)), replace=False))
            db_lst.append(_db[_idx].tolist())
            i += 1

        output_file_name = Config.dataset_file + '_newsimi_downsampling_' + str(rate) + '.pkl'
        with open(output_file_name, 'wb') as fh:
            pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("_downsample_dataset done. rate={}".format(rate))
        return

    def _distort_dataset(rate):
        query_lst = []
        db_lst = []
        i = 0
        for _, v in test_trajs.merc_seq.items():
            if i < n_query:
                _q = np.array(v)[::2]
                for _row in range(_q.shape[0]):
                    if random.random() < rate:
                        _q[_row] = _q[_row] + [tool_funcs.truncated_rand(), tool_funcs.truncated_rand()]
                query_lst.append(_q.tolist())
            
            _db = np.array(v)[1::2]
            for _row in range(_db.shape[0]):
                if random.random() < rate:
                    _db[_row] = _db[_row] + [tool_funcs.truncated_rand(), tool_funcs.truncated_rand()]
            db_lst.append(_db.tolist())
            i += 1

        output_file_name = Config.dataset_file + '_newsimi_distort_' + str(rate) + '.pkl'
        with open(output_file_name, 'wb') as fh:
            pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("_distort_dataset done. rate={}".format(rate))
        return

    _raw_dataset()

    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _downsample_dataset(rate)

    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _distort_dataset(rate)

    logging.info('generate_newsimi_test_dataset done. @={:.0f}s'.format(time.time() - _time))
    return


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
        handlers=[
            logging.FileHandler(Config.root_dir + '/exp/log/' + tool_funcs.log_file_name(), mode='w'),
            logging.StreamHandler()
        ]
    )
    
    Config.dataset = 'porto'
    Config.post_value_updates()
    
    logging.info('=== Porto Data Preprocessing ===')
    logging.info('Config: dataset={}, prefix={}'.format(Config.dataset, Config.dataset_prefix))
    logging.info('Bounds: lon=[{}, {}], lat=[{}, {}]'.format(
        Config.min_lon, Config.max_lon, Config.min_lat, Config.max_lat))
    
    # Step 1: 转换训练数据
    logging.info('\n=== Step 1: Converting training data ===')
    convert_porto_data()
    
    # Step 2: 转换测试数据
    logging.info('\n=== Step 2: Converting test data ===')
    convert_porto_test_data()
    
    # Step 3: 初始化网格空间和嵌入
    logging.info('\n=== Step 3: Initializing cellspace and embeddings ===')
    init_cellspace()
    
    # Step 4: 生成相似性测试数据集（已跳过）
    # logging.info('\n=== Step 4: Generating similarity test dataset ===')
    # generate_newsimi_test_dataset()
    
    logging.info('\n=== All preprocessing completed! ===')

