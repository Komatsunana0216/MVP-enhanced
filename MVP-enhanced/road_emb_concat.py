import torch
import math
def get_road(df):
    road_list = []
    df['cpath_list'].apply(lambda row: road_list.extend(row))
    return list(set(road_list))
def get_road_emb_from_concat(seq_model, test_data, without_gps=False, batch_size=1024, update_road='mean', city='chengdu'):
    assert update_road in ['mean', 'weight', 'route'], 'update_road must be one of [\'mean\', \'weight\', \'route\']'

    route_data, masked_route_assign_mat, gps_data, masked_gps_assign_mat, route_assign_mat, gps_length, dataset = test_data

    if without_gps:
        gps_data = torch.zeros_like(gps_data) # 无实际意义，用于补位
        masked_gps_assign_mat = torch.zeros_like(masked_gps_assign_mat) # 无实际意义，用于补位
        gps_length = torch.ones_like(gps_length) # 无实际意义，用于补位

    # 都放到显存里面放不下，需要分batch处理
    max_len = dataset['route_length'].max()
    with torch.no_grad():
        gps_road_joint_rep_list, route_road_joint_rep_list = [], []
        # print("这个数是多少",route_data.shape[0] // batch_size + 1)
        # print("被除数",route_data.shape[0])
        # print("除数",batch_size+1)
        for i in range(math.ceil(route_data.shape[0] / batch_size)):  # 最后不足batch_size的case不要了
            # print("进行到batch",i)

            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if end_idx > route_data.shape[0]:
                end_idx = None
            batch_route_data = route_data[start_idx:end_idx].cuda()
            batch_masked_route_assign_mat = masked_route_assign_mat[start_idx:end_idx].cuda()
            batch_gps_data = gps_data[start_idx:end_idx].cuda()
            batch_masked_gps_assign_mat = masked_gps_assign_mat[start_idx:end_idx].cuda()
            batch_route_assign_mat = route_assign_mat[start_idx:end_idx].cuda()
            # if i == 3029:
            #     print("batch_gps_data",batch_gps_data)
            batch_gps_length = gps_length[start_idx:end_idx].cuda()
            # _, _, _, _, gps_road_joint_rep, _, route_road_joint_rep, _ ,_,_\
            #     = seq_model(batch_route_data, batch_masked_route_assign_mat, batch_gps_data,
            #                 batch_masked_gps_assign_mat, batch_route_assign_mat, batch_gps_length)
            # _, _, _, _, gps_road_joint_rep, _, route_road_joint_rep, _ ,_\
            #     = seq_model(batch_route_data, batch_masked_route_assign_mat, batch_gps_data,
            #                 batch_masked_gps_assign_mat, batch_route_assign_mat, batch_gps_length)
            # 模型返回9个值，最后一个是loss1
            _, _, _, _, gps_road_joint_rep, _, route_road_joint_rep, _, _ \
                = seq_model(batch_route_data, batch_masked_route_assign_mat, batch_gps_data,
                            batch_masked_gps_assign_mat, batch_route_assign_mat, batch_gps_length)

            # _, _, _, _, gps_road_joint_rep, _, route_road_joint_rep, _, _,_,_ \
            #     = seq_model(batch_route_data, batch_masked_route_assign_mat, batch_gps_data,
            #                 batch_masked_gps_assign_mat, batch_route_assign_mat, batch_gps_length)

            del batch_route_data, batch_masked_route_assign_mat, batch_gps_data, batch_masked_gps_assign_mat, batch_route_assign_mat, batch_gps_length
            padding = torch.zeros(
                (
                gps_road_joint_rep.shape[0], max_len - gps_road_joint_rep.shape[1], gps_road_joint_rep.shape[2])).cuda()

            gps_road_joint_rep = torch.cat([gps_road_joint_rep, padding], dim=1).cpu()
            route_road_joint_rep = torch.cat([route_road_joint_rep, padding], dim=1).cpu()

            gps_road_joint_rep_list.append(gps_road_joint_rep)
            route_road_joint_rep_list.append(route_road_joint_rep)

            del gps_road_joint_rep, route_road_joint_rep
            # torch.cuda.empty_cache() # 清空显存

    gps_road_joint_rep = torch.cat(gps_road_joint_rep_list, dim=0).cpu()  # 注意gps_road_joint_rep_list中不能为空
    #todo 这写错了，改一下

    route_road_joint_rep = torch.cat(route_road_joint_rep_list, dim=0).cpu()

    road_list = get_road(dataset)
    print('number of roads observed: {}'.format(len(road_list)))

    if update_road == 'mean':
        road_embedding = get_road_embedding1(road_list, gps_road_joint_rep, route_road_joint_rep, route_assign_mat,
                                             seq_model)
    elif update_road == 'weight':
        road_embedding = get_road_embedding2(road_list, gps_road_joint_rep, route_road_joint_rep, route_assign_mat,
                                             seq_model, city)
    elif update_road == 'route':
        road_embedding = get_road_embedding_concat(road_list, gps_road_joint_rep, route_road_joint_rep, route_assign_mat,
                                             seq_model)
    return road_embedding,road_list

def get_road_embedding_concat(road_list, gps_road_joint_rep, route_road_joint_rep, route_assign_mat, seq_model):
    # 处理在训练时被观测到的路段
    road_joint_rep = torch.cat([gps_road_joint_rep.unsqueeze(2), route_road_joint_rep.unsqueeze(2)], dim=-1)
    # print("road_joint_rep的形状",road_joint_rep.shape)
    road_embedding = torch.zeros((seq_model.node_embedding.weight.shape[0], road_joint_rep.shape[-1])).cuda()
    route_assign_mat = route_assign_mat[:road_joint_rep.shape[0]]
    for road_id in road_list:
        indexes = torch.nonzero(route_assign_mat==road_id)
        rep_list = [road_joint_rep[index[0]][index[1]] for index in indexes]
        road_rep = torch.cat(rep_list, dim=0)
        # print("road_rep的形状", road_rep.shape)
        road_rep = torch.mean(road_rep, dim=0)

        road_embedding[road_id] = road_rep.cuda()

    # # 处理在训练时被未被观测到的路段
    # indexes = torch.nonzero(torch.sum(road_embedding, dim=1) == 0)
    # unseen_route_assign_mat = indexes
    # print("unseen_route_assign_mat",unseen_route_assign_mat)
    # # road_rep, _ ,_,_= seq_model.encode_route(None, unseen_route_assign_mat, unseen_route_assign_mat)
    # road_rep, _,_ = seq_model.encode_route(None, unseen_route_assign_mat, unseen_route_assign_mat)
    # # road_rep, _= seq_model.encode_route(None, unseen_route_assign_mat, unseen_route_assign_mat)
    # for i, index in enumerate(indexes):
    #     road_embedding[index.item()] = road_rep[i]

    return road_embedding.detach()