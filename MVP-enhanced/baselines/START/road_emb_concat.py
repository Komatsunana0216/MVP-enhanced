import torch
import math

def get_road_emb_from_concat(downstream_model, dataloader,graph_dict,vocab,device, batch_size=1024,  city='chengdu', num_roads=None):






    # 都放到显存里面放不下，需要分batch处理

    downstream_model.eval()
    with torch.no_grad():
        rep_list = []
        route_assign_mat=[]
        road_list = []
        i=0
        for batch in dataloader:
            i+=1
            print("进入第几个epoch",i)
            X, _, padding_masks, batch_temporal_mat = batch
            X = X.to(device)

            padding_masks = padding_masks.to(device)  # 0s: ignore
            batch_temporal_mat = batch_temporal_mat.to(device)
            # 获取 token 级别的 embedding（通过 bert 模型）
            road_emb, _, _ = downstream_model.bert(x=X, padding_masks=padding_masks, batch_temporal_mat=batch_temporal_mat,
                                     graph_dict=graph_dict)
            batch_input=[]
            X = X.cpu()
            # 使用实际路段数量作为 padding 占位符
            pad_token = num_roads if num_roads else vocab.vocab_size
            for j in range(X.shape[0]):
                temp=[
                    pad_token if vocab.index2loc[index] == '<pad>' else vocab.index2loc[index]
                    for index in X[j, 1:, 0]
                ]
                batch_input.append(temp)
                road_list.extend([item for item in temp if item != pad_token])


            route_assign_mat.append(torch.LongTensor(batch_input))
            rep_list.append(road_emb.cpu())
            del road_emb,X,batch_temporal_mat,padding_masks,batch,temp
            torch.cuda.empty_cache()
        all_road_rep = torch.cat(rep_list, dim=0).cpu()
        route_assign_mat = torch.cat(route_assign_mat, dim=0).cpu()
        road_list_set=set(road_list)

        road_list=list(road_list_set)
        print('number of roads observed: {}'.format(len(road_list)))
        # 使用实际观察到的最大路段 ID + 1 来确定矩阵大小
        max_road_id = max(road_list) if road_list else 0
        actual_num_roads = max(max_road_id + 1, num_roads if num_roads else road_feature.shape[0])
        print(f'max road ID: {max_road_id}, embedding matrix size: {actual_num_roads}')
        road_embedding = get_road_embedding_concat(road_list, all_road_rep,
                                                       route_assign_mat,
                                                       graph_dict['node_features'],
                                                       num_roads=actual_num_roads)
        return road_embedding,road_list

def get_road_embedding_concat(road_list, all_road_rep, route_assign_mat, road_feature, num_roads=None):
    # 处理在训练时被观测到的路段
    # 使用实际路段数量，如果未指定则使用 road_feature 的大小
    if num_roads is None:
        num_roads = road_feature.shape[0]
    road_embedding = torch.zeros((num_roads, all_road_rep.shape[-1])).cuda()
    print("road_embedding形状",road_embedding.shape)
    for road_id in road_list:
        indexes = torch.nonzero(route_assign_mat == road_id)
        rep_list = [all_road_rep[index[0]][index[1]] for index in indexes]
        road_rep = torch.cat(rep_list, dim=0)
        # print("road_rep的形状", road_rep.shape)
        road_rep = torch.mean(road_rep, dim=0)

        road_embedding[road_id] = road_rep.cuda()

    return road_embedding.detach()






#
#             _,_,_,road_emb=seq_model.roadseg_test(grid_data,road_data)
#             padding = torch.zeros(
#                 (
#                     road_emb.shape[0], max_len - road_emb.shape[1],
#                     road_emb.shape[2])).cuda()
#             route_assign_padding=torch.full((road_emb.shape[0], max_len - road_data['road_traj'].shape[1]),
#                        Config.road_special_tokens['padding_token'], dtype=torch.long).cuda()
#             traj_input=torch.cat([road_data['road_traj'], route_assign_padding], dim=1).cpu()
#             route_assign_mat.append(traj_input)
#             road_rep = torch.cat([road_emb, padding], dim=1).cpu()
#             rep_list.append(road_rep)
#         all_road_rep = torch.cat(rep_list, dim=0).cpu()
#         route_assign_mat = torch.cat(route_assign_mat, dim=0).cpu()
#         print('number of roads observed: {}'.format(len(road_list)))
#         road_embedding = get_road_embedding_concat(road_list, all_road_rep,
#                                                        route_assign_mat,
#                                                        road_feature)
#         return road_embedding
