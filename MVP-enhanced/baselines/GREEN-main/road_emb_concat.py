import torch
import math
from config.config import Config
def get_road_emb_from_concat(seq_model, test_loader,road_feature,road_graph,grid_image,road_list,max_len, type_mask,device, batch_size=1024,  city='chengdu'):






    # 都放到显存里面放不下，需要分batch处理


    with torch.no_grad():
        rep_list = []
        route_assign_mat=[]
        for batch_data in test_loader:
            road_data, grid_data= batch_data
            road_data['g_input_feature'] = road_feature
            road_data['g_edge_index'] = road_graph
            grid_data['grid_image'] = grid_image
            if type_mask==True:
                road_data['road_type']=torch.ones_like(road_data['road_type'])
            for k, v in road_data.items():
                if k == 'mask_road_index' or k=='road_lens':
                    continue
                # print(k)
                road_data[k] = v.to(device)
            for k, v in grid_data.items():
                grid_data[k] = v.to(device)

            _,_,_,road_emb=seq_model.roadseg_test(grid_data,road_data)
            padding = torch.zeros(
                (
                    road_emb.shape[0], max_len - road_emb.shape[1],
                    road_emb.shape[2])).cuda()
            route_assign_padding=torch.full((road_emb.shape[0], max_len - road_data['road_traj'].shape[1]),
                       Config.road_special_tokens['padding_token'], dtype=torch.long).cuda()
            traj_input=torch.cat([road_data['road_traj'], route_assign_padding], dim=1).cpu()
            route_assign_mat.append(traj_input)
            road_rep = torch.cat([road_emb, padding], dim=1).cpu()
            rep_list.append(road_rep)
        all_road_rep = torch.cat(rep_list, dim=0).cpu()
        route_assign_mat = torch.cat(route_assign_mat, dim=0).cpu()
        print('number of roads observed: {}'.format(len(road_list)))
        road_embedding = get_road_embedding_concat(road_list, all_road_rep,
                                                       route_assign_mat,
                                                       road_feature)
        return road_embedding

def get_road_embedding_concat(road_list, all_road_rep, route_assign_mat, road_feature):
    # 处理在训练时被观测到的路段

    road_embedding = torch.zeros((road_feature.shape[0], all_road_rep.shape[-1])).cuda()

    for road_id in road_list:
        indexes = torch.nonzero(route_assign_mat==road_id)
        rep_list = [all_road_rep[index[0]][index[1]] for index in indexes]
        road_rep = torch.cat(rep_list, dim=0)
        # print("road_rep的形状", road_rep.shape)
        road_rep = torch.mean(road_rep, dim=0)

        road_embedding[road_id] = road_rep.cuda()



    return road_embedding.detach()