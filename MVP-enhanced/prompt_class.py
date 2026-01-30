import torch.nn as nn
import torch
class IntervalEmbedding(nn.Module):
    def __init__(self, num_bins, hidden_size):
        super(IntervalEmbedding, self).__init__()
        self.layer1 = nn.Linear(1, num_bins)
        self.emb = nn.Embedding(num_bins, hidden_size)
        self.activation = nn.Softmax()
    def forward(self, x):
        logit = self.activation(self.layer1(x.unsqueeze(-1)))
        output = logit @ self.emb.weight
        return output




class prompt(nn.Module):
    """ miltiscale convolutional kernels
    """
    def __init__(self, vocab_size,edge_features,length_mean,length_std,route_embed_size):
        super().__init__()
        self.vocab_size=vocab_size
        self.edge_features = edge_features
        self.length_mean = length_mean
        self.length_std = length_std
        self.mask_cls=False
        self.highway_embedding = nn.Embedding(15, 31, padding_idx=0)
        self.lanes_embedding = nn.Embedding(10, 31, padding_idx=0)
        self.indegree_embedding = nn.Embedding(8, 31, padding_idx=0)
        self.outdegree_embedding = nn.Embedding(8, 31, padding_idx=0)
        self.rep = nn.Linear(125, route_embed_size)
        encdoer_layer = nn.TransformerEncoderLayer(d_model=route_embed_size, nhead=4, dim_feedforward=route_embed_size,
                                                   batch_first=True)
        self.ef_encoder = nn.TransformerEncoder(encoder_layer=encdoer_layer, num_layers=1)

        self.prompt_minute_embedding = nn.Embedding(1440 + 1, route_embed_size)  # 0 是mask位
        self.prompt_week_embedding = nn.Embedding(7 + 1, route_embed_size)  # 0 是mask位
        self.prompt_delta_embedding = IntervalEmbedding(100, route_embed_size)
        self.timerep = nn.Linear(route_embed_size, route_embed_size)
        self.time_encoder = nn.TransformerEncoder(encoder_layer=encdoer_layer, num_layers=1)



    def forward(self, route_assign_mat,route_data,padding_mask):
    # def forward(self, route_assign_mat, route_data):
        highway_dict = {'living_street': 1, 'motorway': 2, 'motorway_link': 3, 'planned': 4, 'trunk': 5,
                        "secondary": 6, "trunk_link": 7, "tertiary_link": 8, "primary": 9, "residential": 10,
                        "primary_link": 11, "unclassified": 12, "tertiary": 13, "secondary_link": 14}

        infos_tensor = []
        for traj in route_assign_mat:
            infos = []
            for road_seg in traj:
                # print("road_seg是什么",road_seg)
                if road_seg.item() == self.vocab_size:
                    info = [0, 0, 0, 0, 0]
                else:
                    road_features = self.edge_features[self.edge_features['fid'] == road_seg.item()]
                    # print("看看这里拿到的是什么",road_features['highway'].iloc[0])

                    highway_raw = road_features['highway'].iloc[0]
                    highway = highway_dict.get(highway_raw, 0)  # 不认识的类型直接跳过，设为0

                    length = road_features['length'].iloc[0]
                    lanes = road_features['lanes'].iloc[0] if road_features['lanes'].iloc[0] != None else 0
                    # if lanes>=7:
                    #     lanes=7
                    indegree = road_features['indegree'].iloc[0]
                    outdegree = road_features['outdegree'].iloc[0]
                    info = [highway, length, lanes, indegree, outdegree]
                infos.append(info)
            infos_tensor.append(infos)
        infos_tensor = torch.tensor(infos_tensor).cuda()
        infos_tensor = torch.where(torch.isnan(infos_tensor), torch.full_like(infos_tensor, 0), infos_tensor).cuda()
        # print("infos_tensor",infos_tensor)
        # print("形状",infos_tensor.shape)
        highway_data = infos_tensor[:, :, 0].long()
        length_data = infos_tensor[:, :, 1].long()
        length_data = (length_data - self.length_mean) / self.length_std
        lanes_data = infos_tensor[:, :, 2].long()
        indegree_data = infos_tensor[:, :, 3].long()
        outdegree_data = infos_tensor[:, :, 4].long()
        if self.mask_cls:
            test_emb = torch.stack([self.highway_embedding.weight.detach()[9],self.highway_embedding.weight.detach()[6],self.highway_embedding.weight.detach()[13],self.highway_embedding.weight.detach()[10]],dim=0)
            # print("test_emb形状",test_emb.shape)
            highway_emb=test_emb.mean(dim=0)

            highway_emb = highway_emb.repeat(route_assign_mat.shape[0],route_assign_mat.shape[1] , 1)
        else:
            highway_emb = self.highway_embedding(highway_data)
        # print("highway_emb形状",highway_emb.shape)
        lanes_emb = self.lanes_embedding(lanes_data)

        # print("lanes_emb形状", lanes_emb.shape)
        indegree_emb = self.indegree_embedding(indegree_data)
        outdegree_emb = self.outdegree_embedding(outdegree_data)
        # print("highway_emb形状",highway_emb.shape)
        # print("length_data形状",length_data.shape)
        length_data=length_data.unsqueeze(-1)
        edge_input = self.rep(torch.concat([length_data, highway_emb, lanes_emb, indegree_emb, outdegree_emb], dim=-1))
        edge_prompt = self.ef_encoder(edge_input,src_key_padding_mask=padding_mask)
        # edge_prompt = self.ef_encoder(edge_input)

        if route_data is None:
            return edge_prompt
        else:
            week_data = route_data[:, :, 0].long()
            min_data = route_data[:, :, 1].long()
            delta_data = route_data[:, :, 2].float()
            week_emb = self.prompt_week_embedding(week_data)
            min_emb = self.prompt_minute_embedding(min_data)
            delta_emb = self.prompt_delta_embedding(delta_data)
            time_input = self.timerep(week_emb + min_emb + delta_emb)
            time_prompt = self.time_encoder(time_input,src_key_padding_mask=padding_mask)
            # time_prompt = self.time_encoder(time_input)
            return edge_prompt + time_prompt