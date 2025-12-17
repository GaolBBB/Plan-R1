import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse
from typing import Dict

from layers import GraphAttention
from layers import TwoLayerMLP
from utils import compute_angles_lengths_2D
from utils import init_weights
from utils import wrap_angle
from utils import drop_edge_between_samples
from utils import transform_point_to_local_coordinate
from utils import move_dict_to_device
import ipdb
class Backbone(nn.Module):
    def __init__(self,
                 token_dict: Dict,
                 num_tokens: int,
                 interval: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 agent_radius: float,
                 polygon_radius: float,
                 num_attn_layers: int, 
                 num_heads: int,
                 dropout: float) -> None:
        super(Backbone, self).__init__()
        self.token_dict = token_dict
        self.num_tokens = num_tokens
        self.interval = interval
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_historical_intervals = num_historical_steps // interval
        self.num_intervals = (num_future_steps + num_historical_steps) // interval
        self.agent_radius = agent_radius
        self.polygon_radius = polygon_radius
        self.num_attn_layers = num_attn_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self._agent_types = ['Vehicle', 'Pedestrian', 'Bicycle']
        self._agent_type_embs = nn.Embedding(len(self._agent_types), hidden_dim) # self._agent_type_embs(a_type) 就是 把“类别 ID → hidden_dim 维向量”。
        self._identity_types = ['Ego', 'Agent']
        self._identity_type_embs = nn.Embedding(len(self._identity_types), hidden_dim)

        self.agent_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.token_emb_vehicle = nn.Embedding(num_tokens, hidden_dim)
        self.token_emb_pedestrian = nn.Embedding(num_tokens, hidden_dim)
        self.token_emb_bicycle = nn.Embedding(num_tokens, hidden_dim)

        self.fusion_layer = TwoLayerMLP(input_dim=hidden_dim * 2, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.k2k_t_emb_layer = TwoLayerMLP(input_dim=6, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.g2k_emb_layer = TwoLayerMLP(input_dim=6, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.k2k_a_emb_layer = TwoLayerMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.k2k_t_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])
        self.g2k_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=False) for _ in range(num_attn_layers)])
        self.k2k_a_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout, has_edge_attr=True, if_self_attention=True) for _ in range(num_attn_layers)])

        self.apply(init_weights)

    def forward(self, data: Batch, g_embs: torch.Tensor) -> torch.Tensor: # 用在训练阶段（把完整序列一起送进去）
        # ipdb.set_trace()

        # agent embedding
        a_box = data['agent']['box']                                     #[(N1,...,Nb),4]
        a_type = data['agent']['type'].long()
        a_identity = data['agent']['identity'].long()
        a_embs = self.agent_emb_layer(input=a_box) + self._agent_type_embs(a_type) + self._identity_type_embs(a_identity)    #[(N1,...,Nb),D]  a_embs 包含的是 “这个 agent 是谁”:来自 box：尺寸/长宽等几何信息,来自 _agent_type_embs(a_type)：车 / 行人 / 自行车 类型 embedding,来自 _identity_type_embs(a_identity)：Ego / 其他 agent 的身份 embedding

        num_agents, num_intervals, _ = data['agent']['recon_position'].size()   #[(N1,...,Nb),T,2]
        device = data['agent']['recon_position'].device

        vehicle_mask = a_type == 0 # 不是 “0、1、2 这几个数字”，而是一个 布尔向量（bool mask），内容都是true/false [139]
        pedestrian_mask = a_type == 1 # [139]
        bicycle_mask = a_type == 2 # [139]
        a_token_embs = torch.zeros(num_agents, num_intervals, self.hidden_dim, device=device) 
        a_token_embs[vehicle_mask] = self.token_emb_vehicle(data['agent']['recon_token'][vehicle_mask]) # 把 vehicle_mask 视为「对 data['agent']['recon_token'] 第0维的布尔选择。data['agent']['recon_token'][vehicle_mask]输出形状为[80.20],80是选出来的 vehicle 数量，20是时间维度 T。然后通过 self.token_emb_vehicle 把 token ID 转成 embedding，最后赋值给 a_token_embs 对应位置。
        a_token_embs[pedestrian_mask] = self.token_emb_pedestrian(data['agent']['recon_token'][pedestrian_mask])
        a_token_embs[bicycle_mask] = self.token_emb_bicycle(data['agent']['recon_token'][bicycle_mask])

        k_embs = self.fusion_layer(torch.cat([a_embs.unsqueeze(1).expand(-1, num_intervals, -1), a_token_embs], dim=-1)).reshape(-1, self.hidden_dim)   #[(N1,...,Nb)*T,D] 先将每个agent的静态身份扩充到时间维度，再和每个agent在每个时间片上的token运动特征做对齐融合，再通过2层MLP映射到隐藏层维度，最后reshape成[(N1,...,Nb)*T,D]形状。

        # edge embedding
        # k2k_t
        # Temporal self-attention
        k2k_t_position = data['agent']['recon_position'].reshape(-1, 2)     #[(N1,...,Nb)*T,2]
        k2k_t_heading = data['agent']['recon_heading'].reshape(-1)          #[(N1,...,Nb)*T]
        k2k_t_valid_mask = data['agent']['recon_valid_mask'].clone()        #[(N1,...,Nb),T]
        k2k_t_valid_mask = k2k_t_valid_mask.unsqueeze(2) & k2k_t_valid_mask.unsqueeze(1)  #[(N1,...,Nb),T,T]
        k2k_t_edge_index = dense_to_sparse(k2k_t_valid_mask)[0]
        k2k_t_edge_index = k2k_t_edge_index[:, k2k_t_edge_index[0] <= k2k_t_edge_index[1]]
        k2k_t_edge_index = k2k_t_edge_index[:, k2k_t_edge_index[1] - k2k_t_edge_index[0] <= 6]
        k2k_t_edge_vector = transform_point_to_local_coordinate(k2k_t_position[k2k_t_edge_index[0]], k2k_t_position[k2k_t_edge_index[1]], k2k_t_heading[k2k_t_edge_index[1]])
        k2k_t_edge_attr_length, k2k_t_edge_attr_theta = compute_angles_lengths_2D(k2k_t_edge_vector)
        k2k_t_edge_attr_heading = wrap_angle(k2k_t_heading[k2k_t_edge_index[0]] - k2k_t_heading[k2k_t_edge_index[1]])
        k2k_t_edge_attr_interval = (k2k_t_edge_index[0] - k2k_t_edge_index[1]).float() * self.interval
        k2k_t_edge_attr_input = torch.stack([k2k_t_edge_attr_length, torch.cos(k2k_t_edge_attr_theta), torch.sin(k2k_t_edge_attr_theta), torch.cos(k2k_t_edge_attr_heading), torch.sin(k2k_t_edge_attr_heading), k2k_t_edge_attr_interval], dim=-1)
        k2k_t_edge_attr_embs = self.k2k_t_emb_layer(input=k2k_t_edge_attr_input)

        # g2k
        # Agent-Map cross-attention
        g2k_position_g = data['polygon']['position']                                    #[(M1,...,Mb),2]
        g2k_position_k = data['agent']['recon_position'].reshape(-1, 2)                 #[(N1,...,Nb)*T,2]
        g2k_heading_g = data['polygon']['heading']                                      #[(M1,...,Mb)]
        g2k_heading_k = data['agent']['recon_heading'].reshape(-1)                      #[(N1,...,Nb)*T]
        g2k_batch_g = data['polygon']['batch']                                          #[(M1,...,Mb)]
        g2k_batch_k = data['agent']['batch'].repeat_interleave(num_intervals)           #[(N1,...,Nb)*T]
        g2k_valid_mask = data['agent']['recon_valid_mask'].reshape(-1).unsqueeze(0).expand(data['polygon']['position'].size(0), -1)    #[(M1,...,Mb), (N1,...,Nb)*T]
        g2k_valid_mask = drop_edge_between_samples(g2k_valid_mask, batch=(g2k_batch_g, g2k_batch_k))   #[(M1,...,Mb), (N1,...,Nb)*T]
        g2k_edge_index = dense_to_sparse(g2k_valid_mask)[0]
        g2k_edge_index = g2k_edge_index[:, torch.norm(g2k_position_g[g2k_edge_index[0]] - g2k_position_k[g2k_edge_index[1]], p=2, dim=-1) < self.polygon_radius]
        g2k_edge_vector = transform_point_to_local_coordinate(g2k_position_g[g2k_edge_index[0]], g2k_position_k[g2k_edge_index[1]], g2k_heading_k[g2k_edge_index[1]])
        g2k_edge_attr_length, g2k_edge_attr_theta = compute_angles_lengths_2D(g2k_edge_vector)
        g2k_edge_attr_heading_valid_mask = data['polygon']['heading_valid_mask'][g2k_edge_index[0]]
        g2k_edge_attr_heading = wrap_angle(g2k_heading_g[g2k_edge_index[0]] - g2k_heading_k[g2k_edge_index[1]])
        g2k_edge_attr_input = torch.stack([g2k_edge_attr_length, torch.cos(g2k_edge_attr_theta), torch.sin(g2k_edge_attr_theta), torch.cos(g2k_edge_attr_heading), torch.sin(g2k_edge_attr_heading), g2k_edge_attr_heading_valid_mask], dim=-1)
        g2k_edge_attr_embs = self.g2k_emb_layer(input=g2k_edge_attr_input)

        # k2k_a
        # Agent-Agent self-attention
        k2k_a_position = data['agent']['recon_position'].transpose(0, 1).reshape(-1, 2)     #[T*(N1,...,Nb),2]
        k2k_a_heading = data['agent']['recon_heading'].transpose(0, 1).reshape(-1)          #[T*(N1,...,Nb)]
        k2k_a_valid_mask = data['agent']['recon_valid_mask'].transpose(0, 1)                #[T, (N1,...,Nb)]
        k2k_a_valid_mask = k2k_a_valid_mask.unsqueeze(2) & k2k_a_valid_mask.unsqueeze(1)    #[T, (N1,...,Nb), (N1,...,Nb)]
        k2k_a_valid_mask = drop_edge_between_samples(k2k_a_valid_mask, batch=data['agent']['batch'])    #[T, (N1,...,Nb), (N1,...,Nb)]
        k2k_a_edge_index = dense_to_sparse(k2k_a_valid_mask)[0]
        k2k_a_edge_index = k2k_a_edge_index[:, torch.norm(k2k_a_position[k2k_a_edge_index[0]] - k2k_a_position[k2k_a_edge_index[1]], p=2, dim=-1) < self.agent_radius]
        k2k_a_edge_vector = transform_point_to_local_coordinate(k2k_a_position[k2k_a_edge_index[0]], k2k_a_position[k2k_a_edge_index[1]], k2k_a_heading[k2k_a_edge_index[1]])
        k2k_a_edge_attr_length, k2k_a_edge_attr_theta = compute_angles_lengths_2D(k2k_a_edge_vector)
        k2k_a_edge_attr_heading = wrap_angle(k2k_a_heading[k2k_a_edge_index[0]] - k2k_a_heading[k2k_a_edge_index[1]])
        k2k_a_edge_attr_input = torch.stack([k2k_a_edge_attr_length, torch.cos(k2k_a_edge_attr_theta), torch.sin(k2k_a_edge_attr_theta), torch.cos(k2k_a_edge_attr_heading), torch.sin(k2k_a_edge_attr_heading)], dim=-1)
        k2k_a_edge_attr_embs = self.k2k_a_emb_layer(input=k2k_a_edge_attr_input)

        # attention
        # ipdb.set_trace()
        for i in range(self.num_attn_layers):
            # k2k_t
            k_embs = self.k2k_t_attn_layers[i](x=k_embs, edge_index=k2k_t_edge_index, edge_attr=k2k_t_edge_attr_embs)
            # g2k
            k_embs = self.g2k_attn_layers[i](x=[g_embs, k_embs], edge_index=g2k_edge_index, edge_attr=g2k_edge_attr_embs)
            # k2k_a
            k_embs = k_embs.reshape(-1, num_intervals, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)   #[T*(N1,...,Nb), D]
            k_embs = self.k2k_a_attn_layers[i](x=k_embs, edge_index=k2k_a_edge_index, edge_attr=k2k_a_edge_attr_embs)
            k_embs = k_embs.reshape(num_intervals, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)   #[(N1,...,Nb)*T, D]
        k_embs = k_embs.reshape(-1, num_intervals, self.hidden_dim)  #[(N1,...,Nb), T, D]
        # ipdb.set_trace()
        return k_embs
        '''
        对于 batch 中每一个 agent、在 token 时间轴上的每一个时间片 t，给它一个 hidden_dim 维的向量，这个向量已经融合了：
        •	它自己的静态信息（类型 / 尺寸 / ego 等），
        •	它在这个时间片上的 motion token 信息，
        •	它在整个时间维度上的演化上下文（k2k_t），
        •	它与周围地图（polygon）的关系（g2k），
        •	它与其他 agent 在每个时间片的交互（k2k_a）。
        '''
    def pre_inference(self, data: Batch) -> torch.Tensor: # 推理开始时，用已有历史 token 初始化 k_embs_dict
        # ipdb.set_trace()
        device = data['agent']['infer_position'].device
        self.token_dict = move_dict_to_device(self.token_dict, device)

        # agent_encoding
        a_box = data['agent']['box']                                     #[(N1,...,Nb),4]
        a_type = data['agent']['type'].long()
        a_identity = data['agent']['identity'].long()
        a_embs = self.agent_emb_layer(input=a_box) + self._agent_type_embs(a_type) + self._identity_type_embs(a_identity)    #[(N1,...,Nb),D]

        num_agents, num_intervals, _ = data['agent']['infer_position'].size()   #[(N1,...,Nb),T,2]
        vehicle_mask = a_type == 0
        pedestrian_mask = a_type == 1
        bicycle_mask = a_type == 2
        a_token_embs = torch.zeros(num_agents, num_intervals, self.hidden_dim, device=a_embs.device)
        a_token_embs[vehicle_mask] = self.token_emb_vehicle(data['agent']['infer_token'][vehicle_mask])
        a_token_embs[pedestrian_mask] = self.token_emb_pedestrian(data['agent']['infer_token'][pedestrian_mask])
        a_token_embs[bicycle_mask] = self.token_emb_bicycle(data['agent']['infer_token'][bicycle_mask])

        k_embs = self.fusion_layer(torch.cat([a_embs.unsqueeze(1).expand(-1, num_intervals, -1), a_token_embs], dim=-1))   #[(N1,...,Nb),T,D]
        k_embs_dict = {}
        k_embs_dict[0] = k_embs
        return a_embs, k_embs_dict

    def inference(self, data: Batch, g_embs: torch.Tensor, a_embs: torch.Tensor, k_embs_dict: Dict) -> torch.Tensor: # 每次只扩展一步，更新最后一帧 embedding，配合 token 解码器做自回归 roll-out。
        # ipdb.set_trace()
        device = data['agent']['position'].device
        num_agents, num_intervals, _ = data['agent']['infer_position'].size()   #[(N1,...,Nb),T,2]
        self.token_dict = move_dict_to_device(self.token_dict, device)

        if num_intervals == self.num_historical_intervals:
            # the first step of inference
            inference_mask = data['agent']['infer_valid_mask']              #[(N1,...,Nb),T]
        else:
            # other steps of inference
            inference_mask = torch.zeros_like(data['agent']['infer_valid_mask'], device=device, dtype=torch.bool)   #[(N1,...,Nb),T]
            inference_mask[:, -1] = data['agent']['infer_valid_mask'][:, -1]   #[(N1,...,Nb),1]

            a_type = data['agent']['type'].long()
            vehicle_mask = a_type == 0
            pedestrian_mask = a_type == 1
            bicycle_mask = a_type == 2
            a_token_embs = torch.zeros(num_agents, self.hidden_dim, device=a_embs.device)
            a_token_embs[vehicle_mask] = self.token_emb_vehicle(data['agent']['infer_token'][vehicle_mask, -1])
            a_token_embs[pedestrian_mask] = self.token_emb_pedestrian(data['agent']['infer_token'][pedestrian_mask, -1])
            a_token_embs[bicycle_mask] = self.token_emb_bicycle(data['agent']['infer_token'][bicycle_mask, -1])

            k_embs_cur = self.fusion_layer(torch.cat([a_embs, a_token_embs], dim=-1))   #[(N1,...,Nb),D]
            k_embs_dict[0] = torch.cat([k_embs_dict[0], k_embs_cur.unsqueeze(1)], dim=1)   #[(N1,...,Nb),t,D]

        # Interaction
        infer_position = data['agent']['infer_position']    #[(N1,...,Nb),t,2]
        infer_heading = data['agent']['infer_heading']      #[(N1,...,Nb),t]
        infer_valid_mask = data['agent']['infer_valid_mask']    #[(N1,...,Nb),t]

        # edge embedding
        # k2k_t
        k2k_t_position = infer_position.reshape(-1, 2)     #[(N1,...,Nb)*t,2]
        k2k_t_heading = infer_heading.reshape(-1)          #[(N1,...,Nb)*t]
        k2k_t_valid_mask = infer_valid_mask.unsqueeze(2) & inference_mask.unsqueeze(1)  #[(N1,...,Nb),t,t]
        k2k_t_edge_index = dense_to_sparse(k2k_t_valid_mask)[0]
        k2k_t_edge_index = k2k_t_edge_index[:, k2k_t_edge_index[0] <= k2k_t_edge_index[1]]
        k2k_t_edge_index = k2k_t_edge_index[:, k2k_t_edge_index[1] - k2k_t_edge_index[0] <= 6]
        k2k_t_edge_vector = transform_point_to_local_coordinate(k2k_t_position[k2k_t_edge_index[0]], k2k_t_position[k2k_t_edge_index[1]], k2k_t_heading[k2k_t_edge_index[1]])
        k2k_t_edge_attr_length, k2k_t_edge_attr_theta = compute_angles_lengths_2D(k2k_t_edge_vector)
        k2k_t_edge_attr_heading = wrap_angle(k2k_t_heading[k2k_t_edge_index[0]] - k2k_t_heading[k2k_t_edge_index[1]])
        k2k_t_edge_attr_interval = (k2k_t_edge_index[0] - k2k_t_edge_index[1]).float() * self.interval
        k2k_t_edge_attr_input = torch.stack([k2k_t_edge_attr_length, torch.cos(k2k_t_edge_attr_theta), torch.sin(k2k_t_edge_attr_theta), torch.cos(k2k_t_edge_attr_heading), torch.sin(k2k_t_edge_attr_heading), k2k_t_edge_attr_interval], dim=-1)
        k2k_t_edge_attr_embs = self.k2k_t_emb_layer(input=k2k_t_edge_attr_input)

        # g2k
        g2k_position_g = data['polygon']['position']                                    #[(M1,...,Mb),2]
        g2k_position_k = infer_position.reshape(-1, 2)                                  #[(N1,...,Nb)*t,2]
        g2k_heading_g = data['polygon']['heading']                                      #[(M1,...,Mb)]
        g2k_heading_k = infer_heading.reshape(-1)                                       #[(N1,...,Nb)*t]
        g2k_batch_g = data['polygon']['batch']                                          #[(M1,...,Mb)]
        g2k_batch_k = data['agent']['batch'].repeat_interleave(num_intervals)           #[(N1,...,Nb)*t]
        g2k_valid_mask = inference_mask.reshape(-1).unsqueeze(0).expand(data['polygon']['position'].size(0), -1)    #[(M1,...,Mb), (N1,...,Nb)*t]
        g2k_valid_mask = drop_edge_between_samples(g2k_valid_mask, batch=(g2k_batch_g, g2k_batch_k))   #[(M1,...,Mb), (N1,...,Nb)*t]
        g2k_edge_index = dense_to_sparse(g2k_valid_mask)[0]
        g2k_edge_index = g2k_edge_index[:, torch.norm(g2k_position_g[g2k_edge_index[0]] - g2k_position_k[g2k_edge_index[1]], p=2, dim=-1) < self.polygon_radius]
        g2k_edge_vector = transform_point_to_local_coordinate(g2k_position_g[g2k_edge_index[0]], g2k_position_k[g2k_edge_index[1]], g2k_heading_k[g2k_edge_index[1]])
        g2k_edge_attr_length, g2k_edge_attr_theta = compute_angles_lengths_2D(g2k_edge_vector)
        g2k_edge_attr_heading_valid_mask = data['polygon']['heading_valid_mask'][g2k_edge_index[0]]
        g2k_edge_attr_heading = wrap_angle(g2k_heading_g[g2k_edge_index[0]] - g2k_heading_k[g2k_edge_index[1]])
        g2k_edge_attr_input = torch.stack([g2k_edge_attr_length, torch.cos(g2k_edge_attr_theta), torch.sin(g2k_edge_attr_theta), torch.cos(g2k_edge_attr_heading), torch.sin(g2k_edge_attr_heading), g2k_edge_attr_heading_valid_mask], dim=-1)
        g2k_edge_attr_embs = self.g2k_emb_layer(input=g2k_edge_attr_input)

        # k2k_a
        k2k_a_position = infer_position.transpose(0, 1).reshape(-1, 2)      #[t*(N1,...,Nb),2]
        k2k_a_heading = infer_heading.transpose(0, 1).reshape(-1)           #[t*(N1,...,Nb)]
        k2k_a_valid_mask = inference_mask.transpose(0, 1)                   #[t, (N1,...,Nb)]
        k2k_a_valid_mask = k2k_a_valid_mask.unsqueeze(2) & k2k_a_valid_mask.unsqueeze(1)    #[t, (N1,...,Nb), (N1,...,Nb)]
        k2k_a_valid_mask = drop_edge_between_samples(k2k_a_valid_mask, batch=data['agent']['batch'])    #[t, (N1,...,Nb), (N1,...,Nb)]
        k2k_a_edge_index = dense_to_sparse(k2k_a_valid_mask)[0]
        k2k_a_edge_index = k2k_a_edge_index[:, torch.norm(k2k_a_position[k2k_a_edge_index[0]] - k2k_a_position[k2k_a_edge_index[1]], p=2, dim=-1) < self.agent_radius]
        k2k_a_edge_vector = transform_point_to_local_coordinate(k2k_a_position[k2k_a_edge_index[0]], k2k_a_position[k2k_a_edge_index[1]], k2k_a_heading[k2k_a_edge_index[1]])
        k2k_a_edge_attr_length, k2k_a_edge_attr_theta = compute_angles_lengths_2D(k2k_a_edge_vector)
        k2k_a_edge_attr_heading = wrap_angle(k2k_a_heading[k2k_a_edge_index[0]] - k2k_a_heading[k2k_a_edge_index[1]])
        k2k_a_edge_attr_input = torch.stack([k2k_a_edge_attr_length, torch.cos(k2k_a_edge_attr_theta), torch.sin(k2k_a_edge_attr_theta), torch.cos(k2k_a_edge_attr_heading), torch.sin(k2k_a_edge_attr_heading)], dim=-1)
        k2k_a_edge_attr_embs = self.k2k_a_emb_layer(input=k2k_a_edge_attr_input)

        # attention
        for i in range(self.num_attn_layers):
            k_embs = k_embs_dict[i].reshape(-1, self.hidden_dim)   #[(N1,...,Nb)*t,D]
            # k2k_t
            k_embs = self.k2k_t_attn_layers[i](x=k_embs, edge_index=k2k_t_edge_index, edge_attr=k2k_t_edge_attr_embs)
            # g2k
            k_embs = self.g2k_attn_layers[i](x=[g_embs, k_embs], edge_index=g2k_edge_index, edge_attr=g2k_edge_attr_embs)
            # k2k_a
            k_embs = k_embs.reshape(-1, num_intervals, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)    #[t*(N1,...,Nb), D]
            k_embs = self.k2k_a_attn_layers[i](x=k_embs, edge_index=k2k_a_edge_index, edge_attr=k2k_a_edge_attr_embs)
            k_embs = k_embs.reshape(num_intervals, -1, self.hidden_dim).transpose(0, 1)                                 #[(N1,...,Nb), t, D]
            # update k_embs_dict
            # 把第 i 层的输出，写入“下一层”的时间缓存 k_embs_dict[i+1]
            if i+1 in k_embs_dict:
                # 这不是第一次推理：前面步骤已经存过 i+1 层的历史帧
                # 现在只需把“新产生的最后一帧”追加进去，避免重复算老帧
                # 之所以“每层都维护一份”，是因为每层在 t+1 的计算都要用到该层在 1..t 的历史；否则就得层层重算，增量推理的优势（只算新增一帧）就没了。
                k_embs_dict[i+1] = torch.cat([k_embs_dict[i+1], k_embs[:, -1:]], dim=1)   #[(N1,...,Nb),t,D]
            else:
                # 第一次建立 i+1 层缓存：直接把当前 (N, t, D) 全部放进去
                k_embs_dict[i+1] = k_embs                #[(N1,...,Nb),t,D]
        # ipdb.set_trace()
        return k_embs_dict, k_embs[:, -1]