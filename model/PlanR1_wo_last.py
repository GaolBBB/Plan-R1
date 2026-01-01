# 消融掉了对上一步惩罚
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch
import math
from copy import deepcopy

from layers import TwoLayerMLP
from modules import Backbone, MapEncoder
from metrics import minJointADE, minJointFDE, TokenClsAcc, CumulativeReward
from rewards import AgentCollisionReward, ObstacleCollisionReward, ComfortReward, ProgressReward, SpeedLimitReward, OnRoadReward, TTCReward 
from visualization import visualization
from utils import sample_with_top_k_top_p, move_dict_to_device, transform_point_to_global_coordinate, wrap_angle

import ipdb
class PlanR1(pl.LightningModule):
    def __init__(self,
                 mode: str,
                 token_dict_path: str,
                 num_tokens: int = 1024,
                 interval: int = 5,
                 hidden_dim: int = 128,
                 num_historical_steps: int = 20,
                 num_future_steps: int = 80,
                 agent_radius: float = 60,
                 polygon_radius: float = 30,
                 num_attn_layers: int = 6,
                 pred_top_k: int = 1,
                 plan_top_k: int = 1,
                 rollout_top_k: int = 50,
                 num_samples: int = 4,
                 beta: float = 0.1,
                 scaling_factor: float = 0.1,
                 num_hops: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 lr: float = 0.0003,
                 weight_decay: float = 0.0001,
                 warmup_epochs: int = 4,
                 T_max: int = 32,
                 val_visualization: bool = False,
                 comfort_reward_weight: float = 2,
                 ttc_reward_weight: float = 5,
                 speed_limit_reward_weight: float = 4,
                 progress_reward_weight: float = 2,
                 sft_ul_enable: bool = False,
                 sft_ul_weight: float = 1.0,
                 sft_ul_top_k: int = 20,
                 sft_ul_label_smoothing: float = 0.1,
                 rft_ul_enable: bool = False,
                 rft_ul_weight: float = 1.0,
                 rft_ul_backtrack_steps: int = 1, # 只向前惩罚一步
                 rft_ul_backtrack_gamma: float = 0.5, # 前一步权重惩罚系数
                 rft_ul_nearmiss_weight: float = 0.3, # near-miss 的惩罚权重
                 rft_ul_ttc_threshold: float = 0.5 # # TTCReward 输出是0/1 (1=安全,0=危险/near-miss), <0.5 表示 near-miss
                 ) -> None:
        super(PlanR1, self).__init__()
        self.save_hyperparameters()
        self.mode = mode
        self.token_dict = torch.load(token_dict_path)
        self.num_tokens = num_tokens
        self.interval = interval
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_historical_intervals = num_historical_steps // interval
        self.num_future_intervals = num_future_steps // interval
        self.agent_radius = agent_radius
        self.polygon_radius = polygon_radius
        self.num_attn_layers = num_attn_layers
        self.pred_top_k = pred_top_k
        self.plan_top_k = plan_top_k
        self.rollout_top_k = rollout_top_k
        self.num_samples = num_samples
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max

        # vis in validation
        self.val_visualization = val_visualization

        # reward weights
        self.comfort_reward_weight = comfort_reward_weight
        self.ttc_reward_weight = ttc_reward_weight
        self.speed_limit_reward_weight = speed_limit_reward_weight
        self.progress_reward_weight = progress_reward_weight
        # SFT token-level unlikelihood (UL) settings (used only when mode == 'pred')
        self.sft_ul_enable = sft_ul_enable
        self.sft_ul_weight = sft_ul_weight
        self.sft_ul_top_k = sft_ul_top_k
        # RFT (plan) sequence-level unlikelihood (UL) settings
        self.rft_ul_enable = rft_ul_enable
        self.rft_ul_weight = rft_ul_weight
        self.rft_ul_backtrack_steps = int(rft_ul_backtrack_steps)
        self.rft_ul_backtrack_gamma = float(rft_ul_backtrack_gamma)
        self.rft_ul_nearmiss_weight = float(rft_ul_nearmiss_weight)
        self.rft_ul_ttc_threshold = float(rft_ul_ttc_threshold)

        # pred model
        self.pred_backbone = Backbone(
            token_dict=self.token_dict,
            num_tokens=num_tokens,
            interval=interval,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            agent_radius=agent_radius,
            polygon_radius=polygon_radius,
            num_attn_layers=num_attn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        self.pred_map_encoder = MapEncoder(
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            num_heads=num_heads,
            dropout=dropout
        )
        self.pred_decoder_head = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_tokens)

        # plan model
        self.plan_backbone = Backbone(
            token_dict=self.token_dict,
            num_tokens=num_tokens,
            interval=interval,
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            agent_radius=agent_radius,
            polygon_radius=polygon_radius,
            num_attn_layers=num_attn_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        self.plan_map_encoder = MapEncoder(
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            num_heads=num_heads,
            dropout=dropout
        )
        self.plan_decoder_head = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_tokens)

        # metric
        # self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=sft_ul_label_smoothing)
        self.token_cls_acc = TokenClsAcc()
        self.min_joint_ade = minJointADE()
        self.min_joint_fde = minJointFDE()
        self.reward = CumulativeReward()

        # reward
        self.on_road_reward = OnRoadReward()
        self.agent_collision_reward = AgentCollisionReward()
        self.obstacle_collision_reward = ObstacleCollisionReward()

        self.speed_limit_reward = SpeedLimitReward()
        self.comfort_reward = ComfortReward()
        self.progress_reward = ProgressReward()
        self.ttc_reward = TTCReward()

    def training_step(self, data: Batch) -> None:
        if self.mode == 'pred':
            # pred token and reward
            polygon_embs = self.pred_map_encoder(data=data) 
            feat = self.pred_backbone(data=data, g_embs=polygon_embs)
            logits = self.pred_decoder_head(feat) # [N,T,D]
            # compute loss
            target = data['agent']['recon_token'].roll(-1,1)
            target_mask = data['agent']['recon_token_mask'].roll(-1,1)
            target_mask[:, -1] = False
            # cls_loss = self.cls_loss(logits[target_mask], target[target_mask])
            # self.log('train_cls_loss', cls_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            # return cls_loss
            cls_loss = self.cls_loss(logits[target_mask], target[target_mask])
            self.log('train_cls_loss', cls_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            # ipdb.set_trace()
            # token-level unlikelihood (UL): ego-only, future-only, one-step unsafe check via reward modules
            if self.sft_ul_enable:
                ul_loss, ul_stats = self.compute_sft_ul_loss(
                    data=data, logits=logits, target=target, target_mask=target_mask
                )
                weighted_ul_loss = self.sft_ul_weight * ul_loss
                loss = cls_loss + weighted_ul_loss
                self.log('train_ul_loss', ul_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
                self.log('train_ul_loss_weighted', weighted_ul_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

                self.log('train_ul_hit_rate', ul_stats['hit_rate'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
                self.log('train_ul_unsafe_frac', ul_stats['unsafe_frac'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
                self.log('train_top1_unsafe_rate', ul_stats['top1_unsafe_rate'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
                self.log('train_unsafe_prob_mass', ul_stats['unsafe_prob_mass'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
                # --- diagnostics to explain top1_unsafe behavior ---
                self.log('train_top1_eq_gt_rate', ul_stats['top1_eq_gt_rate'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
                self.log('train_gt_unsafe_rate', ul_stats['gt_unsafe_rate'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
                self.log('train_top1_unsafe_rate_non_gt', ul_stats['top1_unsafe_rate_non_gt'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
                return loss

            return cls_loss
        
        elif self.mode == 'plan':
            # advantages, pred_logps, plan_logps, rewards, valid_mask = self.rollout(data)
            advantages, pred_logps, plan_logps, rewards, valid_mask, done, plan_entropies, ttc_reward = self.rollout(data)
            # ipdb.set_trace()
            ratio = (plan_logps - plan_logps.detach()).exp()
            policy_loss = - (ratio * advantages * valid_mask).sum() / valid_mask.sum()

            kl_loss = torch.exp(pred_logps - plan_logps) - (pred_logps - plan_logps) - 1
            kl_loss = (kl_loss * valid_mask).sum() / valid_mask.sum()

            loss = policy_loss + self.beta * kl_loss
            # -------------------------------------------------
            # RFT sequence-level unlikelihood (UL):
            #   - hard unsafe: first done step + weighted backtrack (t0-1, ...)
            #   - near-miss: TTC < threshold
            # Uses plan_logps directly: p = exp(logp), UL = -log(1-p)
            # -------------------------------------------------
            if self.rft_ul_enable:
                # ipdb.set_trace()
                device = plan_logps.device
                B, T = plan_logps.shape
                eps = 1e-6

                vm = valid_mask.float()

                # ---------- (1) hard unsafe: first done step (+ backtrack with decay) ----------
                neg_w = torch.zeros((B, T), device=device, dtype=torch.float32)
                has_done = done.any(dim=1)  # [B]
                if bool(has_done.any()):
                    t0 = done.float().argmax(dim=1)  # first True index for each traj 
                    b_idx = torch.arange(B, device=device)

                    b0 = b_idx[has_done] # 一个batch中所有不安全的轨迹idx
                    t0v = t0[has_done] # 每个不安全的轨迹的第一个不安全的时间步
                    neg_w[b0, t0v] = 1.0  # weight for first done step

                    # backtrack: penalize t0-1 with gamma (and optionally more steps with gamma^l)
                    # L = int(self.rft_ul_backtrack_steps)
                    # gamma = float(self.rft_ul_backtrack_gamma)
                    # for l in range(1, L + 1):
                    #     bt_mask = has_done & (t0 >= l)
                    #     if not bool(bt_mask.any()):
                    #         continue
                    #     b_bt = b_idx[bt_mask]
                    #     t_bt = t0[bt_mask] - l
                    #     w_bt = gamma ** l
                    #     prev = neg_w[b_bt, t_bt]
                    #     neg_w[b_bt, t_bt] = torch.maximum(prev, torch.full_like(prev, w_bt))

                # ---------- (2) near-miss: TTC < threshold ----------
                # Expect ttc_reward to be [B, T]. If shape differs, tell me and we'll adapt.
                near_mask = (ttc_reward < float(self.rft_ul_ttc_threshold))
                near_w = near_mask.float() * float(self.rft_ul_nearmiss_weight)

                # Combine (take max weight if overlaps)
                neg_w = torch.maximum(neg_w, near_w)

                # Apply valid mask
                neg_w = neg_w * vm

                # ---------- compute weighted UL ----------
                neg_denom = neg_w.sum().clamp(min=1.0)
                p = plan_logps.exp().clamp(min=0.0, max=1.0 - eps)
                ul_per_step = -torch.log1p(-p)  # [B, T]
                rft_ul_loss = (ul_per_step * neg_w).sum() / neg_denom

                weighted_rft_ul_loss = float(self.rft_ul_weight) * rft_ul_loss
                loss = loss + weighted_rft_ul_loss

                # ---------- logs (diagnostics) ----------
                valid_denom = vm.sum().clamp(min=1.0)
                neg_step_rate = (neg_w > 0).float().sum() / valid_denom
                near_miss_rate = (near_mask & valid_mask).float().sum() / valid_denom
                done_traj_rate = has_done.float().mean()
                mean_p_neg = p[neg_w > 0].mean() if bool((neg_w > 0).any()) else torch.zeros((), device=device)

                self.log('train_rft_ul_loss', rft_ul_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
                self.log('train_rft_ul_loss_weighted', weighted_rft_ul_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
                self.log('train_rft_neg_step_rate', neg_step_rate, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
                self.log('train_rft_near_miss_rate', near_miss_rate, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
                self.log('train_rft_done_traj_rate', done_traj_rate, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
                self.log('train_rft_mean_p_neg', mean_p_neg, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            # -----------------------------
            # 1) 熵统计（RFT阶段熵变化/是否坍塌）
            # plan_entropies: [B, T]
            # -----------------------------
            vm = valid_mask.float()
            denom = vm.sum().clamp(min=1.0)

            entropy_mean = (plan_entropies * vm).sum() / denom # [B,T]后每个时间步的熵平均,得到一个数
            entropy_min = plan_entropies.masked_fill(~valid_mask, float('inf')).min() # 所有有效时间步中的最小熵
            entropy_min = torch.where(torch.isfinite(entropy_min), entropy_min, torch.tensor(0.0, device=entropy_min.device)) #对 entropy_min 张量进行「异常值清洗」—— 将所有无穷大（inf）或非数值（nan）的元素替换为 0.0，有限值的元素保持不变

            # -----------------------------
            # 2) GRPO组内是否出现“全安全/全不安全”
            # done: [B, T]，True表示发生不安全/终止事件
            # 单条轨迹安全：整个T都没有done
            # -----------------------------
            traj_safe = ~done.any(dim=1)  # [B]
            B = traj_safe.shape[0]
            if B % self.num_samples == 0:
                traj_safe_g = traj_safe.view(self.num_samples, -1)  # [num_sample, B]
                frac_all_safe = traj_safe_g.all(dim=0).float().mean() # 一个batch下，一个prompt进行rollout的num_sample个样本全部安全的prompt个数的比例
                frac_all_unsafe = (~traj_safe_g).all(dim=0).float().mean() # 一个batch下，一个prompt进行rollout的num_sample个样本全部不安全的prompt个数的比例
                safe_rate = traj_safe_g.float().mean() # 一个batch中所有rollout轨迹里的安全轨迹率
            else:
                frac_all_safe = torch.tensor(0.0, device=traj_safe.device)
                frac_all_unsafe = torch.tensor(0.0, device=traj_safe.device)
                safe_rate = traj_safe.float().mean()

            # -----------------------------
            # 3) reward / advantage masked mean/std
            # rewards/advantages: [B, T]
            # -----------------------------
            reward_mean = (rewards * vm).sum() / denom
            reward_sq_mean = ((rewards ** 2) * vm).sum() / denom
            reward_std = (reward_sq_mean - reward_mean ** 2).clamp(min=0.0).sqrt()

            adv_mean = (advantages * vm).sum() / denom
            adv_sq_mean = ((advantages ** 2) * vm).sum() / denom
            adv_std = (adv_sq_mean - adv_mean ** 2).clamp(min=0.0).sqrt()
            self.log('train_policy_loss', policy_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_kl_loss', kl_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_reward', rewards.mean(), prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            # --- new logs for analysis/plotting ---
            self.log('train_entropy_mean', entropy_mean, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_entropy_min', entropy_min, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

            self.log('train_group_frac_all_safe', frac_all_safe, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_group_frac_all_unsafe', frac_all_unsafe, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_group_safe_rate', safe_rate, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

            self.log('train_reward_mean_masked', reward_mean, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_reward_std_masked', reward_std, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_adv_mean_masked', adv_mean, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('train_adv_std_masked', adv_std, prog_bar=False, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
            return loss

    def pred_inference(self, data: Batch):
        map_encoder = self.pred_map_encoder
        backbone = self.pred_backbone
        decoder_head = self.pred_decoder_head

        polygon_embs = map_encoder(data=data)
        agent_embs, k_embs_dict = backbone.pre_inference(data=data)

        for _ in range(self.num_future_intervals):
            k_embs_dict, k_embs_step = backbone.inference(data=data, g_embs=polygon_embs, a_embs=agent_embs, k_embs_dict=k_embs_dict)
            logits_step = decoder_head(k_embs_step)
            action_step = sample_with_top_k_top_p(logits_step.unsqueeze(1), top_k=self.pred_top_k).squeeze(1).squeeze(1)

            data = self.transition(data, action_step)

        position = data['agent']['infer_position'][:, self.num_historical_intervals:]
        heading = data['agent']['infer_heading'][:, self.num_historical_intervals:]
        valid_mask = data['agent']['infer_valid_mask'][:, self.num_historical_intervals:]

        return data, position, heading, valid_mask

    def plan_inference(self, data: Batch):
        ego_index = data['agent']['ptr'][:-1]

        pred_polygon_embs = self.pred_map_encoder(data=data)
        pred_agent_embs, pred_k_embs_dict = self.pred_backbone.pre_inference(data=data)

        plan_polygon_embs = self.plan_map_encoder(data=data)
        plan_agent_embs, plan_k_embs_dict = self.plan_backbone.pre_inference(data=data)

        for _ in range(self.num_future_intervals):
            pred_k_embs_dict, pred_k_embs_step = self.pred_backbone.inference(data=data, g_embs=pred_polygon_embs, a_embs=pred_agent_embs, k_embs_dict=pred_k_embs_dict)
            plan_k_embs_dict, plan_k_embs_step = self.plan_backbone.inference(data=data, g_embs=plan_polygon_embs, a_embs=plan_agent_embs, k_embs_dict=plan_k_embs_dict)

            pred_logits_step = self.pred_decoder_head(pred_k_embs_step)
            plan_logits_step = self.plan_decoder_head(plan_k_embs_step)
            action_step = sample_with_top_k_top_p(pred_logits_step.unsqueeze(1), top_k=self.pred_top_k).squeeze(1).squeeze(1)
            ego_action_step = sample_with_top_k_top_p(plan_logits_step[ego_index].unsqueeze(1), top_k=self.plan_top_k).squeeze(1).squeeze(1)
            action_step[ego_index] = ego_action_step

            data = self.transition(data, action_step)

        position = data['agent']['infer_position'][:, self.num_historical_intervals:]
        heading = data['agent']['infer_heading'][:, self.num_historical_intervals:]
        valid_mask = data['agent']['infer_valid_mask'][:, self.num_historical_intervals:]

        return data, position, heading, valid_mask

    def validation_step(self, data: Batch, batch_idx: int) -> None:
        if self.mode == 'pred':
            # pred token and reward
            polygon_embs = self.pred_map_encoder(data=data) 
            feat = self.pred_backbone(data=data, g_embs=polygon_embs)
            logits = self.pred_decoder_head(feat)
            # compute loss
            target = data['agent']['recon_token'].roll(-1,1)
            target_mask = data['agent']['recon_token_mask'].roll(-1,1)
            target_mask[:, -1] = 0
            cls_loss = self.cls_loss(logits[target_mask], target[target_mask])
            self.log('val_token_cls_acc', self.token_cls_acc(logits[target_mask], target[target_mask]), prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_cls_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            # inference
            _, position, heading, valid_mask = self.pred_inference(data)

        elif self.mode == 'plan':
            # inference
            data, position, heading, valid_mask = self.plan_inference(data)
            # compute rewards
            rewards, _, _ , _= self.reward_fn(data)
            self.reward.update(rewards)
            self.log('val_reward', self.reward, prog_bar=True, on_step=False, on_epoch=True)

        agent_batch = data['agent']['batch']
        agent_pred_traj = unbatch(position.unsqueeze(1), agent_batch)
        agent_target_traj = unbatch(data['agent']['position'][:, self.num_historical_steps+self.interval::self.interval], agent_batch)
        agent_mask = unbatch(data['agent']['visible_mask'][:, self.num_historical_steps+self.interval::self.interval] & valid_mask, agent_batch)

        self.min_joint_ade.update(agent_pred_traj, agent_target_traj, agent_mask)
        self.min_joint_fde.update(agent_pred_traj, agent_target_traj, agent_mask)
        self.log('val_min_joint_ade', self.min_joint_ade, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_min_joint_fde', self.min_joint_fde, prog_bar=True, on_step=False, on_epoch=True)

        if self.val_visualization:
            visualization(data, position, heading)

    def freeze_pred_model(self):
        # eval mode
        self.pred_map_encoder.eval()
        self.pred_backbone.eval()
        self.pred_decoder_head.eval()
        # freeze
        for p in self.pred_map_encoder.parameters():
            p.requires_grad = False
        for p in self.pred_backbone.parameters():
            p.requires_grad = False
        for p in self.pred_decoder_head.parameters():
            p.requires_grad = False

    def on_train_start(self):
        if self.mode == 'plan':
            self.plan_map_encoder.load_state_dict(self.pred_map_encoder.state_dict())
            self.plan_backbone.load_state_dict(self.pred_backbone.state_dict())
            self.plan_decoder_head.load_state_dict(self.pred_decoder_head.state_dict())
            self.freeze_pred_model()

    def transition(self, data, action):
        next_data = data.clone()

        next_data['agent']['infer_token'] = torch.cat([next_data['agent']['infer_token'], action.unsqueeze(1)], dim=1)
        next_data['agent']['infer_token_mask'] = torch.cat([next_data['agent']['infer_token_mask'], next_data['agent']['infer_token_mask'][:, -1:]], dim=1)

        a_type = data['agent']['type']
        vehicle_mask = a_type == 0
        pedestrian_mask = a_type == 1
        bicycle_mask = a_type == 2

        token = torch.zeros(action.size(0), 3, device=action.device)
        self.token_dict = move_dict_to_device(self.token_dict, action.device)
        token[vehicle_mask] = self.token_dict['Vehicle'][action[vehicle_mask]]
        token[pedestrian_mask] = self.token_dict['Pedestrian'][action[pedestrian_mask]]
        token[bicycle_mask] = self.token_dict['Bicycle'][action[bicycle_mask]]
        token_position = transform_point_to_global_coordinate(token[:, :2], next_data['agent']['infer_position'][:, -1], next_data['agent']['infer_heading'][:, -1])
        token_heading = wrap_angle(token[:, 2] + next_data['agent']['infer_heading'][:, -1])

        next_data['agent']['infer_position'] = torch.cat([next_data['agent']['infer_position'], token_position.unsqueeze(1)], dim=1)
        next_data['agent']['infer_heading'] = torch.cat([next_data['agent']['infer_heading'], token_heading.unsqueeze(1)], dim=1)
        next_data['agent']['infer_valid_mask'] = torch.cat([next_data['agent']['infer_valid_mask'], next_data['agent']['infer_valid_mask'][:, -1:]], dim=1)

        return next_data
        
    def rollout(self, data):
        # copy data
        data_list = data.to_data_list()
        data_list_copy = deepcopy(data_list)
        for _ in range(self.num_samples - 1):
            data_list += data_list_copy
        data = Batch.from_data_list(data_list)
        ego_index = data['agent']['ptr'][:-1]
        # ipdb> data['log_name']
        # ['2718', '1521', '3036', '0967', '2718', '1521', '3036', '0967','2718', '1521', '3036', '0967','2718', '1521', '3036', '0967',]

        # initialize
        pred_polygon_embs = self.pred_map_encoder(data=data)
        pred_agent_embs, pred_k_embs_dict = self.pred_backbone.pre_inference(data=data)
        plan_polygon_embs = self.plan_map_encoder(data=data)
        plan_agent_embs, plan_k_embs_dict = self.plan_backbone.pre_inference(data=data)

        # inference
        for step in range(self.num_future_intervals):
            pred_k_embs_dict, pred_k_embs_step = self.pred_backbone.inference(data=data, g_embs=pred_polygon_embs, a_embs=pred_agent_embs, k_embs_dict=pred_k_embs_dict)
            plan_k_embs_dict, plan_k_embs_step = self.plan_backbone.inference(data=data, g_embs=plan_polygon_embs, a_embs=plan_agent_embs, k_embs_dict=plan_k_embs_dict)
            
            pred_logits_step = self.pred_decoder_head(pred_k_embs_step)
            plan_logits_step = self.plan_decoder_head(plan_k_embs_step)

            action_step = sample_with_top_k_top_p(pred_logits_step.unsqueeze(1), top_k=1).squeeze(1).squeeze(1)
            ego_action_step = sample_with_top_k_top_p(plan_logits_step[ego_index].unsqueeze(1), top_k=self.rollout_top_k).squeeze(1).squeeze(1)
            
            pred_dist_step = Categorical(logits=pred_logits_step[ego_index])
            plan_dist_step = Categorical(logits=plan_logits_step[ego_index])
            plan_entropy_step = plan_dist_step.entropy().unsqueeze(1)  # [B, 1]
            if step == 0:
                pred_logps = pred_dist_step.log_prob(ego_action_step).unsqueeze(1)
                plan_logps = plan_dist_step.log_prob(ego_action_step).unsqueeze(1)
                plan_entropies = plan_entropy_step
            else:
                pred_logps = torch.cat([pred_logps, pred_dist_step.log_prob(ego_action_step).unsqueeze(1)], dim=1)
                plan_logps = torch.cat([plan_logps, plan_dist_step.log_prob(ego_action_step).unsqueeze(1)], dim=1)
                plan_entropies = torch.cat([plan_entropies, plan_entropy_step], dim=1) #最终[B,T]，即每个时间步下这个分布的熵
            action_step[ego_index] = ego_action_step
            data = self.transition(data, action_step)

        # rewards, _, valid_mask = self.reward_fn(data)
        rewards, done, valid_mask, ttc_reward = self.reward_fn(data)

        advantages = self.compute_ae_process_supervision(rewards, valid_mask)

        # return advantages, pred_logps, plan_logps, rewards, valid_mask
        return advantages, pred_logps, plan_logps, rewards, valid_mask, done, plan_entropies, ttc_reward
    
    def compute_ae_outcome_supervision(self, rewards):
        # group computation
        rewards_reshape = rewards.view(self.num_samples, -1)
        rewards_mean = rewards_reshape.mean(dim=0)
        rewards_std = rewards_reshape.std(dim=0)

        advantages = (rewards_reshape - rewards_mean) / (rewards_std + 1e-4)
        advantages = advantages.view(-1).unsqueeze(-1).expand(-1, self.num_future_intervals)

        return advantages
    
    def compute_ae_process_supervision(self, rewards, valid_mask):
        B, T = rewards.shape
        # ipdb.set_trace()
        rewards_reshape = rewards.view(self.num_samples, -1, T).transpose(0, 1)
        valid_mask_reshape = valid_mask.view(self.num_samples, -1, T).transpose(0, 1)
        rewards_reshape = rewards_reshape * valid_mask_reshape.float()

        rewards_mean = rewards_reshape.sum(dim=[1, 2]) / valid_mask_reshape.sum(dim=[1, 2]) # torch.Size([4,4,16])-> torch.Size([4])

        # normalization
        # rewards_std = (rewards_reshape ** 2).sum(dim=[1, 2]) / valid_mask_reshape.sum(dim=[1, 2]) - rewards_mean ** 2
        # rewards_std = torch.sqrt(rewards_std.clamp(min=1e-4))
        # rewards_norm = (rewards_reshape - rewards_mean.view(-1, 1, 1)) / (rewards_std.view(-1, 1, 1) + 1e-4)

        # centering + scaling
        rewards_norm = (rewards_reshape - rewards_mean.view(-1, 1, 1)) / self.scaling_factor

        rewards_norm = rewards_norm.transpose(0, 1).reshape(B, T)

        rewards_norm[~valid_mask] = 0.0
        advantages = torch.zeros_like(rewards_norm)
        for step in reversed(range(T)):
            if step == T - 1:
                advantages[:, step] = rewards_norm[:, step]
            else:
                advantages[:, step] = rewards_norm[:, step] + advantages[:, step + 1] * valid_mask[:, step + 1].float()

        return advantages
    def compute_sft_ul_loss(self, data: Batch, logits: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor):
        """Token-level UL for SFT (mode=='pred'): ego-only, future-only.

        Negative candidates come from top-K tokens at each future step. A candidate is unsafe if one-step
        execution triggers any of (agent collision / obstacle collision / off-road) according to reward modules.

        Returns:
            ul_loss (scalar tensor)
            stats (dict): {'hit_rate': ..., 'unsafe_frac': ..., 'top1_unsafe_rate': ..., 'unsafe_prob_mass': ...}
        """
        # ipdb.set_trace()
        device = logits.device
        ego_index = data['agent']['ptr'][:-1]  # [B]
        B = ego_index.numel()
        if B == 0:
            zero = torch.zeros((), device=device)
            return zero, {'hit_rate': zero, 'unsafe_frac': zero}

        # ego-only tensors
        logits_ego = logits[ego_index]          # [B, 20, V]
        target_ego = target[ego_index]          # [B, 20]
        mask_ego = target_mask[ego_index]       # [B, 20]

        H = self.num_historical_intervals
        F = self.num_future_intervals
        start_t = H - 1  # maps to j=t+1=H

        total_ul = torch.zeros((), device=device)
        unsafe_term_count = torch.zeros((), device=device)
        checked_term_count = torch.zeros((), device=device)
        hit_step_count = torch.zeros((), device=device)
        valid_step_count = torch.zeros((), device=device)
        top1_unsafe_count = torch.zeros((), device=device)
        top1_total_count = torch.zeros((), device=device)
        unsafe_mass_total = torch.zeros((), device=device)
        unsafe_mass_denom = torch.zeros((), device=device)
        # --- diagnostics to understand why top1_unsafe may not improve ---
        top1_eq_gt_count = torch.zeros((), device=device)
        top1_eq_gt_total = torch.zeros((), device=device)

        gt_unsafe_count = torch.zeros((), device=device)
        gt_total_count = torch.zeros((), device=device)

        top1_unsafe_non_gt_count = torch.zeros((), device=device)
        top1_non_gt_total = torch.zeros((), device=device)

        K = int(self.sft_ul_top_k)
        eps = 1e-6

        # iterate over future intervals only (j in [H, H+F-1])
        for i in range(F):
            # 每个时刻
            t = start_t + i # TODO 从3开始，真对吗？
            j = t + 1

            valid_b = mask_ego[:, t]
            if not bool(valid_b.any()):
                continue

            valid_step_count = valid_step_count + 1.0
            # top-1 unsafe rate (argmax token)
            top1_tok = torch.argmax(logits_ego[:, t, :], dim=-1)  # [B]
            top1_unsafe_b = self._unsafe_one_step_for_ego(
                data=data, ego_index=ego_index, j=j, ego_tokens=top1_tok
            )  # [B]
            top1_unsafe_count = top1_unsafe_count + (top1_unsafe_b & valid_b).sum().float()
            top1_total_count = top1_total_count + valid_b.sum().float()

            # --- diagnostics ---
            gt_tok_step = target_ego[:, t]  # [B]

            # (1) top1 == GT rate
            eq_gt_b = (top1_tok == gt_tok_step) & valid_b
            top1_eq_gt_count = top1_eq_gt_count + eq_gt_b.sum().float()
            top1_eq_gt_total = top1_eq_gt_total + valid_b.sum().float()

            # (2) GT unsafe rate under the same one-step checker (diagnostic only; GT is not penalized by UL)
            gt_unsafe_b = self._unsafe_one_step_for_ego(
                data=data, ego_index=ego_index, j=j, ego_tokens=gt_tok_step
            )  # [B]
            gt_unsafe_count = gt_unsafe_count + (gt_unsafe_b & valid_b).sum().float()
            gt_total_count = gt_total_count + valid_b.sum().float()

            # (3) top1 unsafe rate restricted to positions where top1 != GT
            non_gt_top1_b = (top1_tok != gt_tok_step) & valid_b
            top1_unsafe_non_gt_count = top1_unsafe_non_gt_count + (top1_unsafe_b & non_gt_top1_b).sum().float()
            top1_non_gt_total = top1_non_gt_total + non_gt_top1_b.sum().float()

            # unsafe probability mass denominator counts valid (b,t)
            unsafe_mass_denom = unsafe_mass_denom + valid_b.sum().float()

            # candidates: [B, K]
            topk = torch.topk(logits_ego[:, t, :], k=K, dim=-1).indices # torch.Size([8, 20]) 这一时刻下[B,K]
            gt = target_ego[:, t].unsqueeze(1) #这一时刻下的GT [B,1]
            non_gt = topk.ne(gt) # 看看B个ego这个时刻采样出的K个token是不是GT

            # probabilities at this step
            probs = torch.softmax(logits_ego[:, t, :], dim=-1)  # [B, V]

            step_hit = False

            # naive UL: loop over k; each call checks all B egos for their k-th candidate
            for k in range(K):
                cand_tok = topk[:, k]  # [B]
                cand_ok = non_gt[:, k] & valid_b # 只有不是GT且有效的才可以进入UL候选
                if not bool(cand_ok.any()):
                    continue

                unsafe_b = self._unsafe_one_step_for_ego(
                    data=data, ego_index=ego_index, j=j, ego_tokens=cand_tok
                )  # [B]
                unsafe_b = unsafe_b & cand_ok 

                # gather p(cand)
                p_sel = probs.gather(1, cand_tok.unsqueeze(1)).squeeze(1)
                p_sel = p_sel.clamp(min=0.0, max=1.0 - eps)

                # unsafe probability mass over top-K (sum of p for unsafe candidates)
                if bool(unsafe_b.any()):
                    unsafe_mass_total = unsafe_mass_total + p_sel[unsafe_b].sum()
                # accumulate only unsafe terms
                if bool(unsafe_b.any()):
                    step_hit = True
                    total_ul = total_ul + (-torch.log1p(-p_sel[unsafe_b])).sum()
                    unsafe_term_count = unsafe_term_count + unsafe_b.sum().float()

                # count checked terms (for unsafe_frac)
                checked_term_count = checked_term_count + cand_ok.sum().float()

            if step_hit:
                hit_step_count = hit_step_count + 1.0
        # ipdb.set_trace()
        denom = unsafe_term_count.clamp(min=1.0)
        ul_loss = total_ul / denom

        # stats
        hit_rate = hit_step_count / valid_step_count.clamp(min=1.0)
        unsafe_frac = unsafe_term_count / checked_term_count.clamp(min=1.0)
        top1_unsafe_rate = top1_unsafe_count / top1_total_count.clamp(min=1.0)
        unsafe_prob_mass = unsafe_mass_total / unsafe_mass_denom.clamp(min=1.0)
        # --- diagnostics ---
        top1_eq_gt_rate = top1_eq_gt_count / top1_eq_gt_total.clamp(min=1.0)
        gt_unsafe_rate = gt_unsafe_count / gt_total_count.clamp(min=1.0)
        top1_unsafe_rate_non_gt = top1_unsafe_non_gt_count / top1_non_gt_total.clamp(min=1.0)


        return ul_loss, {
            'hit_rate': hit_rate.detach(),
            'unsafe_frac': unsafe_frac.detach(),
            'top1_unsafe_rate': top1_unsafe_rate.detach(),
            'unsafe_prob_mass': unsafe_prob_mass.detach(),
            # diagnostics
            'top1_eq_gt_rate': top1_eq_gt_rate.detach(),
            'gt_unsafe_rate': gt_unsafe_rate.detach(),
            'top1_unsafe_rate_non_gt': top1_unsafe_rate_non_gt.detach(),
        }
    def _unsafe_one_step_for_ego(self, data: Batch, ego_index: torch.Tensor, j: int, ego_tokens: torch.Tensor) -> torch.Tensor:
        """One-step unsafe check for ego at future interval index j (j in [H, H+F-1]).

        Builds a temporary infer_* sequence with future length T=1 by taking recon_* history for the first H
        intervals and a single evaluated frame at index H.

        Evaluated frame:
          - non-ego agents use GT recon_position/recon_heading at time j
          - ego uses candidate pose obtained by applying token delta to GT pose at time j-1

        Returns:
            unsafe: [B] bool, where B == len(ego_index)
        """
        # ipdb.set_trace()
        device = ego_tokens.device
        H = self.num_historical_intervals

        # GT recon sequences (interval-level)
        recon_pos = data['agent']['recon_position'].to(device)         # [N, 20, 2]
        recon_yaw = data['agent']['recon_heading'].to(device)          # [N, 20]
        recon_valid = data['agent']['recon_valid_mask'].to(device)     # [N, 20]

        # ego pose before executing the candidate token (at j-1)
        ego_pos_before = recon_pos[ego_index, j - 1]  # [B, 2]
        ego_yaw_before = recon_yaw[ego_index, j - 1]  # [B]

        # decode token to local delta (dx, dy, dyaw) depending on agent type
        self.token_dict = move_dict_to_device(self.token_dict, device)
        ego_type = data['agent']['type'][ego_index].to(device)  # [B]
        vehicle_mask = ego_type == 0
        pedestrian_mask = ego_type == 1
        bicycle_mask = ego_type == 2

        token_delta = torch.zeros(ego_tokens.size(0), 3, device=device)
        if bool(vehicle_mask.any()):
            token_delta[vehicle_mask] = self.token_dict['Vehicle'][ego_tokens[vehicle_mask]]
        if bool(pedestrian_mask.any()):
            token_delta[pedestrian_mask] = self.token_dict['Pedestrian'][ego_tokens[pedestrian_mask]]
        if bool(bicycle_mask.any()):
            token_delta[bicycle_mask] = self.token_dict['Bicycle'][ego_tokens[bicycle_mask]]

        ego_pos_after = transform_point_to_global_coordinate(token_delta[:, :2], ego_pos_before, ego_yaw_before) # [B,2]
        ego_yaw_after = wrap_angle(token_delta[:, 2] + ego_yaw_before) # [B]

        # evaluated snapshot at time j: start from GT for all agents, replace ego with candidate
        pos_all = recon_pos[:, j].clone()      # [N, 2]
        yaw_all = recon_yaw[:, j].clone()      # [N]
        valid_all = recon_valid[:, j].clone()  # [N]
        # 把ego的替换下来
        pos_all[ego_index] = ego_pos_after
        yaw_all[ego_index] = ego_yaw_after
        valid_all[ego_index] = torch.ones_like(valid_all[ego_index], dtype=valid_all.dtype)

        # build infer_* with future length T=1 (infer_len = H+1)
        infer_position = torch.cat([recon_pos[:, :H], pos_all.unsqueeze(1)], dim=1)  # [N, H+1, 2]
        infer_heading = torch.cat([recon_yaw[:, :H], yaw_all.unsqueeze(1)], dim=1)   # [N, H+1]
        infer_valid_mask = torch.cat([recon_valid[:, :H], valid_all.unsqueeze(1)], dim=1)  # [N, H+1]

        data_tmp = data.clone()
        data_tmp['agent']['infer_position'] = infer_position
        data_tmp['agent']['infer_heading'] = infer_heading
        data_tmp['agent']['infer_valid_mask'] = infer_valid_mask

        agent_collision_done, _ = self.agent_collision_reward(data_tmp)
        obstacle_collision_done, _ = self.obstacle_collision_reward(data_tmp)
        on_road_done, _ = self.on_road_reward(data_tmp)

        done = agent_collision_done | obstacle_collision_done | on_road_done  # [B, 1]
        return done.squeeze(1)
    def reward_fn(self, data):
        # progress reward
        agent_collision_done, agent_collision_reward = self.agent_collision_reward(data)
        obstacle_collision_done, obstacle_collision_reward = self.obstacle_collision_reward(data)
        on_road_done, on_road_reward = self.on_road_reward(data)
        done = agent_collision_done | obstacle_collision_done | on_road_done

        valid_mask = (~done).float().cumprod(dim=1).bool()
        valid_mask = torch.cat([torch.ones(done.size(0), 1, device=done.device, dtype=torch.bool), valid_mask[:, :-1]], dim=1)

        comfort_reward = self.comfort_reward(data)
        ttc_reward = self.ttc_reward(data)

        # outcome reward
        progress_reward = self.progress_reward(data)
        speed_limit_reward = self.speed_limit_reward(data)
        progress_reward = progress_reward.unsqueeze(1).expand(-1, self.num_future_intervals) 
        speed_limit_reward = speed_limit_reward.unsqueeze(1).expand(-1, self.num_future_intervals)

        reward = on_road_reward * obstacle_collision_reward * agent_collision_reward * (
                 self.comfort_reward_weight * comfort_reward + 
                 self.ttc_reward_weight * ttc_reward + 
                 self.speed_limit_reward_weight * speed_limit_reward + 
                 self.progress_reward_weight * progress_reward) / (self.comfort_reward_weight + self.ttc_reward_weight + self.speed_limit_reward_weight + self.progress_reward_weight)

        return reward, done, valid_mask, ttc_reward
    
    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        
        warmup_epochs = self.warmup_epochs
        T_max = self.T_max

        def warmup_cosine_annealing_schedule(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs + 1) / (T_max - warmup_epochs + 1)))

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_annealing_schedule),
            'interval': 'epoch',
            'frequency': 1
        }

        return [optimizer], [scheduler]