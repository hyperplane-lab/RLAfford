import os
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
import numpy as np
from random import shuffle, randint
import yaml
from tasks.hand_base.base_task import BaseTask
from utils.contact_buffer import ContactBuffer
from tqdm import tqdm
from tasks.franka_pap_partial import PAPPartial
from Collision_Predictor_Module.CollisionPredictor.code.train_with_RL import CollisionPredictor

def quat_axis(q, axis=0):
    """ ?? """
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

class PAPA2O(PAPPartial) :

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False, log_dir=None):
        
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=is_multi_agent, log_dir=log_dir)
        # self.num_obs += 1
        self.task_meta["mask_dim"] = 3
        self.task_meta["obs_dim"] = self.num_obs
        self.task_meta['need_update'] = True
        self.num_feature = cfg["env"]["pointFeatureDim"]
        self.CP_iter = cfg['cp']['CP_iter']
        self.obs_buf = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, self.num_obs), device=self.device, dtype=torch.float)

        self.CollisionPredictor = CollisionPredictor(self.cfg, self.log_dir)
        self.depth_bar = self.cfg["env"]["depth_bar"]
        self.success_rate_bar = self.cfg["cp"]["success_rate_bar"]
        torch.save(self.selected_pc, "/home/shengjie/gyr/E2EAff/logs/PAPA2O/full_map.pt")
        self.one_per_obj_pc = self.selected_pc
        self.raw_map = torch.zeros((self.tot_num, self.objPCDownsampleNum), device=self.device)
        self.map = torch.zeros((self.tot_num, self.objPCDownsampleNum), device=self.device)
        self._refresh_map()


    def _refresh_map(self) :
        
        # predict all points at the begin
        # with torch.no_grad() :
        #     self.map = self.CollisionPredictor.pred_one_batch(
        #         self.selected_cabinet_pc[:, :, :4],
        #         self.success_rate,
        #         num_train=self.env_num_train
        #     ).to(self.device)
        #     self.map *= self.selected_cabinet_pc[:, :, 3]
        #     self.cabinet_pc[:, :, 4] = self.map.to(self.device).repeat_interleave(self.env_per_object, dim=0)
        pass

    def get_map(self, raw_point_clouds, raw_buffer_list):

        map_dis_bar = self.cfg['env']['map_dis_bar']
        top_tensor = torch.tensor([x.top for x in self.contact_buffer_list], device=self.device)
        buffer_size = (raw_buffer_list[-1]).buffer.shape[0]

        buffer = torch.zeros((self.tot_num, buffer_size, 3)).to(self.device)
        for i in range(self.tot_num):
            buffer[i] = raw_buffer_list[i].buffer[:, 0:3]
        raw_point_clouds = raw_point_clouds.float()
        buffer = buffer.float()
        dist_mat = torch.cdist(raw_point_clouds, buffer, p=2)
        if_eff = dist_mat<map_dis_bar
        
        for i in range(self.tot_num):
            if_eff[i, :, top_tensor[i]:] = False

        tot = if_eff.sum(dim=2)
        # tot = torch.log(tot+1)
        tot_scale = tot/(tot.max()+1e-8)  # env*pc

        return tot_scale
    

    def update(self, iter) :

        CP_info = {}
        # used_success_rate = self._detailed_view(self.success_rate).mean(dim=-1)
        
        # if used_success_rate.mean() > self.success_rate_bar :
            # do training only when success rate is enough
        self.raw_map = self.get_map(self.one_per_obj_pc[:, :, :3], self.contact_buffer_list)
        full_map = torch.concat((self.raw_map.reshape(self.num_envs, 2048, 1), self.one_per_obj_pc[:, :, :3]), dim=-1)
        torch.save(full_map, "/home/shengjie/gyr/E2EAff/logs/PAPA2O/map.pt")
        # # stack them together to make resample easier
        # stacked_pc_target = torch.cat(
        #     (
        #         self.one_per_obj_pc[:, :, 3].view(self.tot_num, -1, 1),
        #         self.raw_map.view(self.tot_num, -1, 1)
        #     ),
        #     dim=2
        # )

        # # in training, sample a few points to train CP each epoch
        # minibatch_size = self.cfg["cp"]["cp_minibatch_size"]
        # for i in range(self.CP_iter):
        #     info_list = []
        #     sampled_pc_target = self.sample_points(
        #         stacked_pc_target,
        #         self.objPCDownsampleNum,
        #     )
        #     # sampled_pc_target = self._data_argumentation(sampled_pc_target)
        #     for cur_pc_target, cur_success_rate in zip(
        #             torch.split(sampled_pc_target, minibatch_size),
        #             # torch.split(used_success_rate, minibatch_size)
        #         ) :
        #         # self._refresh_pointcloud_visualizer(cur_pc_target[0, :, :3], cur_pc_target[0, :, 3])
        #         cur_map, cur_info = self.CollisionPredictor.pred_one_batch(
        #             cur_pc_target[:, :, :4],
        #             cur_success_rate,
        #             target=cur_pc_target[:, :, 4],
        #             num_train=self.cabinet_num_train
        #         )
        #         info_list.append(cur_info)
        # self.CollisionPredictor.network_lr_scheduler.step()

        # # collecting training info
        # if self.CP_iter :
        #     for key in info_list[0] :
        #         tmp = 0
        #         for info in info_list :
        #             tmp += info[key]
        #         CP_info[key] = tmp/len(info_list)
    
        # self._refresh_map()

        # return CP_info
        pass

    def _detailed_view(self, tensor) :

        shape = tensor.shape
        return tensor.view(self.tot_num, self.env_per_object, *shape[1:])

    def rand_row(self, tensor, dim_needed):  
        row_total = tensor.shape[1]
        return tensor[:, torch.randint(low=0, high=row_total, size=(dim_needed,)),:]


def transform(x, pos, rot) :

    return quat_apply(rot, x) + pos


def append_mask(x, mask) :

    return torch.cat((x, mask.view(1, 1, -1).repeat_interleave(x.shape[0], dim=0).repeat_interleave(x.shape[1], dim=1)), dim=2)