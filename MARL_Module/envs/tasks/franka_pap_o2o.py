import os
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
import numpy as np
from random import shuffle, randint
import yaml
from MARL_Module.envs.tasks.franka_pap import PAPRaw
from utils.contact_buffer import ContactBuffer
from tqdm import tqdm
from pointnet2_ops import pointnet2_utils
def quat_axis(q, axis=0):
    """ ?? """
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

class PAPO2O(PAPRaw) :

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False, log_dir=None):
        
        self.objPCDownsampleNum = cfg["env"]["objPointDownsampleNum"]
        self.handPCDownsampleNum = cfg["env"]["handPointDownsampleNum"]
        self.moving_pc_mode = cfg["env"]["moving_pc_mode"]
        self.pointCloudDownsampleNum = self.objPCDownsampleNum + self.handPCDownsampleNum 
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=is_multi_agent, log_dir=log_dir)

        self.num_obs += 4

        self.task_meta["obs_dim"] = self.num_obs

        self.num_feature = cfg["env"]["pointFeatureDim"]
        self.obs_buf = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, self.num_obs), device=self.device, dtype=torch.float)
        self.task_meta["mask_dim"] = 1
        
        if cfg["env"]["visualizePointcloud"] == True :
            import open3d as o3d
            from utils.o3dviewer import PointcloudVisualizer
            self.pointCloudVisualizer = PointcloudVisualizer()
            self.pointCloudVisualizerInitialized = False
            self.o3d_pc = o3d.geometry.PointCloud()
        else :
            self.pointCloudVisualizer = None
        self.selected_pc = self.sample_points(self.obj_pc, self.objPCDownsampleNum, sample_method='random')


    # @TimeCounter
    def sample_points(self, points, sample_num=1000, sample_method='random'):
        eff_points = points
        if sample_method == 'random':
            sampled_points = self.rand_row(eff_points, sample_num)
        elif sample_method == 'furthest':
            sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points.reshape(1, *eff_points.shape), sample_num)
            sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
        return sampled_points
    
    def _refresh_pointcloud_visualizer(self, point_clouds, data) :

        if isinstance(point_clouds, list) :
            for a in point_clouds :
                a = a.cpu().numpy()
            points = np.concatenate(point_clouds, axis=0)
        else :
            points = point_clouds.cpu().numpy()
        
        if isinstance(data, list):
            for a in data :
                a = a.cpu().numpy()
            colors = np.concatenate(data, axis=0)
        else :
            colors = data.cpu().numpy()

        import open3d as o3d
        hand_rot = self.hand_rigid_body_tensor[..., 3:7]
        hand_down_dir = quat_axis(hand_rot, 2)
        hand_pos = self.hand_rigid_body_tensor[..., 0:3] + hand_down_dir * 0.130
        points = np.concatenate((points, np.array(hand_pos)), axis=0)
        colors = plt.get_cmap()(colors)[:, :3]
        colors = np.concatenate((colors, np.ones(hand_pos.shape)), axis=0)
        self.o3d_pc.points = o3d.utility.Vector3dVector(points)
        self.o3d_pc.colors = o3d.utility.Vector3dVector(colors)

        if self.pointCloudVisualizerInitialized == False :
            self.pointCloudVisualizer.add_geometry(self.o3d_pc)
            self.pointCloudVisualizerInitialized = True
        else :
            self.pointCloudVisualizer.update(self.o3d_pc)

    def _refresh_observation(self) :
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.cfg["env"]["driveMode"] == "ik" :
            self.gym.refresh_jacobian_tensors(self.sim)

        point_clouds = self.compute_point_cloud_state()

        # print(point_clouds[0].mean(dim=0), point_clouds[1].mean(dim=0))

        point_clouds[:, :, :3] -= self.franka_root_tensor[:, :3].view(-1, 1, 3)
        self.obs_buf[:, :, :4].copy_(point_clouds)
        self.obs_buf[:, :, 4:] = self._get_base_observation().view(self.num_envs, 1, -1).repeat_interleave(self.pointCloudDownsampleNum, dim=1)
        # self.obs_buf_states = self._get_base_observation()
        if self.pointCloudVisualizer != None :
            self._refresh_pointcloud_visualizer(
                [point_clouds[0, :, :3], transform(self.contact_buffer_list[0].all()[:, :3], self.object_root_tensor[:, :3]-self.franka_root_tensor[:, :3],  self.object_root_tensor[:, 3:7])],
                [point_clouds[0, :, 3], torch.ones((self.contact_buffer_list[0].top,)) * 0.5]
            )

    def rand_row(self, tensor, dim_needed):  
        row_total = tensor.shape[1]
        return tensor[:, torch.randint(low=0, high=row_total, size=(dim_needed,)),:]

    def compute_point_cloud_state(self) :
        obj_r = self.root_tensor[:, 1, 3:7]
        obj_p = self.root_tensor[:, 1, 0:3]

        lfinger_p = self.rigid_body_tensor[:, self.hand_lfinger_rigid_body_index, :3]
        lfinger_r = self.rigid_body_tensor[:, self.hand_lfinger_rigid_body_index, 3:7]
        rfinger_p = self.rigid_body_tensor[:, self.hand_rfinger_rigid_body_index, :3]
        rfinger_r = self.rigid_body_tensor[:, self.hand_rfinger_rigid_body_index, 3:7]
        
        self.selected_pc = self.sample_points(self.obj_pc, self.objPCDownsampleNum, sample_method='random')
        selected_lfinger_pc = self.sample_points(self.franka_left_finger_pc, self.handPCDownsampleNum//2, sample_method='random')
        selected_rfinger_pc = self.sample_points(self.franka_right_finger_pc, self.handPCDownsampleNum//2, sample_method='random')

        obj_pc_shape = self.selected_pc.shape
        lfinger_shape = selected_lfinger_pc.shape
        rfinger_shape = selected_rfinger_pc.shape
        if self.moving_pc_mode:
            obj_point_clouds = quat_apply(obj_r.view(-1, 1, 4).repeat_interleave(obj_pc_shape[1], dim=1), self.selected_pc[:, :, :3].float()) + obj_p.view(-1, 1, 3)
        else:
            obj_point_clouds = self.selected_pc
        lfinger_point_clouds = quat_apply(lfinger_r.view(-1, 1, 4).repeat_interleave(lfinger_shape[1], dim=1), selected_lfinger_pc) + lfinger_p.view(-1, 1, 3)
        rfinger_point_clouds = quat_apply(rfinger_r.view(-1, 1, 4).repeat_interleave(rfinger_shape[1], dim=1), selected_rfinger_pc) + rfinger_p.view(-1, 1, 3)
        obj_point_clouds = append_mask(obj_point_clouds, torch.tensor([0], device=self.device))
        lfinger_point_clouds = append_mask(lfinger_point_clouds, torch.tensor([1], device=self.device))
        rfinger_point_clouds = append_mask(rfinger_point_clouds, torch.tensor([1], device=self.device))

        point_clouds = torch.cat((obj_point_clouds, lfinger_point_clouds, rfinger_point_clouds), dim=1)

        return point_clouds


def transform(x, pos, rot) :

    return quat_apply(rot, x) + pos


def append_mask(x, mask) :

    return torch.cat((x, mask.view(1, 1, -1).repeat_interleave(x.shape[0], dim=0).repeat_interleave(x.shape[1], dim=1)), dim=2)