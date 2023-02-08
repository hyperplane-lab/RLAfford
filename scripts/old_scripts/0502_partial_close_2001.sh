python train.py --task=OneFrankaCabinetPCPartial --task_config=cfg/franka_cabinet_PC_partial_cloud_close.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --seed=2001

# door_reward = self.cabinet_dof_coef * self.cabinet_dof_tensor[:, 0]
# # act_penalty = torch.norm(self.eff_act, p=2, dim=1) * 0.005
# action_penalty = torch.sum((self.actions[:, :7]-self.franka_dof_tensor[:, :7, 0])**2, dim=-1)*0.01

# dis_penalty = torch.norm(hand_pos - handle_pos, p=2, dim=1) * self.cabinet_have_handle_tensor * 10.0

# d = torch.norm(hand_pos - handle_pos, p=2, dim=-1)
# # reward for reaching the handle
# dist_reward = 1.0 / (1.0 + d**2)
# dist_reward *= dist_reward
# dist_reward = torch.where(d <= 0.02, dist_reward*2, dist_reward)
# dist_reward *= self.cabinet_have_handle_tensor

# diff_from_success = torch.abs(self.cabinet_dof_tensor_spec[:, :, 0]-self.success_dof_states.view(self.cabinet_num, -1)).view(-1)
# success = (diff_from_success < 0.01)
# success_bonus = success * 3.0
# # print(dist_reward*10, door_reward, action_penalty, success_bonus)
# self.rew_buf = 2*door_reward +3*dist_reward - action_penalty + success_bonus