python train.py --task=OneFrankaCabinetPCPartialCPMap --task_config=cfg/franka_drawer_PC_partial_cp_map_open_handle.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/open_drawer_config.yaml --headless --rl_device=cuda:2 --sim_device=cuda:2 --cp_device=cuda:2 --seed=0 --experiment=open_drawer_norew0