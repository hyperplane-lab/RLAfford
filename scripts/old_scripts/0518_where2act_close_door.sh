python train.py --task=OneFrankaCabinetPCWhere2act --task_config=cfg/franka_cabinet_PC_partial_cp_map_close.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:1 --sim_device=cuda:1 --cp_device=cuda:1 --seed=6 --experiment=0518where2act_close_door