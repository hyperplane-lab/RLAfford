python train.py --task=OneFrankaCabinetPCPartial --task_config=cfg/franka_cabinet_PC_partial_cloud_open_handle.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/opendoor_config.yaml --headless --rl_device=cuda:1 --sim_device=cuda:1 --seed=1501 --experiment=0517_partial_open_door_rerun