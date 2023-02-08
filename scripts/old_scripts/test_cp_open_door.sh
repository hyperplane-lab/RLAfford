python train.py --task=OneFrankaCabinetPCPartialCPMap --task_config=cfg/franka_cabinet_PC_partial_cp_map_open_handle_test.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/opendoor_config.yaml --headless --rl_device=cuda:3 --sim_device=cuda:3 --cp_device=cuda:3 --test --seed=0 --model_dir=./logs/franka_cabinet_PC_partial_cp_map/ppo_pc_pure/ppo_pc_pure_cp_open_352_cyp_nolog_klfixed_128feat_seed6/model_3600.pt