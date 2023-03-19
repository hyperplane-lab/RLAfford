# Experiments


## Close Door
#### State
python train.py --task=OneFrankaCabinet --task_config=cfg/franka_cabinet_state_close.yaml --algo=ppo --algo_config=cfg/ppo/config.yaml --rl_device=cuda:0 --sim_device=cuda:0 --pipeline=cpu --seed=0
#### Partial
python train.py --task=OneFrankaCabinetPCPartial --task_config=cfg/franka_cabinet_PC_partial_cloud_close.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --seed=0
python train.py --task=OneFrankaCabinetPCPartial --task_config=cfg/franka_cabinet_PC_partial_cloud_close_test.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --test --seed=0
#### CP Map 
python train.py --task=OneFrankaCabinetPCPartialCPMap --task_config=cfg/franka_cabinet_PC_partial_cp_map_close.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --cp_device=cuda:0 --seed=0
python train.py --task=OneFrankaCabinetPCPartialCPMap --task_config=cfg/franka_cabinet_PC_partial_cp_map_close_test.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --cp_device=cuda:0 --test --seed=0
#### Where2act Map
python train.py --task=OneFrankaCabinetPCWhere2act --task_config=cfg/franka_cabinet_PC_partial_where2act_close.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cpu --sim_device=cpu --seed=0
python train.py --task=OneFrankaCabinetPCWhere2act --task_config=cfg/franka_cabinet_PC_partial_where2act_close_test.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cpu --sim_device=cpu --test --seed=0

## Open Door
#### State
python train.py --task=OneFrankaCabinet --task_config=cfg/franka_cabinet_state_open_handle.yaml --algo=ppo --algo_config=cfg/ppo/opendoor_config.yaml --rl_device=cuda:0 --sim_device=cuda:0 --pipeline=cpu --seed=0
#### Partial
python train.py --task=OneFrankaCabinetPCPartial --task_config=cfg/franka_cabinet_PC_partial_cloud_open_handle.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/opendoor_config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --seed=0
python train.py --task=OneFrankaCabinetPCPartial --task_config=cfg/franka_cabinet_PC_partial_cloud_open_handle_test.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/opendoor_config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --test --seed=0
#### CP Map
python train.py --task=OneFrankaCabinetPCPartialCPMap --task_config=cfg/franka_cabinet_PC_partial_cp_map_open_handle.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/opendoor_config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --cp_device=cuda:0 --seed=0
python train.py --task=OneFrankaCabinetPCPartialCPMap --task_config=cfg/franka_cabinet_PC_partial_cp_map_open_handle_test.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/opendoor_config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --cp_device=cuda:0 --test --seed=0
#### Where2act Map
python train.py --task=OneFrankaCabinetPCWhere2act --task_config=cfg/franka_cabinet_PC_partial_where2act_open.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/opendoor_config.yaml --headless --rl_device=cpu --sim_device=cpu --seed=0
python train.py --task=OneFrankaCabinetPCWhere2act --task_config=cfg/franka_cabinet_PC_partial_where2act_open_test.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/opendoor_config.yaml --headless --rl_device=cpu --sim_device=cpu --test --seed=0

## Close Drawer
#### State
python train.py --task=OneFrankaCabinet --task_config=cfg/franka_drawer_state_close.yaml --algo=ppo --algo_config=cfg/ppo/config.yaml --rl_device=cuda:0 --sim_device=cuda:0 --seed=0 --pipeline=cpu
#### Partial
python train.py --task=OneFrankaCabinetPCPartial --task_config=cfg/franka_drawer_PC_partial_cloud_close.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --seed=0
python train.py --task=OneFrankaCabinetPCPartial --task_config=cfg/franka_drawer_PC_partial_cloud_close_test.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --test --seed=0
#### CP Map
python train.py --task=OneFrankaCabinetPCPartialCPMap --task_config=cfg/franka_drawer_PC_partial_cp_map_close.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --cp_device=cuda:0 --seed=0
python train.py --task=OneFrankaCabinetPCPartialCPMap --task_config=cfg/franka_drawer_PC_partial_cp_map_close_test.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --cp_device=cuda:0 --test --seed=0
#### Where2act Map
python train.py --task=OneFrankaCabinetPCWhere2act --task_config=cfg/franka_drawer_PC_partial_where2act_close.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cpu --sim_device=cpu --seed=0
python train.py --task=OneFrankaCabinetPCWhere2act --task_config=cfg/franka_drawer_PC_partial_where2act_close_test.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cpu --sim_device=cpu --test --seed=0

## Open Drawer
#### State
python train.py --task=OneFrankaCabinet --task_config=cfg/franka_drawer_state_open_handle.yaml --algo=ppo --algo_config=cfg/ppo/config.yaml --rl_device=cuda:0 --sim_device=cuda:0 --pipeline=cpu --seed=0
#### Partial
python train.py --task=OneFrankaCabinetPCPartial --task_config=cfg/franka_drawer_PC_partial_cloud_open_handle.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --seed=0
python train.py --task=OneFrankaCabinetPCPartial --task_config=cfg/franka_drawer_PC_partial_cloud_open_handle_test.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --test --seed=0
#### CP Map
python train.py --task=OneFrankaCabinetPCPartialCPMap --task_config=cfg/franka_drawer_PC_partial_cp_map_open_handle.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --cp_device=cuda:0 --seed=0
python train.py --task=OneFrankaCabinetPCPartialCPMap --task_config=cfg/franka_drawer_PC_partial_cp_map_open_handle_test.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --cp_device=cuda:0 --test --seed=0
#### Where2act Map
python train.py --task=OneFrankaCabinetPCWhere2act --task_config=cfg/franka_drawer_PC_partial_where2act_open.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cpu --sim_device=cpu --seed=0
python train.py --task=OneFrankaCabinetPCWhere2act --task_config=cfg/franka_drawer_PC_partial_where2act_open_test.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cpu --sim_device=cpu --test --seed=0

## Chair
#### State
python train.py --task=TwoFrankaChair --task_config=cfg/franka_chair_state_push.yaml --algo=ppo --algo_config=cfg/ppo/config.yaml --rl_device=cuda:0 --sim_device=cuda:0 --seed=0
#### Partial
python train.py --task=TwoFrankaChairPCPartial --task_config=cfg/franka_chair_PC_partial_cloud_push.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --seed=0
python train.py --task=TwoFrankaChairPCPartial --task_config=cfg/franka_chair_PC_partial_cloud_push_test.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --test --seed=0
#### CP Map
python train.py --task=TwoFrankaChairPCPartialCPMap --task_config=cfg/franka_chair_PC_partial_cp_map_push.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --cp_device=cuda:0 --seed=0
python train.py --task=TwoFrankaChairPCPartialCPMap --task_config=cfg/franka_chair_PC_partial_cp_map_push_test.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/config.yaml --headless --rl_device=cuda:0 --sim_device=cuda:0 --cp_device=cuda:0 --test --seed=0

## Pick and Place
#### State
python train.py --task=PAPRaw --task_config=cfg/pap_raw.yaml --algo=ppo --algo_config=cfg/ppo/pap_config.yaml --rl_device=cuda:0 --sim_device=cuda:0 --pipeline=gpu --graphics_device_id=0 --headless --seed=0

#### Partial 
python train.py --task=PAPPartial --task_config=cfg/pap_partial.yaml --algo=ppo_pc_pure --algo_config=cfg/ppo_pc_pure/pap_config.yaml --rl_device=cuda:0 --sim_device=cuda:0 --headless --seed=0


## Other tasks will update soon!