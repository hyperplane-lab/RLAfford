# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.



def process_MultiAgentRL(args, env, config, model_dir=""):

    config["n_rollout_threads"] = env.num_envs
    config["n_eval_rollout_threads"] = env.num_envs

    if args.algo in ["mappo", "happo", "hatrpo", "mappo_pc_pure"]:
        # on policy marl
        from algorithms.algorithms.runner import Runner
        marl = Runner(vec_env=env,
                    config=config,
                    model_dir=model_dir,
                    is_testing = args.test
                    )
    elif args.algo == 'maddpg':
        # off policy marl
        from algorithms.maddpg.runner import Runner
        marl = Runner(vec_env=env,
            config=config,
            model_dir=model_dir
            )

    return marl
