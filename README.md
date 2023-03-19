# RLAfford

Official Implementation of "RLAfford: End-to-end Affordance Learning with Reinforcement Learning" ICRA 2023

- [Webpage](https://sites.google.com/view/rlafford/)
- [ArXiv](https://arxiv.org/pdf/2209.12941.pdf)

## Introduction

Learning to manipulate 3D articulated objects in an interactive environment has been challenging in reinforcement learning (RL) studies. It is hard to train a policy that can generalize over different objects with vast semantic categories, diverse shape geometry, and versatile functionality. 

Visual affordance provides object-centric information priors that offer actionable semantics for objects with movable parts. For example, an effective policy should know the pulling force on the handle to open a door. 

Nevertheless, how to learn affordance in an end-to-end fashion within the RL process is unknown. In this study, we fill such a research gap by designing algorithms that can automatically learn affordance semantics through a *contact prediction* process. 

The contact predictor allows the agent to learn the affordance information (*i.e.*, where to act for the robotic arm on the object) from previous manipulation experience, and such affordance semantics then helps the agent learn effective policies through RL updates.   
We use our framework on several downstream tasks. The experimental result and analysis demonstrate the effectiveness of end-to-end affordance learning.

## Requirements
We test our code in NVIDIA-driver version $\geq$ 515, cuda Version $\geq$ 11.7 and  python $\geq$ 3.8 environment can run successfully, if the version is not correct may lead to errors, such as `segmentation fault`.

Some dependencies can be installed by

```sh
pip install -r ./requirements.txt
```

### [Isaac Gym](https://developer.nvidia.com/isaac-gym)

Our framework is implemented on Isaac Gym simulator, the version we used is Preview Release 4. You may encounter errors in installing packages, most solutions can be found in the official docs.

### [Pointnet2](https://github.com/daerduoCarey/where2act/tree/main/code)

Install pointnet++ manually.

```sh
cd {the dir for packages}
git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
cd Pointnet2_PyTorch
# [IMPORTANT] comment these two lines of code:
#   https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
pip install -r requirements.txt
pip install -e .
```

Finally, run the following to install other packages.

```sh
# make sure you are at the repository root directory
pip install -r requirements.txt
```

### [Maniskill-Learn](https://github.com/haosulab/ManiSkill-Learn)

```sh
cd {the dir for packages}
git clone https://github.com/haosulab/ManiSkill-Learn.git
cd ManiSkill-Learn/
pip install -e .
```

## Dataset Preparation

Download the dataset from [google drive](https://drive.google.com/drive/folders/1FyTuz17uSmAbVSmJUbgb-7OgRM5TalCK?usp=sharing) and extract it. Move the `asset` folder to the root of this project. The dataset includes objects from SAPIEN dataset along with additional information processed by us. Code to prepare the dataset can be accessed in [this github repo](https://github.com/boshi-an/SapienDataset).

## Reproduce the Results

Once the dataset is ready, you will be able to run the whole training and testing process using the command in [Experiments.md](https://github.com/hyperplane-lab/RLAfford/blob/main/Experiments.md).

## Draw the Pointcloud

We used Mitsuba3 to draw pointcloud and affordance map. Mitsuba provided beautiful visualizations. Scripts can be accesed in the repo [Visualization](https://github.com/GengYiran/Draw_PointCloud) .

## Cite

```latex
@article{geng2022end,
  title={End-to-End Affordance Learning for Robotic Manipulation},
  author={Geng, Yiran and An, Boshi and Geng, Haoran and Chen, Yuanpei and Yang, Yaodong and Dong, Hao},
  journal={International Conference on Robotics and Automation (ICRA)},
  year={2023}
}
```

Feel free to contact hao.dong@pku.edu.cn for collaboration
