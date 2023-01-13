# SAT
An autonomous driving agent with a Safety model and the ATtention mechanism in a multi-task framework.

## Requirements

CARLA version == 0.9.10.1
python >= 3.7

Anaconda is suggested for python environment management. 
Create the conda env with  
`conda env create -f environment.yml -n gym_carla`

The conda env can be activated with command  
`conda activate gym_carla`

Before you do anything, please make sure that:

1. the CARLA path in `gym_carla/config/carla_config.py` is set to your CARLA root path
2. the carla client is running correctly,
following carla commands are suggested:  
`./CarlaUE4.sh -opengl -quality-level=Low -ResX=400 -ResY=300 -carla-rpc-port=2000`  
Carla port number is alternative, however you should make sure it coordinates with the one you set in codes.

## Usage

You can run the RL training with following command, and replace `your_projecct_path` with your own path  
```
python your_projecct_path/SAT/rl_agents/td3_old/multi_task/developing/train_ablation.py
```

You can also run the script
`gym_carla/safety_layer/data_collector.py`
to collect data for the safety layer training.

Also, you can run the script
`gym_carla/safety_layer/train_loop.py`
to train the safety layer.

The safety layer model weights in our paper will be released later.

## Citation

If you use this project in your work, please consider citing it with:
```
@article{liu2022multi,
  title={Multi-task safe reinforcement learning for navigating intersections in dense traffic},
  author={Liu, Yuqi and Gao, Yinfeng and Zhang, Qichao and Ding, Dawei and Zhao, Dongbin},
  journal={Journal of the Franklin Institute},
  year={2022},
  publisher={Elsevier}
}
```

