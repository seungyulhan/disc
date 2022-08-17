# Dimension-Wise Importance Sampling Weight Clipping

This repository is an implementation of Dimension-Wise Importance Sampling Weight Clipping for Sample-Efficient Reinforcement Learning (ICML 2019)
```
@article{han2019dimension,
  title={Dimension-Wise Importance Sampling Weight Clipping for Sample-Efficient Reinforcement Learning},
  author={Han, Seungyul and Sung, Youngchul},
  journal={arXiv preprint arXiv:1905.02363},
  year={2019}
}
```

## Dependencies
The implementation is based on Open AI baselines (https://github.com/openai/baselines)

It requires Python 3.*/Tensorflow.

## Local Installation

1.Install Anaconda & Mujoco version 2.1 (https://github.com/openai/mujoco-py#install-mujoco)

2.Unzip disc.zip into your installation path

    '''
    cd <installation_path>
    unzip disc.zip
    cd disc
    '''
    
3.Create environment

    '''
    sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libglew-dev patchelf
    conda create -n disc python=3.6
    conda activate disc
    pip install tensorflow==1.4 gym==0.15.4 'mujoco-py<2.2,>=2.1'
    python setup.py install
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    '''

## Training on Mujoco tasks
    '''
    cd <installation_path>/disc
    source activate disc
    python -m baselines.run_disc --env=Humanoid-v2 --num_timesteps 1e7 --log_dir ./Results/Humanoid-v2
    '''



