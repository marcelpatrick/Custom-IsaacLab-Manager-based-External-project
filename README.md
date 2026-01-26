# Custom-IsaacLab-Manager-based-External-project

# Pre-requisites:

- Install IsaacLab and create a python environment: https://github.com/marcelpatrick/IsaacSim-IsaacLab-installation-for-Windows-Easy-Tutorial?search=1
- Create an External Project: https://github.com/marcelpatrick/create-a-new-external-isaaclab-project/blob/main/README.md

# 0. Setup

- Activate your environment: using "env_isaaclab" in this example: `conda activate env_isaaclab`
- In your environments root folder, run `code MyIsaacLabProject2` to open it on VSCode.
- Run a standard task that already came with the IsaacLab project as a test. In Anaconda Prompt terminal, inside your env root folder, run `python MyIsaacLabProject2/scripts/rsl_rl/train.py --task=Template-Myisaaclabproject2-v0`

- Learning results
```
                       Learning iteration 149/150

                       Computation: 106277 steps/s (collection: 0.537s, learning 0.080s)
             Mean action noise std: 0.10
          Mean value_function loss: 0.0012
               Mean surrogate loss: 0.0065
                 Mean entropy loss: -0.8685
                       Mean reward: 4.87
               Mean episode length: 297.93
              Episode_Reward/alive: 0.9936
        Episode_Reward/terminating: -0.0001
           Episode_Reward/pole_pos: -0.0128
           Episode_Reward/cart_vel: -0.0034
           Episode_Reward/pole_vel: -0.0013
      Episode_Termination/time_out: 0.9939
Episode_Termination/cart_out_of_bounds: 0.0061
--------------------------------------------------------------------------------
                   Total timesteps: 9830400
                    Iteration time: 0.62s
                      Time elapsed: 00:01:39
                               ETA: 00:00:00
```

# 1. Modify training settings

- File `rsl_rl_ppo_cfg.py`: `C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\myisaaclabproject2\agents\rsl_rl_ppo_cfg.py`

Default parameters: 
```py
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 150
    save_interval = 50
    experiment_name = "cartpole_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

New parameters
```py
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    # max_iterations = 150
    max_iterations = 50
    save_interval = 50
    experiment_name = "cartpole_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=8,
        num_mini_batches=4,
        # learning_rate=1.0e-3,
        learning_rate=1.0e-2,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

New training results:
```py
                        Learning iteration 49/50

                       Computation: 84981 steps/s (collection: 0.632s, learning 0.139s)
             Mean action noise std: 0.32
          Mean value_function loss: 0.0001
               Mean surrogate loss: 0.0013
                 Mean entropy loss: 0.2871
                       Mean reward: 4.91
               Mean episode length: 300.00
              Episode_Reward/alive: 1.0000
        Episode_Reward/terminating: 0.0000
           Episode_Reward/pole_pos: -0.0082
           Episode_Reward/cart_vel: -0.0058
           Episode_Reward/pole_vel: -0.0036
      Episode_Termination/time_out: 0.9957
Episode_Termination/cart_out_of_bounds: 0.0044
--------------------------------------------------------------------------------
                   Total timesteps: 3276800
                    Iteration time: 0.77s
                      Time elapsed: 00:00:38
                               ETA: 00:00:00
```
