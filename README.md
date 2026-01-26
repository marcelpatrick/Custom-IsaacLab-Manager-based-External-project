# Custom-IsaacLab-Manager-based-External-project

# Pre-requisites:

- Install IsaacLab and create a python environment: https://github.com/marcelpatrick/IsaacSim-IsaacLab-installation-for-Windows-Easy-Tutorial?search=1
- Create an External Project: https://github.com/marcelpatrick/create-a-new-external-isaaclab-project/blob/main/README.md

# 0. Setup

- Activate your environment: using "env_isaaclab" in this example: `conda activate env_isaaclab`
- In your environments root folder, run `code MyIsaacLabProject2` to open it on VSCode.
- Run a standard task that already came with the IsaacLab project as a test. In Anaconda Prompt terminal, inside your env root folder, run the cartpole task `python MyIsaacLabProject2/scripts/rsl_rl/train.py --task=Template-Myisaaclabproject2-v0`

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

Change the reward function: 

- File `myisaaclabproject2_env_cfg.py`, path: `C:\Users\myali\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\myisaaclabproject2\myisaaclabproject2_env_cfg.py`

Default parameters: 
```py
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )
```

New Parameters (doubling penalties)
```py
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    # Default: terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-4.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        # Default: weight=-1.0,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        # Default: weight=-0.01,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        # Default: weight=-0.005,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )
```

New Traing results
```
                        Learning iteration 49/50

                       Computation: 80882 steps/s (collection: 0.664s, learning 0.146s)
             Mean action noise std: 0.37
          Mean value_function loss: 0.0003
               Mean surrogate loss: 0.0029
                 Mean entropy loss: 0.4241
                       Mean reward: 4.79
               Mean episode length: 297.57
              Episode_Reward/alive: 0.9936
        Episode_Reward/terminating: -0.0001
           Episode_Reward/pole_pos: -0.0108
           Episode_Reward/cart_vel: -0.0131
           Episode_Reward/pole_vel: -0.0083
      Episode_Termination/time_out: 0.9174
Episode_Termination/cart_out_of_bounds: 0.0830
--------------------------------------------------------------------------------
                   Total timesteps: 3276800
                    Iteration time: 0.81s
                      Time elapsed: 00:00:38
                               ETA: 00:00:00
```

#2. Run a Different training task
- Create a new project with a rl library that contaings Isaac-Humanoid-v0 config files
    -> *********** PAREI AQUI

Claude:
```
That's an excellent question, and I want to give you the most accurate answer. Let me search for more specific guidance on this, because the documents you've provided cover creating new tasks from scratch, but not explicitly how to import existing Isaac Lab tasks into an external project.Looking at your past conversations, I can see you've worked through creating external projects and understand the structure well. However, the documents provided don't explicitly cover importing existing Isaac Lab tasks into an external project—they focus on creating new ones from scratch.

Let me be direct about the cleanest approach based on Isaac Lab's architecture, even though I should verify details:

**The conceptually cleanest way would be:**

1. **Find Isaac-Humanoid-v0's definition in Isaac Lab's source code.** This task exists somewhere in the Isaac Lab repository you have installed locally. It will have config files (`*_env_cfg.py`) and environment files (`*_env.py`).

2. **Copy those files into your external project's task structure.** This means placing them in your project's equivalent location:
   ```
   source/YourProject/YourProject/tasks/manager/humanoid/
   ├── __init__.py
   ├── humanoid_env.py
   ├── humanoid_env_cfg.py
   └── agents/
   ```

3. **Register it in the gym registry.** In your `__init__.py`, call `gym.register()` pointing to the humanoid environment and config, just like the template does for the placeholder task.

4. **Update your training scripts to reference it.** Modify `scripts/rsl_rl/train.py` (or whichever RL library you chose) to include the humanoid task in its imports.

This keeps your external project clean and self-contained—all the config files you might want to modify are right there in your project folder, version-controlled separately from Isaac Lab.

However, I want to give you the exact file paths and code snippets, which requires knowing precisely where Isaac-Humanoid-v0 is defined in your Isaac Lab installation. Would you be willing to run a quick command to help me locate it, or would you prefer I search the official Isaac Lab documentation for the exact answer?
```

  
- Run a different training task (different environment, different robot) using the same parameters you just modified.
- In this example, run the Humanoid task" `python MyIsaacLabProject2/scripts/rsl_rl/train.py --task=Isaac-Humanoid-v0`
