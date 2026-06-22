# Custom-IsaacLab-Manager-based-External-project

**Video Tutorials**
- Video 1: https://youtu.be/L701v9QuQG4 - Running a Task Already in Your Project (Cartpole) - Change parameters: Chapter 1.2
- Video 2: https://youtu.be/fq9DRCj9i7E - Running a Different training task with Default Parameters: Chapter 2.1

# Pre-requisites:

- Install IsaacLab and create a python environment: https://github.com/marcelpatrick/IsaacSim-IsaacLab-installation-for-Windows-Easy-Tutorial?search=1
- Create an External Project: https://github.com/marcelpatrick/create-a-new-external-isaaclab-project/blob/main/README.md

# 0. Setup

- Activate your environment (using "env_isaaclab" in this example): `conda activate env_isaaclab`
- If you want to open your external project on VSCode (using "MyIsaacLabProject2" in this example): In your environment's root folder, run `code MyIsaacLabProject2` to open it on VSCode.

# 1. Running a Task Already in Your Project (Cartpole)
- When we create an external project using [Template Generator](https://github.com/marcelpatrick/create-a-new-external-isaaclab-project/blob/main/README.md), it is created with only one task: Cartpole
- Here we will run the Cartpole task that already came with the project

<summary>Running a Task Already in Your Project (Expand)</summary>
  <details>

## 1.1. Run with default parameters:
- Here, we will run Cartpole task with its default training and reward function parameters. 
- In Anaconda Prompt terminal, inside your env root folder, run the cartpole task `python MyIsaacLabProject2/scripts/rsl_rl/train.py --task=Template-Myisaaclabproject2-v0`

<details>
  <summary>Parameters and Results</summary>
  
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
</details>

## 1.2 Change Parameters

### 1.2.1: Change Training Parameters
- Training Parameters are in this File `rsl_rl_ppo_cfg.py`: Path: `C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\myisaaclabproject2\agents\rsl_rl_ppo_cfg.py`
- Or, if you changed the task folder name to `cartpole` in the previous tutorial: Path: `C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\cartpole\agents\rsl_rl_ppo_cfg.py`

<details>
  <summary>Parameters and Results</summary>
  
- Default parameters: 
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
        clip_param=0.2, # how much % the new policy can differ from the old policy, prevents radical updates.
        entropy_coef=0.005, # how much exploration the robot does, how random are the moves it tries, before it starts learning (curiosity coefficient)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3, # how fast the algorithm tries to converge towards the point of minimum error
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

- New parameters
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
In Anaconda Prompt terminal, inside your env root folder `(your environment) C:\Users\[YOUR USER]`, run the cartpole task with `python MyIsaacLabProject2/scripts/rsl_rl/train.py --task=Template-Myisaaclabproject2-v0`

- New training results:
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
</details>

### 1.2.2. Change the Reward Function (MDC) parameters: 

- Reward Function (MDC) parameters are in this File `myisaaclabproject2_env_cfg.py`, path: `C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\myisaaclabproject2\myisaaclabproject2_env_cfg.py`
- Or, if you changed the task folder name to `cartpole` in the previous tutorial: Path: `C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\cartpole\myisaaclabproject2_env_cfg.py`

<details>
  <summary>Parameters and Results</summary>
  
- Default parameters: 
```py
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward: Every single timestep the pole is still up and the cart is still on the rail (not terminated), reward = +1. If higher, it incentivizes the robot to stay alive longer. 
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty: The "game over" penalty. The instant the episode fails (pole falls past the limit, or cart slides off the track)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright: This measures how far the pole's angle is from upright (target: 0.0)
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # (4) Shaping tasks: lower cart velocity: penalizes if cart moves too fast - discourages violent moves
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) Shaping tasks: lower pole angular velocity: penalizes if pole moves too fast - discourages violent moves
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )
```

- New Parameters (doubling penalties)
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
Run again: In Anaconda Prompt terminal, inside your env root folder`(your environment) C:\Users\[YOUR USER]`, run the cartpole task with `python MyIsaacLabProject2/scripts/rsl_rl/train.py --task=Template-Myisaaclabproject2-v0`

- New Training results
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
</details>

</details>

Next, we are going to run different tasks in our external project: 
- First, make sure that your external project was created with the Reinforcement Learning libraries required to run the desired task: refer to https://github.com/marcelpatrick/create-a-new-external-isaaclab-project#1-create-a-new-external-project-using-the-template-wizard 
- Tasks in these examples run on rsl_rl or rl_games libraries that implement a PPO algorithm.
- If you try to run a task that requires a library not installed in your project, it will error. 

# 2 Running different tasks with default parameters

<summary>Running a different task with default parameters (Expand)</summary>
  <details>

- When you cloned the main IsaacLab project to your local machine, it already came with standard built-in tasks that are saved locally to your disk inside the IsaacLab folder.
- Your external project does not contain these tasks/files. But you can run them from your external project with command `(env_isaaclab) C:\Users\[YOUR USER]\MyIsaacLabProject2>python scripts/rsl_rl/train.py --task=[TASK NAME]`
- This is because when you install the IsaacLab cloned project and your external project with `pip install -e` (https://github.com/marcelpatrick/create-a-new-external-isaaclab-project#2-install-the-project) you make these projects importable in the same Python env.
- Also, the tasks in these two projects conform to the Gymnasium standard that allows their tasks to be called and ran in the same manner from different projects. 
  - Gymnasium is a Python library that defines a standard way to code reinforcement learning tasks and allows us to name them.
    - The standard ensures that every task (no matter how different) always has the same main components (`reset(), step()` etc), allowing them to be runnable by the same script (`train.py`)
    - The naming functionality allows `train.py` to find and run tasks by their assigned names.
- Your external project will then access whatever built-in tasks you name and run them with their default parameters. With this approach, it is not possible to modify the task parameters because the task's config files live in the read-only IsaacLab install, not in your project.
  
- These are some of the other built-in tasks you can run from your external project. From inside your external project folder `(env_isaaclab) C:\Users\[YOUR USER]\MyIsaacLabProject2>` run:
  - `python scripts/rsl_rl/train.py --task=Isaac-Humanoid-v0`
  - `python scripts/rsl_rl/train.py --task=Isaac-Quadcopter-Direct-v0`
  - `python scripts/rsl_rl/train.py --task=Isaac-Ant-v0`
  - `python scripts/rsl_rl/train.py --task=Isaac-Velocity-Rough-Unitree-A1-Play-v0`
 - Note that you just need to switch their names and the same command and script (`train.py`) is capable of running them. Again, this is because they were all registered in the same python env (with `pip install -e`) and they all follow Gymnasium standards.
   
 - Benefit: The master project (IsaacLab) remains safe and untouched. Your project stays a safe sandbox. Your custom edits, configs, and experiments live only in MyIsaacLabProject. You can break, rename, or delete things freely without ever touching the shared IsaacLab install that everything else depends on. Built-in tasks give you a ready-made baseline to test or compare against, and running them from your project lets you use them as a starting point — then copy the one task you actually want to modify, while still being able to run the unmodified baseline task from the IsaacLab master project, for comparison.

  </details>
  
# 3. Running different tasks with Custom Parameters

<summary>Run a different task with custom parameters (Expand)</summary>
  <details>

## 3.1. Setting Up a New Task in Your External Project `My-Humanoid-v0`
- Because the project generated by the Template Generator only comes with the Cartpole task by default, you are going to have to copy any new tasks into your project from the original [IsaacLab](https://github.com/isaac-sim/IsaacLab) git repo. 
    
1. Copy the humanoid folder from the IsaacLab master project: `C:\Users\[YOUR USER]\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\humanoid`, into your external project's task folder: `C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based`
2. Add this task as an import inside the Project's `__init__.py`: in the manager_based folder inside your external project: `"C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\__init__.py"`.
     ```py
     import gymnasium as gym  # noqa: F401
     
     from . import humanoid # <<< ADD THIS LINE
     ```
    
3. Rename: Add "Template" to your project's nameID inside the task's (Humanoid) `__init__.py`: 
  - Open `__init__.py`: `"C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\humanoid\__init__.py"` 
    ```py
    gym.register(
    id="Template-Isaac-Humanoid-v0",   # <<< Add "Template-" prefix to avoid name collision with OpenAI Gym Humanoid-v2
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_env_cfg:HumanoidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    ```
  -> **Your project's name must contain "Template-"** before the name; otherwise, it will not register on Gymnasium and you will not be able to run it with your custom parameters.
  
4. Run the copied Humanoid task with default parameters to test it
- Run `python scripts/list_envs.py` to ensure that your environment has been registered in Gymnasium. Your project nameID must appear on the list containing "Template-" (eg: Template-Isaac-Humanoid-v0) otherwise, it won't register on Gymnasium.
  
- In your Anaconda Prompt terminal, activate your Python environment and go to the project root: `(env_isaaclab) C:\Users\[YOUR USER]\MyIsaacLabProject2>`, run: `python scripts/rsl_rl/train.py --task=Template-Isaac-Humanoid-v0`. Exit with Ctrl C

## 3.2: Modifying Parameters
    
- First I am going to run the `Template-My-Isaac-Humanoid-v0` task with its original parameters to have a baseline to compare results after changes.
  - However, for testing purposes, I will just modify the training parameter `max_iterations` in `rsl_rl_ppo_cfg.py` from 1000 to 50 for the simulation to finish faster.
 
<summary>Baseline Results</summary>

  <details> 

```
        ################################################################################
                            Learning iteration 49/50 
    
                           Computation: 59242 steps/s (collection: 2.063s, learning 0.150s)
                 Mean action noise std: 0.84
              Mean value_function loss: 0.0271
                   Mean surrogate loss: -0.0083
                     Mean entropy loss: 26.0180
                           Mean reward: 2.73
                   Mean episode length: 63.78
               Episode_Reward/progress: 0.0436
                  Episode_Reward/alive: 0.1252
                Episode_Reward/upright: 0.0043
         Episode_Reward/move_to_target: 0.0305
              Episode_Reward/action_l2: -0.0114
                 Episode_Reward/energy: -0.0139
       Episode_Reward/joint_pos_limits: -0.0150
          Episode_Termination/time_out: 0.0000
      Episode_Termination/torso_height: 1.0000
    --------------------------------------------------------------------------------
                       Total timesteps: 6553600
                        Iteration time: 2.21s
                          Time elapsed: 00:01:47
                                   ETA: 00:00:02
```

  </details>

### 3.2.1: Modifying Reward Parameters (MDP)

- File: `humanoid_env_cfg`
- Path: `C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\humanoid\humanoid_env_cfg.py`

<summary>New Reward Parameters</summary>
  <details>
    
  ```py
    
    @configclass
    class RewardsCfg:
        """Reward terms for the MDP."""
    
        # (1) Reward for moving forward toward the target
        # increasing rewards makes robot try walking sooner towards the target instead of just ballancing in place
        progress = RewTerm(func=mdp.progress_reward, weight=1.5, params={"target_pos": (1000.0, 0.0, 0.0)})
        # (2) Stay alive bonus
        # lowering it prevents the robot from “playing it safe” by barely moving just to stay alive
        alive = RewTerm(func=mdp.is_alive, weight=0.5)
        # (3) gives points for keeping the body upright; the threshold is a marging
        # Increases makes the robot stabilize earlier
        upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.5, params={"threshold": 0.95})
        # (4) Reward for moving in the right direction
        # Increasing it nudges the robot to go straight instead of shuffling sideways or spinning
        move_to_target = RewTerm(
            func=mdp.move_to_target_bonus, weight=0.2, params={"threshold": 0.9, "target_pos": (1000.0, 0.0, 0.0)}
        )
        # (5) Penalty for large action commands: subtracts points when the robot uses very large/jerky control commands.
        # Lowering it allows for more exploration
        action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)
        # (6) Penalty for energy consumption
        # Reducing allows more energy to learn faster and make more exploratory movements
        energy = RewTerm(
            func=mdp.power_consumption,
            weight=-0.001,
            params={
                "gear_ratio": {
                    ".*_waist.*": 67.5,
                    ".*_upper_arm.*": 67.5,
                    "pelvis": 67.5,
                    ".*_lower_arm": 45.0,
                    ".*_thigh:0": 45.0,
                    ".*_thigh:1": 135.0,
                    ".*_thigh:2": 45.0,
                    ".*_shin": 90.0,
                    ".*_foot.*": 22.5,
                }
            },
        )
        # (7) Penalty for reaching close to joint limits
        # Penalty for bending joints too close to their limit angles; threshold is a margin. 
        # Reducing lets the robot use its full range while learning; later it can tighten it for safety
        joint_pos_limits = RewTerm(
            func=mdp.joint_pos_limits_penalty_ratio,
            weight=-0.05,
            params={
                "threshold": 0.95,
                "gear_ratio": {
                    ".*_waist.*": 67.5,
                    ".*_upper_arm.*": 67.5,
                    "pelvis": 67.5,
                    ".*_lower_arm": 45.0,
                    ".*_thigh:0": 45.0,
                    ".*_thigh:1": 135.0,
                    ".*_thigh:2": 45.0,
                    ".*_shin": 90.0,
                    ".*_foot.*": 22.5,
                },
            },
        )

   ```
  </details>

### 3.2.2: Modifying Training Parameters
- File: `rsl_rl_ppo_cfg.py`
- Path: `C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\humanoid\agents\rsl_rl_ppo_cfg.py`

<summary>New Training Parameters: (Expand)</summary>
  <details>
    
  ```py
        
          @configclass
          class HumanoidPPORunnerCfg(RslRlOnPolicyRunnerCfg):
              num_steps_per_env = 64 # <<< Changed from 32 to 64
                  # Meaning: how many simulator steps you collect per environment before each policy update.
                  # Increasing makes it wait to gather more observations before updating the policy, which can improve learning stability, 
              max_iterations = 50 # <<< Changed from 1000 to 50
                  # Meaning: how many training update cycles PPO will run total.
                  # More iterations gives it more chances to learn. Decreased to 100 just for this test to finish quicker
              save_interval = 50
                  # Meaning: how often to save checkpoints
              experiment_name = "my_humanoid" # <<< CUSTOMIZE YOUR EXPERIMENT NAME:
              policy = RslRlPpoActorCriticCfg(
                  init_noise_std=1.0, # too high = falls faster; too low = not enough exploration to discover walking.
                  actor_obs_normalization=True, # <<< Changed to True. improves stability and speed of learning when observations have very different scales
                  critic_obs_normalization=True, # <<< Changed to True. same as above, but for the value estimator (the part judging how good states are)
                  actor_hidden_dims=[400, 200, 100],
                  critic_hidden_dims=[400, 200, 100],
                  activation="elu",
              )
              algorithm = RslRlPpoAlgorithmCfg(
                  value_loss_coef=1.0,
                  use_clipped_value_loss=True,
                  clip_param=0.2,  # Limit on how big the policy update step can be. Ensures a smoother learning curve, preventing large updates that could destabilize training.
                  entropy_coef=0.005, # <<< Changed from 0 to 0.005. Extra reward for exploring more (trying new actions)
                  num_learning_epochs=8, # <<< Changed from 5 to 8. Number of times you learn from the same collected data per update.
                  num_mini_batches=4,
                  learning_rate=3e-4, # <<< Changed from 5.0e-4 to 3e-4.How big each learning step is
                  schedule="adaptive",
                  gamma=0.99,
                  lam=0.95,
                  desired_kl=0.01,
                  max_grad_norm=1.0,
              )
          
  ```
    
  </details>

## 3.3: Run 
- In your Anaconda Prompt terminal, activate your Python environment and go to the project root: `(env_isaaclab) C:\Users\[YOUR USER]\MyIsaacLabProject2>`, run: `python scripts/rsl_rl/train.py --task=Template-Isaac-Humanoid-v0`. Exit with Ctrl C
- In the training and reward metrics in the CLI, you should see "Learning iteration" with the new value x/50, instead of x/1000

# 4. Results

<summary>New Results</summary>

  <details> 
      ```
                            Learning iteration 49/50
    
                           Computation: 20176 steps/s (collection: 12.269s, learning 0.723s)
                 Mean action noise std: 0.95
              Mean value_function loss: 0.1038
                   Mean surrogate loss: -0.0206
                     Mean entropy loss: 28.7003
                           Mean reward: 5.38
                   Mean episode length: 108.27
               Episode_Reward/progress: 0.1557
                  Episode_Reward/alive: 0.2179
                Episode_Reward/upright: 0.0057
         Episode_Reward/move_to_target: 0.0494
              Episode_Reward/action_l2: -0.0272
                 Episode_Reward/energy: -0.0367
       Episode_Reward/joint_pos_limits: -0.0339
          Episode_Termination/time_out: 0.0000
      Episode_Termination/torso_height: 1.0000
    --------------------------------------------------------------------------------
                       Total timesteps: 13107200
                        Iteration time: 12.99s
                          Time elapsed: 00:06:06
                                   ETA: 00:00:07
  ```

  ```
  
  </details>

-> As you can see, after the parameter adjustments, the `mean reward` metric increased significantly, showing that training with the new parameters was more successful. 
- However, `Episode_Termination/time_out` shows that 0 episondes lasted until timeout and `Episode_Termination/torso_height` tells us that 100% ended up with the robot falling which indicates that the policy still isn't stable. What is expected due to the low number of iterations (50).
  



