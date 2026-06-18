# Custom-IsaacLab-Manager-based-External-project

**Video Tutorials**
- Video 1: https://youtu.be/L701v9QuQG4 

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

- New Traing results
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

# 2. Adding a new Task to your project (Humanoid)
- First, make sure that your external project was created with the Reinforcement Learning libraries required to run the desired task: refer to https://github.com/marcelpatrick/create-a-new-external-isaaclab-project#1-create-a-new-external-project-using-the-template-wizard 
- Tasks in these examples run on rsl_rl or rl_games libraries that implement a PPO algorithm.
- If you try to run a task that requires a library not installed in your project, it will error. 

## 2.1. Running a Different training task with Default Parameters

<summary>Running a different task with default parameters (Expand)</summary>
  <details>

- When you cloned the main IsaacLab project to your local machine, it already came with standard built-in tasks that are saved locally to your disk inside the IsaacLab folder.
- Your external project does not contain these tasks/files. But you can run them from your external project with command `(env_isaaclab) C:\Users\[YOUR USER]\MyIsaacLabProject2>python scripts/rsl_rl/train.py --task=[TASK NAME]`
- This is because when you install the IsaacLab cloned project and your external project with `pip install -e` (https://github.com/marcelpatrick/create-a-new-external-isaaclab-project#2-install-the-project) you make these projects importable in the same Python env.
- Also, these two projects conform to the Gymnasium standard that allows their tasks to be called and ran in the same manner from different projects. 
  - Gymnasium is a Python library that defines (and enforces) a standard interface for reinforcement learning environments and allows us to name tasks.
    - The standard it enforces makes so that every task (no matter how different) always has the same main components (eg.reset(), step()methods), allowing them to be runnable by the same script (train.py)
    - The naming functionality allows train.py to find and run tasks by their assigned names.
- Your external project will then access whatever built-in tasks you name and run them with their default parameters. With this approach, it is not possible to modify the task parameters 
- These are some of the other built-in tasks you can run from your external project. From inside your external project folder `(env_isaaclab) C:\Users\[YOUR USER]\MyIsaacLabProject2>` run:
  - `python scripts/rsl_rl/train.py --task=Isaac-Humanoid-v0`
  - `python scripts/rsl_rl/train.py --task=Isaac-Quadcopter-Direct-v0`
  - `python scripts/rsl_rl/train.py --task=Isaac-Ant-v0`
  - `python scripts/rsl_rl/train.py --task=Isaac-Velocity-Rough-Unitree-A1-Play-v0`
 - Note that you just need to switch their names and the same command and script (train.py) are capable of running them. Again, this is because they were all registered in the same python env (with `pip install -e`) and they all follow Gymnasium standards.
 - Benefit: The master project (IsaacLab) remains safe and untouched. Your project stays a safe sandbox. Your custom edits, configs, and experiments live only in MyIsaacLabProject. You can break, rename, or delete things freely without ever touching the shared IsaacLab install that everything else depends on. Built-in tasks give you a ready-made baseline to test or compare against, and running them from your project lets you use them as a starting point — then copy the one task you actually want to modify it, while still being able to run the unmodified baseline task from the IsaacLab master project, for comparison.

  </details>
  
## 2.2. Run a Different training task with your Custom Parameters

<summary>Run a different task with custom parameters (Expand)</summary>
  <details>

### 2.2.1.Setting Up a New Task in Your External Project
- Because the project generated by the Template Generator only comes with the Cartpole task by default, you are going to have to copy any new tasks into your project from the original [IsaacLab](https://github.com/isaac-sim/IsaacLab) git repo. 

**Task: `My-Humanoid-v0`**

**FILES TO COPY**
Files to Copy (from the [IsaacLab](https://github.com/isaac-sim/IsaacLab) original project) into your external project

1. `__init__.py` (under "humanoid" folder):
- humanoid task's init - registers My-Humanoid-v0 within Gymnasium (`gym.register()`) so Isaac Lab can find it.
- path: `C:\Users\[YOUR USER]\isaaclab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\humanoid\__init__.py`

2. `humanoid_env_cfg.py`: Rewards
- Contains all the environment configuration including MDP parameters: rewards, observations, terminations etc.
- path: `C:\Users\[YOUR USER]\isaaclab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\humanoid\humanoid_env_cfg.py`

3. `rsl_rl_ppo_cfg.py`: Training
- Contains the RL training parameters.
- path: `C:\Users\[YOUR USER]\isaaclab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\humanoid\agents\rsl_rl_ppo_cfg.py`

4. `__init__.py`:
- Either copy or create an empty __init__.py file
- This file only exists to tell the compiler that this "agent" folder is a Python package so imports can work. Without it, Python won't recognize agents/ as a module and imports will fail.
- path: `C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\humanoid\agents\__init__.py`


**WHERE TO COPY**
- In your external project, replicate this file structure:
```
source/MyIsaacLabProject2/MyIsaacLabProject2/tasks/manager_based/
│
├── __init__.py                      ← CURRENT, MODIFY: add import for humanoid
│
├── myisaaclabproject2/              ← CURRENT, KEEP (template placeholder)
│   ├── __init__.py
│   ├── myisaaclabproject2_env_cfg.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── (agent config files)
│   └── mdp/
│       ├── __init__.py
│       └── (mdp files)
│
└── humanoid/                        ← NEW FOLDER: create this
    │
    ├── __init__.py                  ← COPY from Isaac Lab, MODIFY paths
    │
    ├── humanoid_env_cfg.py          ← COPY from Isaac Lab, MODIFY reward params
    │
    └── agents/                      ← NEW FOLDER: create this
        │
        ├── __init__.py              ← COPY or CREATE an empty __init__.py file
        │
        └── rsl_rl_ppo_cfg.py        ← COPY from Isaac Lab, MODIFY training params
```

**CHANGES NEEDED**

1. Create new Folders
- "humanoid". Path: `source/MyIsaacLabProject2/MyIsaacLabProject2/tasks/manager_based/humanoid/`
- "agents" (inside "humanoid"). Path: `source/MyIsaacLabProject2/MyIsaacLabProject2/tasks/manager_based/humanoid/agents/`

2. Modify existing files
- Manager_based folder `__init__.py`:
- Action: Add humanoid task import
- Path in your external project: `C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\__init__.py`

```py
import gymnasium as gym  # noqa: F401

from . import humanoid  # noqa: F401       <<< ADDED THIS LINE
```

3. Create empty `__init__.py` file inside "agents" folder
- Original Path: `C:\Users\[YOUR USER]\isaaclab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\humanoid\agents\__init__.py`
- Path in your external project: `C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\humanoid\agents\__init__.py`

4. Copy Files
   
- **4.1. HUMANOID TASK'S INIT** (`__init__.py`)
Action: Copy from original project and modify
- Rename: Add your project's name in `id`
  -> **Your project's name must contain "Template-" before the name otherwise it will not register on Gymnasium and you will not be able to run it with your custom parameters** 
- Replace `f"{agents.__name__}` with the entire paths `isaaclab_tasks.manager_based.classic.humanoid.agents` - since we didn't import the entire agents folder. Use full path strings for the training configs pointing to Isaac Lab's original files
- Delete `"rl_games_cfg_entry_point": "isaaclab_tasks.manager_based.classic.humanoid.agents:rl_games_ppo_cfg.yaml"` - You can delete this line if you're only using `rsl_rl` for training. The entry points are only needed for the RL libraries you actually use
- Original path: `C:\Users\[YOUR USER]\isaaclab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\humanoid\__init__.py`
- Path in your external project: `C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\humanoid\__init__.py`

<details>
  <summary>Click to expand __init__.py</summary>
  
```py
import gymnasium as gym

from . import agents  # ← KEEP THIS since you have a local agents folder

gym.register(
    id="Template-My-Isaac-Humanoid-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_env_cfg:HumanoidEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidPPORunnerCfg",  # ← Uses YOUR local config
        "rl_games_cfg_entry_point": "isaaclab_tasks.manager_based.classic.humanoid.agents:rl_games_ppo_cfg.yaml",  # ← Still uses Isaac Lab's
    },
)
```
</details>

- **4.2: ENVIRONMENT CONFIG** (`humanoid_env_cfg.py`)
  Action: Copy from the original project — no modifications needed (except if you want to edit reward parameters)
  Original Path: `C:\Users\[YOUR USER]\isaaclab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\humanoid\humanoid_env_cfg.py`
  Path in your external project: `C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\humanoid\humanoid_env_cfg.py`

<details>
  <summary>Click to expand humanoid_env_cfg</summary>

```py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

from isaaclab_assets.robots.humanoid import HUMANOID_CFG  # isort:skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a humanoid robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )

    # robot
    robot = HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
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
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot", "right_foot"])},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving forward
    progress = RewTerm(func=mdp.progress_reward, weight=1.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    # (2) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    # (3) Reward for non-upright posture
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})
    # (4) Reward for moving in the right direction
    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus, weight=0.5, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
    )
    # (5) Penalty for large action commands
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    # (6) Penalty for energy consumption
    energy = RewTerm(
        func=mdp.power_consumption,
        weight=-0.005,
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
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio,
        weight=-0.25,
        params={
            "threshold": 0.98,
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


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.8})


@configclass
class HumanoidEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Humanoid walking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0


```
</details>

- **4.3. TRAINING PARAMETERS** (`rsl_rl_ppo_cfg.py`)
  Action: Copy from the original project, no changes needed (except if you want to edit training parameters)
  Original Path: `C:\Users\[YOUR USER]\isaaclab\source\isaaclab_tasks\isaaclab_tasks\manager_based\classic\humanoid\agents\rsl_rl_ppo_cfg.py`

<details>
  <summary>click here to expand rsl_rl_ppo_cfg.py</summary>
  
```py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class HumanoidPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50

    # CUSTOMIZE YOUR EXPERIMEN NAME:
    experiment_name = "my_humanoid"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```
</details>

</details>

# 3. Modify Parameters

<summary>Modify Parameters (Expand)</summary>
  <details>
    
- First I am going to run the `Template-My-Isaac-Humanoid-v0` task with its original parameters to have a baseline to compare results after changes.
  - However, for testing purposes, I will just modify the training parameter `max_iterations` in `rsl_rl_ppo_cfg.py` from 1000 to 50 for the simulation to finish faster.
 
Baseline Results

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

## 3.1: Modify Reward Parameters (MDP)
File: `humanoid_env_cfg`
Path: `C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\humanoid\humanoid_env_cfg.py`

<summary>New Reward Parameters</summary>
  <details>
    
    ```py
          # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
    # All rights reserved.
    #
    # SPDX-License-Identifier: BSD-3-Clause
    
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import RewardTermCfg as RewTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass
    
    import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
    
    from isaaclab_assets.robots.humanoid import HUMANOID_CFG  # isort:skip
    
    
    ##
    # Scene definition
    ##
    
    
    @configclass
    class MySceneCfg(InteractiveSceneCfg):
        """Configuration for the terrain scene with a humanoid robot."""
    
        # terrain
        terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            debug_vis=False,
        )
    
        # robot
        robot = HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
        # lights
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
    
    
    ##
    # MDP settings
    ##
    
    
    @configclass
    class ActionsCfg:
        """Action specifications for the MDP."""
    
        joint_effort = mdp.JointEffortActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            scale={
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
        )
    
    
    @configclass
    class ObservationsCfg:
        """Observation specifications for the MDP."""
    
        @configclass
        class PolicyCfg(ObsGroup):
            """Observations for the policy."""
    
            base_height = ObsTerm(func=mdp.base_pos_z)
            base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
            base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
            base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
            base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
            base_up_proj = ObsTerm(func=mdp.base_up_proj)
            base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
            joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
            joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
            feet_body_forces = ObsTerm(
                func=mdp.body_incoming_wrench,
                scale=0.01,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=["left_foot", "right_foot"])},
            )
            actions = ObsTerm(func=mdp.last_action)
    
            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True
    
        # observation groups
        policy: PolicyCfg = PolicyCfg()
    
    
    @configclass
    class EventCfg:
        """Configuration for events."""
    
        reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={"pose_range": {}, "velocity_range": {}},
        )
    
        reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.2, 0.2),
                "velocity_range": (-0.1, 0.1),
            },
        )
    
    
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
    
    
    @configclass
    class TerminationsCfg:
        """Termination terms for the MDP."""
    
        # (1) Terminate if the episode length is exceeded
        time_out = DoneTerm(func=mdp.time_out, time_out=True)
        # (2) Terminate if the robot falls
        torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.8})
    
    
    @configclass
    class HumanoidEnvCfg(ManagerBasedRLEnvCfg):
        """Configuration for the MuJoCo-style Humanoid walking environment."""
    
        # Scene settings
        scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=True)
        # Basic settings
        observations: ObservationsCfg = ObservationsCfg()
        actions: ActionsCfg = ActionsCfg()
        # MDP settings
        rewards: RewardsCfg = RewardsCfg()
        terminations: TerminationsCfg = TerminationsCfg()
        events: EventCfg = EventCfg()
    
        def __post_init__(self):
            """Post initialization."""
            # general settings
            self.decimation = 2
            self.episode_length_s = 16.0
            # simulation settings
            self.sim.dt = 1 / 120.0
            self.sim.render_interval = self.decimation
            self.sim.physx.bounce_threshold_velocity = 0.2
            # default friction material
            self.sim.physics_material.static_friction = 1.0
            self.sim.physics_material.dynamic_friction = 1.0
            self.sim.physics_material.restitution = 0.0


    ``` 
  </details>

## 3.2: Modify Training Paramters
File: `rsl_rl_ppo_cfg.py`
Path: `C:\Users\[YOUR USER]\MyIsaacLabProject2\source\MyIsaacLabProject2\MyIsaacLabProject2\tasks\manager_based\humanoid\agents\rsl_rl_ppo_cfg.py`

<summary>New Training Parameters</summary>
  <details>
    
    ```py
          # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
    # All rights reserved.
    #
    # SPDX-License-Identifier: BSD-3-Clause
    
    from isaaclab.utils import configclass
    
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
    
    
    @configclass
    class HumanoidPPORunnerCfg(RslRlOnPolicyRunnerCfg):
        # Meaning: how many simulator steps you collect per environment before each policy update.
        # Increasing makes it wait to gather more observations before updating the policy, which can improve learning stability, but also increases memory usage and training time per update.
        num_steps_per_env = 64 
        # Meaning: how many training update cycles PPO will run total.
        # More iterations gives it more chances to learn
        max_iterations = 100
        # how often to save checkpoints
        save_interval = 50
        
        # CUSTOMIZE YOUR EXPERIMEN NAME:
        experiment_name = "my_humanoid"
        policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,     # too high = thrashing and falling; too low = not enough exploration to discover walking.
            # Meaning: automatically rescales what the policy “sees” to more consistent ranges
            # improves stability and speed of learning when observations have very different scales
            actor_obs_normalization=True,
            # same as above, but for the value estimator (the part judging how good states are)
            critic_obs_normalization=True,
            actor_hidden_dims=[400, 200, 100],
            critic_hidden_dims=[400, 200, 100],
            activation="elu",
        )
        algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            # Limit on how big the policy update step can be
            # calculates diff in action for each specific action.
            # ensures a smoother learning curve, preventing large updates that could destabilize training.
            clip_param=0.2,
            # Exploring more. extra reward for being a bit random (trying new actions)
            entropy_coef=0.005,
            # how many times you learn from the same collected data per update.
            num_learning_epochs=8,
            num_mini_batches=4,
            # how big each learning step is
            # slightly smaller steps often prevent “learn then collapse” behavior in locomotion
            learning_rate=3e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            # Limit on how big the policy update step can be
            # calculates diff in action provability across all possible actions of that policy
            # ensures a smoother learning curve, preventing large updates that could destabilize training.
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
  
    ```
    
  </details>

  </details>
  
# 4. Run

In your Anaconda Prompt terminal, at the project root (or inside the VSCode project)  `(env_isaaclab) C:\Users\[YOUR USER]\MyIsaacLabProject2>` , run: `python scripts/rsl_rl/train.py --task=Template-My-Isaac-Humanoid-v0`
Exit with `Ctrl C`

Run `python scripts/list_envs.py` to ensure the your environment got registered in Gymnasium. Your project name must appear on the list.
  -> Your project name **must** start with "Template-" (like `Template-My-Isaac-Humanoid-v0`) otherwise it won't register

# 5. Results

***New Results**
```
################################################################################
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

-> As you can see, after the parameter adjustments, the `mean reward` metric increased significantly, showing that training with the new parameters was more successful. 
- However, `Episode_Termination/time_out` shows that 0 episondes lasted until timeout and `Episode_Termination/torso_height` tells us that 100% ended up with the robot falling which indicates that the policy still isn't stable. What is expected due to the low number of iterations (50).
- 

# SIMPLE

1. Create project
2. Copy new task folder from the original IsaacLab project to your project.
3. Rename folder from MyIsaacLabProject to Cartpole
4. TASK'S INIT
   - Path: `"C:\Users\[YOUR USER]\MyIsaacLabProject5\source\MyIsaacLabProject5\MyIsaacLabProject5\tasks\manager_based\ant\__init__.py"`
   - Rename project's name in id -> Your project's name must contain "Template-" before the name otherwise it will not register on Gymnasium and you will not be able to run it with your custom parameters. eg `Template-My-Isaac-Ant-v0`
   - `id="Template-My-Isaac-Ant-v0",      # << RENAME`
5. Rename Experiment
  - Path: "C:\Users\[YOUR USER]\MyIsaacLabProject5\source\MyIsaacLabProject5\MyIsaacLabProject5\tasks\manager_based\ant\agents\rsl_rl_ppo_cfg.py"
  - `experiment_name = "my_ant" # << RENAME`
6. Customize Training Parameters
  - File: `rl_games_ppo_cfg.yaml`. The YAML file will depend on the library you are using. Eg. if you use rsl_rl the file will be `rsl_rl_ppo_cfg`
  - Path: `"C:\Users\[YOUR USER]\MyIsaacLabProject5\source\MyIsaacLabProject5\MyIsaacLabProject5\tasks\manager_based\ant\agents\rl_games_ppo_cfg.yaml"`
  - Example:
```py
    max_epochs: 20
    save_best_after: 10
    save_frequency: 10
```
7. Customize Reward Parameters
 - File: `ant_env_cfg`
 - Path: `"C:\Users\[YOUR USER]\MyIsaacLabProject5\source\MyIsaacLabProject5\MyIsaacLabProject5\tasks\manager_based\ant\ant_env_cfg.py"`
 - Example:
```py

```
8. Read results
 - run: `tensorboard --logdir logs/rl_games/ant/2026-02-03_10-18-27` or select the last created folder inside `logs/rl_games/ant`
 - Open the provided localhost address in your browser


