# Solo8 – MuJoCo Playground Extension

## Overview
This project extends the existing *MuJoCo Playground* framework with a custom **Solo8** environment.

The goal is to simulate the **Solo8 quadruped robot** developed by the Open Dynamic Robot Initiative and to train it using reinforcement learning to achieve a stable **trotting gait**.

A trotting gait is characterized by the synchronized movement of diagonal leg pairs (front-left with hind-right and front-right with hind-left), alternating over time and including short flight phases where no foot touches the ground.

---

## Project Structure

### `solo8/`
Main directory containing all project-specific code.

#### Core Files
- `base.py`  
  Provides the base environment (`Solo8Env`) used by all tasks.  
  It handles:
  - Loading MuJoCo models and assets (XML + meshes)
  - Simulation setup (timestep, physics parameters)
  - Actuator configuration (PD control gains)
  - Sensor access (velocity, IMU, foot positions, etc.)

  This file already implements most shared functionality (e.g., observation-related data), making it easy to define new tasks on top of it.

- `solo8_constants.py`  
  Contains robot-specific constants used across environments.

- `trotting_demonstration_trajectory.py`  
  Defines reference trajectories for imitation learning.

---

### `solo8/xmls/`
Contains all MuJoCo XML files describing the robot and simulation setup.

- `solo8_mjx_feetonly.xml`  
  Main robot model used in simulation (geometry, joints, actuators, etc.)

- `scene_mjx_feetonly_flat_terrain.xml`  
  Defines the simulation scene and environment, and imports the robot model

- Additional XML files:
  - `actuators.xml` – actuator definitions  
  - `contacts.xml` – contact configuration  
  - `floorplane.xml` – ground plane  
  - `sensors.xml` – sensor setup  
  - `shared.xml` – shared components  
  - `world_unconstrained.xml` – world definition  

- `meshes/`  
  Contains all mesh files used for rendering

---

### `solo8/trotting_demonstration/`
Contains the full **two-stage training pipeline** and evaluation tools.

- `trotting_demonstration_stage1.py`  
  First training stage: imitation learning based on a predefined trajectory

- `trotting_demonstration_stage2.py`  
  Second stage: improves robustness and removes timing dependence

- `demonstration_trajectory.py`  
  Generates a reference trajectory that roughly captures the desired trotting motion

- `evaluate_stage1.py`, `trotting_diagnostics.py`  
  Used for evaluation and visualization

- `diagnostics_output/`  
  Stores generated plots and results

---

## Training Approach

The training is split into two stages:

### Stage 1 – Imitation Learning
A reference trajectory is used, and the agent is trained to follow it as closely as possible.  
This helps the policy learn the basic coordination pattern required for trotting.

### Stage 2 – Robust Locomotion
In the second stage:
- Time dependency is relaxed
- The learned behavior becomes more flexible
- Robustness is improved against disturbances
- Domain randomization is introduced

---

## Running Training

### Training with Visualization (rscope)

In one terminal:
```bash
python learning/train_jax_ppo.py --env_name Solo8TrottingDemonstrationStage1 --rscope_envs 16 --run_evals=False --deterministic_rscope=True
```

In another terminal:
```bash
python -m rscope
```

---

### Restoring from Checkpoint (Stage 2)

```bash
python learning/train_jax_ppo.py --env_name Solo8TrottingDemonstrationStage2 --rscope_envs 16 --run_evals=False --deterministic_rscope=True --load_checkpoint_path logs/<timestamp>/checkpoints
```

Example:
```bash
python learning/train_jax_ppo.py --env_name Solo8TrottingDemonstrationStage2 --rscope_envs 16 --run_evals=False --deterministic_rscope=True --load_checkpoint_path logs/Solo8TrottingDemonstrationStage1-20260310-135721/checkpoints
```

---

## Extending the Project

The project is designed to be easily extendable.

New tasks can be added by simply creating a new file in the `solo8` directory.  
The `Solo8Env` base class already provides all core functionality (simulation setup, sensors, actions), so new environments can focus purely on task definition and reward design.
