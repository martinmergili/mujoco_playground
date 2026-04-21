# Solo8

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
  Contains robot- and environment-specific constants, including XML paths, sensor names, and foot-related definitions.  
  It ensures consistent naming and configuration across all environments and tasks.

- `trotting_gait_tracking.py`  
  - Needs to be renamed
  - Description needs to be extended: Comprises our from-scratch approach (tries to generate a trotting gait pattern through RL with random exploration - no prior knowledge, depends heavily on reward structure)

---

### `solo8/xmls/`
Contains all MuJoCo XML files (MJCF) describing the robot and simulation setup.

- `solo8_mjx_feetonly.xml`  
  Main robot model used in simulation. It defines the full Solo8 system, including
  - Kinematics (floating base + 4 legs - front/hind, left/right, each consisting of a hip and knee joint)
  - Dynamics (masses, inertias, joint limits, damping)
  - PD-controlled actuators
  - Relevant sensors (IMU, velocities, foot positions, contact sensors).

  Collision handling is intentionally simplified: collisions are disabled globally and only enabled for the **feet**, meaning that only foot–ground contacts are considered. This improves simulation stability and is sufficient for locomotion tasks like trotting.

  For more complex environments, this model can be extended by enabling additional collision geometries and contact interactions.

- `scene_mjx_feetonly_flat_terrain.xml`  
  Defines the simulation environment and includes the Solo8 robot model. It sets up a simple flat terrain with a plane ground, basic lighting, textures, and rendering settings.

  The floor is configured to only interact with the robot’s feet (matching the collision setup in the robot model), ensuring stable and efficient contact simulation.

  A predefined **`home` keyframe** specifies the robot’s initial pose, including base position/orientation and joint configuration. This provides a consistent starting state for simulations and training.

- `meshes/`  
  Contains all mesh files used for rendering

---

### `solo8/trotting_demonstration/`
Contains the full **two-stage training pipeline** and evaluation tools.

- `demonstration_trajectory.py`  
  Generates a reference trajectory that roughly captures the desired trotting motion, using a Bézier-based swing profile for the z-position and a sinusoidal function for the x-position

- `trotting_demonstration_stage1.py`
  The first stage is used to track the demonstration trajectory. This helps the policy learn the basic coordination pattern required for trotting.

- `trotting_demonstration_stage2.py`  
  Second stage: improves robustness, removes timing dependence, and introduces domain randomisation to facilitate a better sim-to-real transfer

- `evaluate_stage1.py`, `trotting_diagnostics.py`  
  Used for evaluation and visualization

- `diagnostics_output/`  
  Stores generated plots and results

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

## Trouble shooting

---

## Extending the Project

The project is designed to be easily extendable.

New tasks can be added by simply creating a new file in the `solo8` directory.  
The `Solo8Env` base class already provides all core functionality (simulation setup, sensors, actions), so new environments can focus purely on task definition and reward design.

Additionally, new environments can be added to train the robot in a different scene (e.g. rough terrain or obstacle-filled scenes). As different scenes were used in other robot platforms (e.g. Go1), their implementations can be used for guidance.
  - Path from the XML needs to be added to the solo8_constants.py
  - To run the training with different scenes ... ( what to change in the command?)

  - Improve reward design
  - Use trajectory optimisation (TO) to get a better reference trajectory for stage 1 training
