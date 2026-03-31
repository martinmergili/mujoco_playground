"""Post-training diagnostic evaluation for Solo8 Trotting Stage 1.
 
Standalone script — does NOT modify train_jax_ppo.py.
Loads a checkpoint produced by train_jax_ppo.py, reconstructs the policy,
runs an evaluation rollout with full metric recording, and generates
the multi-panel diagnostic plots.
 
File placement:
  mujoco_playground/_src/locomotion/solo8/trotting_demonstration/
  ├── stage1.py
  ├── demonstration_trajectory.py
  ├── trotting_diagnostics.py        ← metric extraction + plotting
  └── evaluate_stage1.py             ← THIS FILE
 
Usage:
  # After training finishes (checkpoint saved to logs/<exp>/checkpoints/):
 
  # Evaluate with trained policy:
  python evaluate_stage1.py --checkpoint_path logs/<exp>/checkpoints
 
  # Evaluate open-loop (no policy, just PD tracking default pose):
  python evaluate_stage1.py --open_loop
 
  # Evaluate and zoom into a time window:
  python evaluate_stage1.py --checkpoint_path logs/<exp>/checkpoints \
      --time_start 2.0 --time_end 8.0
 
  # Save raw metric data for later analysis:
  python evaluate_stage1.py --checkpoint_path logs/<exp>/checkpoints \
      --save_npz rollout_data.npz
"""
 
import functools
import json
import os
import warnings
 
os.environ["MUJOCO_GL"] = "egl"
 
from absl import app
from absl import flags
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
 
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
 
# Diagnostics module (same package).
from mujoco_playground._src.locomotion.solo8.trotting_demonstration.trotting_diagnostics import (
    extract_step_metrics,
    run_evaluation_rollout,
    plot_trotting_diagnostics,
    save_rollout_data,
)
 
 
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
 
# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
 
_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "Solo8TrottingDemonstrationStage1",
    "Registered environment name.",
)
_CHECKPOINT_PATH = flags.DEFINE_string(
    "checkpoint_path",
    None,
    "Path to checkpoint directory (from train_jax_ppo.py). "
    "If a directory, uses the latest checkpoint inside it.",
)
_OPEN_LOOP = flags.DEFINE_boolean(
    "open_loop",
    False,
    "If true, run with zero actions (test PD tracking of default pose). "
    "No checkpoint needed.",
)
_N_STEPS = flags.DEFINE_integer(
    "n_steps", 1000, "Number of evaluation steps."
)
_SEED = flags.DEFINE_integer("seed", 0, "Random seed for evaluation.")
_TIME_START = flags.DEFINE_float(
    "time_start", None, "Start of time window for zoomed plot."
)
_TIME_END = flags.DEFINE_float(
    "time_end", None, "End of time window for zoomed plot."
)
_SAVE_NPZ = flags.DEFINE_string(
    "save_npz", None, "Path to save raw metric data as .npz."
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", "diagnostics_output", "Directory for output plots."
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation.")
_PLAYGROUND_CONFIG_OVERRIDES = flags.DEFINE_string(
    "playground_config_overrides", None, "JSON overrides for env config.",
)
 
 
# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors train_jax_ppo.py logic)
# ---------------------------------------------------------------------------
 
def resolve_checkpoint_path(path_str: str) -> epath.Path:
    """Resolve a checkpoint path, picking the latest if it's a directory."""
    ckpt = epath.Path(path_str).resolve()
    if ckpt.is_dir():
        subdirs = [d for d in ckpt.glob("*") if d.is_dir()]
        subdirs.sort(key=lambda x: int(x.name))
        if not subdirs:
            raise FileNotFoundError(f"No checkpoint subdirs in {ckpt}")
        return subdirs[-1]
    return ckpt
 
 
def load_trained_policy(env_name, checkpoint_path, impl="jax", config_overrides=None):
    """Load a trained policy from a checkpoint.
 
    Reconstructs the Brax PPO network and restores parameters.
    Returns (inference_fn, params, env) where inference_fn(obs, rng) -> action.
    """
    # Load env config.
    env_cfg = registry.get_default_config(env_name)
    env_cfg["impl"] = impl
    overrides = {}
    if config_overrides:
        overrides = json.loads(config_overrides)
 
    env = registry.load(env_name, config=env_cfg, config_overrides=overrides)
 
    # PPO params (same defaults as training).
    ppo_params = locomotion_params.brax_ppo_config(env_name, impl)
 
    training_params = dict(ppo_params)
    if "network_factory" in training_params:
        del training_params["network_factory"]
 
    network_factory = ppo_networks.make_ppo_networks
    if hasattr(ppo_params, "network_factory"):
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory
        )
 
    if "num_eval_envs" in training_params:
        del training_params["num_eval_envs"]
 
    # Remove keys we'll pass explicitly to avoid "multiple values" errors.
    for key in ("num_timesteps", "seed", "restore_checkpoint_path",
                "wrap_env_fn", "num_eval_envs", "run_evals",
                "log_training_metrics", "training_metrics_steps"):
        training_params.pop(key, None)
 
    ckpt = resolve_checkpoint_path(checkpoint_path)
    print(f"Loading checkpoint: {ckpt}")
 
    # Use Brax's train with 0 timesteps + restore to just load the policy.
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        **training_params,
        network_factory=network_factory,
        seed=0,
        num_timesteps=0,
        restore_checkpoint_path=ckpt,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        num_eval_envs=1,
        run_evals=False,
    )
 
    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)
 
    return jit_inference_fn, params, env
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main(argv):
    del argv
 
    outdir = epath.Path(_OUTPUT_DIR.value).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
 
    # --- Load environment ---
    env_cfg = registry.get_default_config(_ENV_NAME.value)
    env_cfg["impl"] = _IMPL.value
    overrides = {}
    if _PLAYGROUND_CONFIG_OVERRIDES.value:
        overrides = json.loads(_PLAYGROUND_CONFIG_OVERRIDES.value)
 
    env = registry.load(
        _ENV_NAME.value, config=env_cfg, config_overrides=overrides
    )
 
    # --- Build policy function ---
    if _OPEN_LOOP.value:
        print("Running open-loop evaluation (zero actions).")
        policy_fn = None
        use_default = True
        label = "open_loop"
    else:
        if _CHECKPOINT_PATH.value is None:
            raise ValueError(
                "Provide --checkpoint_path or use --open_loop for no-policy eval."
            )
        jit_inference_fn, _, _ = load_trained_policy(
            _ENV_NAME.value,
            _CHECKPOINT_PATH.value,
            impl=_IMPL.value,
            config_overrides=_PLAYGROUND_CONFIG_OVERRIDES.value,
        )
 
        # Wrap Brax inference_fn(obs, rng) -> (action, extras)
        # into the simpler policy_fn(obs) -> action expected by the rollout.
        eval_rng = jax.random.PRNGKey(_SEED.value + 999)
 
        def policy_fn(obs):
            act, _ = jit_inference_fn(obs, eval_rng)
            return act
 
        use_default = False
        label = "trained"
 
    # --- Run evaluation rollout ---
    print(f"Running {_N_STEPS.value}-step evaluation rollout (seed={_SEED.value})...")
    rollout_data = run_evaluation_rollout(
        env,
        policy_fn=policy_fn,
        n_steps=_N_STEPS.value,
        seed=_SEED.value,
        use_default_action=use_default,
    )
    actual_steps = len(rollout_data["time"])
    print(f"Completed {actual_steps} steps.")
 
    # --- Save raw data if requested ---
    if _SAVE_NPZ.value:
        save_rollout_data(rollout_data, _SAVE_NPZ.value)
 
    # --- Generate full plot ---
    full_path = str(outdir / f"stage1_{label}_full.png")
    plot_trotting_diagnostics(
        rollout_data,
        title=f"Solo8 Trotting Stage 1 — {label.replace('_', ' ').title()}",
        save_path=full_path,
    )
 
    # --- Generate zoomed plot if time range specified ---
    if _TIME_START.value is not None and _TIME_END.value is not None:
        zoom_path = str(outdir / f"stage1_{label}_zoomed.png")
        plot_trotting_diagnostics(
            rollout_data,
            title=(
                f"Solo8 Trotting Stage 1 — {label.replace('_', ' ').title()} "
                f"({_TIME_START.value}–{_TIME_END.value} s)"
            ),
            time_range=(_TIME_START.value, _TIME_END.value),
            save_path=zoom_path,
        )
 
    print(f"All outputs saved to {outdir}/")
 
 
if __name__ == "__main__":
    app.run(main)
 