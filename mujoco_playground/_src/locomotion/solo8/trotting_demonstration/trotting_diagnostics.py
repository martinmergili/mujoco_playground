"""Metric recording and multi-panel plotting for Solo8 trotting Stage 1.

Integrates with the Stage 1 DRL environment to:
  1. Run evaluation rollouts while recording per-timestep diagnostics.
  2. Generate publication-quality multi-panel charts aligned on a shared
     time axis — styled after the footfall/CPG plots in the reference figure.

Recorded metrics per timestep:
  - Footfall (binary contact for FL, FR, HL, HR)
  - Body orientation: roll, pitch, yaw (from quaternion)
  - Foot positions in x and z (world frame, per leg)
  - Base x-velocity and x-acceleration
  - Base z-velocity and z-acceleration

Usage (standalone evaluation + plot):
  from trotting_diagnostics import run_evaluation_rollout, plot_trotting_diagnostics

  env = TrottingDemonstrationStage1()
  data = run_evaluation_rollout(env, policy_fn, n_steps=1000, seed=42)
  fig = plot_trotting_diagnostics(data)
  fig.savefig("trotting_diagnostics.png", dpi=200)

Usage (generate from pre-recorded numpy arrays):
  data = np.load("rollout_metrics.npz")
  fig = plot_trotting_diagnostics(data)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

try:
    import jax
    import jax.numpy as jp
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


# ---------------------------------------------------------------------------
# Quaternion → Euler utilities (JAX-compatible)
# ---------------------------------------------------------------------------

def quat_to_euler_jax(quat):
    """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw].

    Uses the ZYX (yaw-pitch-roll) convention. Requires JAX.
    """
    if not _HAS_JAX:
        raise ImportError("JAX is required for quat_to_euler_jax")
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = jp.clip(sinp, -1.0, 1.0)
    pitch = jp.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jp.arctan2(siny_cosp, cosy_cosp)

    return jp.array([roll, pitch, yaw])


def quat_to_euler_np(quat):
    """Numpy version of quaternion to Euler conversion."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


# ---------------------------------------------------------------------------
# Per-step metric extraction (works on a single mjx.Data snapshot)
# ---------------------------------------------------------------------------

def extract_step_metrics(env, data) -> Dict[str, Any]:
    """Extract all diagnostic metrics from a single simulation state.

    Args:
        env: TrottingDemonstrationStage1 environment instance.
        data: mjx.Data at the current timestep.

    Returns:
        Dictionary with scalar/array values for each metric.
    """
    # --- Foot contact (binary) ---
    contact = jp.array([
        data.sensordata[env._mj_model.sensor_adr[sid]] > 0
        for sid in env._feet_floor_found_sensor
    ]).astype(jp.float32)  # (4,) FL, FR, HL, HR

    # --- Body orientation ---
    quat = data.qpos[3:7]  # [w, x, y, z]
    euler = quat_to_euler_jax(quat)  # [roll, pitch, yaw]

    # --- Foot positions (world frame) ---
    # site_xpos gives world-frame positions for each foot site.
    foot_pos = data.site_xpos[env._feet_site_id]  # (4, 3)
    foot_x = foot_pos[:, 0]  # (4,) x-positions
    foot_z = foot_pos[:, 2]  # (4,) z-positions

    # --- Base velocity (world frame) ---
    # qvel[0:3] = base linear velocity in world frame
    base_vx = data.qvel[0]
    base_vz = data.qvel[2]

    # --- Base position ---
    base_x = data.qpos[0]
    base_z = data.qpos[2]

    return {
        "contact": contact,           # (4,)
        "roll": euler[0],             # scalar (rad)
        "pitch": euler[1],            # scalar (rad)
        "yaw": euler[2],              # scalar (rad)
        "foot_x": foot_x,            # (4,)
        "foot_z": foot_z,            # (4,)
        "base_vx": base_vx,          # scalar
        "base_vz": base_vz,          # scalar
        "base_x": base_x,            # scalar
        "base_z": base_z,            # scalar
    }


# ---------------------------------------------------------------------------
# Evaluation rollout with full metric recording
# ---------------------------------------------------------------------------

def run_evaluation_rollout(
    env,
    policy_fn: Callable,
    n_steps: int = 1000,
    seed: int = 0,
    use_default_action: bool = False,
) -> Dict[str, np.ndarray]:
    """Run a single evaluation episode and record all diagnostics.

    This runs OUTSIDE jit — it steps the environment one timestep at a
    time so we can extract and store full diagnostic data.

    Args:
        env: TrottingDemonstrationStage1 instance.
        policy_fn: Callable(obs) -> action. If None, uses zero actions.
        n_steps: Maximum number of steps.
        seed: Random seed for environment reset.
        use_default_action: If True, use zero actions (open-loop demo test).

    Returns:
        Dictionary of numpy arrays, each of shape (T, ...) where T is the
        number of steps actually taken (may be < n_steps on early termination).
        Keys: time, contact, roll, pitch, yaw, foot_x, foot_z,
              base_vx, base_vz, base_ax, base_az.
    """
    rng = jax.random.PRNGKey(seed)
    state = jax.jit(env.reset)(rng)

    dt = env.dt
    records = {
        "time": [],
        "contact": [],
        "roll": [],
        "pitch": [],
        "yaw": [],
        "foot_x": [],
        "foot_z": [],
        "base_vx": [],
        "base_vz": [],
        "base_x": [],
        "base_z": [],
    }

    step_fn = jax.jit(env.step)

    prev_vx = None
    prev_vz = None
    ax_list = []
    az_list = []

    for i in range(n_steps):
        t = i * dt

        # Extract metrics from current state.
        metrics = extract_step_metrics(env, state.data)

        # Store.
        records["time"].append(float(t))
        records["contact"].append(np.array(metrics["contact"]))
        records["roll"].append(float(metrics["roll"]))
        records["pitch"].append(float(metrics["pitch"]))
        records["yaw"].append(float(metrics["yaw"]))
        records["foot_x"].append(np.array(metrics["foot_x"]))
        records["foot_z"].append(np.array(metrics["foot_z"]))
        records["base_vx"].append(float(metrics["base_vx"]))
        records["base_vz"].append(float(metrics["base_vz"]))
        records["base_x"].append(float(metrics["base_x"]))
        records["base_z"].append(float(metrics["base_z"]))

        # Compute acceleration via finite difference.
        cur_vx = float(metrics["base_vx"])
        cur_vz = float(metrics["base_vz"])
        if prev_vx is not None:
            ax_list.append((cur_vx - prev_vx) / dt)
            az_list.append((cur_vz - prev_vz) / dt)
        else:
            ax_list.append(0.0)
            az_list.append(0.0)
        prev_vx = cur_vx
        prev_vz = cur_vz

        # Step.
        if use_default_action:
            action = jp.zeros(env.mjx_model.nu)
        else:
            action = policy_fn(state.obs)

        state = step_fn(state, action)

        # Early termination.
        if float(state.done) > 0.5:
            break

    # Convert to numpy arrays.
    result = {
        "time": np.array(records["time"]),
        "contact": np.stack(records["contact"]),     # (T, 4)
        "roll": np.array(records["roll"]),            # (T,)
        "pitch": np.array(records["pitch"]),          # (T,)
        "yaw": np.array(records["yaw"]),              # (T,)
        "foot_x": np.stack(records["foot_x"]),        # (T, 4)
        "foot_z": np.stack(records["foot_z"]),        # (T, 4)
        "base_vx": np.array(records["base_vx"]),      # (T,)
        "base_vz": np.array(records["base_vz"]),      # (T,)
        "base_ax": np.array(ax_list),                  # (T,)
        "base_az": np.array(az_list),                  # (T,)
        "base_x": np.array(records["base_x"]),        # (T,)
        "base_z": np.array(records["base_z"]),        # (T,)
        "dt": dt,
    }
    return result


# ---------------------------------------------------------------------------
# Multi-panel plotting (styled after the reference figure)
# ---------------------------------------------------------------------------

# Leg labels and colors matching the reference figure style.
LEG_LABELS = ["FL", "FR", "HL", "HR"]
LEG_COLORS = {
    "FL": "#1f77b4",   # blue
    "FR": "#2ca02c",   # green
    "HL": "#d62728",   # red
    "HR": "#9467bd",   # purple
}
FOOTFALL_BAR_COLOR = "black"


def _add_footfall_panel(ax, time, contact, leg_labels=LEG_LABELS):
    """Draw Gantt-chart style footfall bars.

    Each leg gets a horizontal row. Filled bars = foot on ground.
    Dashed lines = swing (no contact).
    """
    n_legs = contact.shape[1]
    dt = time[1] - time[0] if len(time) > 1 else 0.02

    for leg_idx in range(n_legs):
        y_center = n_legs - 1 - leg_idx  # top-to-bottom: FL, FR, HL, HR

        # Draw dashed baseline (swing phase).
        ax.hlines(
            y_center, time[0], time[-1],
            colors="black", linewidth=0.5, linestyles="dashed", zorder=1,
        )

        # Find contiguous contact intervals.
        c = contact[:, leg_idx]
        in_contact = c > 0.5
        intervals = []
        start = None
        for i, v in enumerate(in_contact):
            if v and start is None:
                start = i
            elif not v and start is not None:
                intervals.append((time[start], time[i - 1] + dt))
                start = None
        if start is not None:
            intervals.append((time[start], time[-1] + dt))

        # Draw filled bars for contact.
        for (t0, t1) in intervals:
            ax.barh(
                y_center, t1 - t0, left=t0, height=0.6,
                color=FOOTFALL_BAR_COLOR, edgecolor="none", zorder=2,
            )

    ax.set_yticks(list(range(n_legs)))
    ax.set_yticklabels(list(reversed(leg_labels)), fontsize=9, fontweight="bold")
    ax.set_ylim(-0.5, n_legs - 0.5)
    ax.set_ylabel("Footfalls", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelbottom=False)


def _add_orientation_panel(ax, time, roll, pitch, yaw):
    """Plot body roll, pitch, yaw in degrees."""
    ax.plot(time, np.degrees(roll), color="#e377c2", linewidth=1.0, label="Roll")
    ax.plot(time, np.degrees(pitch), color="#ff7f0e", linewidth=1.0, label="Pitch")
    ax.plot(time, np.degrees(yaw), color="#17becf", linewidth=1.0, label="Yaw")
    ax.axhline(0, color="black", linewidth=0.4, linestyle="--")
    ax.set_ylabel("Orientation\n[deg]", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=3, framealpha=0.8)
    ax.tick_params(axis="x", labelbottom=False)


def _add_foot_position_panel(ax, time, foot_data, axis_label, ylabel):
    """Plot per-leg foot positions."""
    for leg_idx, label in enumerate(LEG_LABELS):
        ax.plot(
            time, foot_data[:, leg_idx],
            color=LEG_COLORS[label], linewidth=1.0, label=label,
        )
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7, ncol=4, framealpha=0.8)
    ax.tick_params(axis="x", labelbottom=False)


def _add_velocity_accel_panel(ax, time, vel, accel, axis_name,
                              smooth_window=21, smooth_poly=3):
    """Plot velocity and acceleration on twin y-axes."""
    color_vel = "#1f77b4"

    # Smooth velocity curve (window must be odd and > poly order).
    if smooth_window > 0 and len(vel) > smooth_window:
        vel_smooth = savgol_filter(vel, smooth_window, smooth_poly)
    else:
        vel_smooth = vel

    ax.plot(time, vel_smooth, color=color_vel, linewidth=1.0,
            label=f"{axis_name} vel")
    # Optionally show raw signal faintly underneath:
    ax.plot(time, vel, color=color_vel, linewidth=0.3, alpha=0.25)

    ax.set_ylabel(f"{axis_name} velocity\n[m/s]", fontsize=10,
                  fontweight="bold", color=color_vel)
    ax.tick_params(axis="y", labelcolor=color_vel)

    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1, labels1,
              loc="upper right", fontsize=7, ncol=2, framealpha=0.8)

    return ax

def _add_flight_phase_shading(axes, time, contact, color="#7a7a7a", alpha=0.5):
    """Shade vertical bands where all feet are airborne (full suspension).

    Args:
        axes: List of all subplot axes — shading spans every panel.
        time: (T,) time array.
        contact: (T, 4) binary contact array.
    """
    dt = time[1] - time[0] if len(time) > 1 else 0.02

    # Flight = no foot in contact.
    all_airborne = contact.sum(axis=1) == 0  # (T,) boolean

    # Find contiguous flight intervals.
    intervals = []
    start = None
    for i, airborne in enumerate(all_airborne):
        if airborne and start is None:
            start = i
        elif not airborne and start is not None:
            intervals.append((time[start], time[i - 1] + dt))
            start = None
    if start is not None:
        intervals.append((time[start], time[-1] + dt))

    # Shade across all panels.
    for ax in axes:
        for (t0, t1) in intervals:
            ax.axvspan(t0, t1, color=color, alpha=alpha, zorder=0,
                       linewidth=0)


def plot_trotting_diagnostics(
    data: Dict[str, np.ndarray],
    title: str = "Solo8 Trotting — Stage 1 Diagnostics",
    time_range: Optional[tuple] = None,
    figsize: tuple = (16, 12),
    save_path: Optional[str] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Generate a publication-quality multi-panel diagnostic chart.

    Panels (top to bottom):
      1. Footfall (Gantt-chart bars for FL, FR, HL, HR)
      2. Body orientation (roll, pitch, yaw in degrees)
      3. Foot x-positions (4 legs)
      4. Foot z-positions (4 legs)
      5. X-axis velocity & acceleration (dual y-axis)
      6. Z-axis velocity & acceleration (dual y-axis)

    All panels share the same time x-axis for direct comparison.

    Args:
        data: Dictionary from run_evaluation_rollout() or np.load().
        title: Figure title.
        time_range: Optional (t_start, t_end) to zoom in.
        figsize: Figure size in inches.
        save_path: If set, saves figure to this path.
        dpi: Resolution for saving.

    Returns:
        matplotlib Figure.
    """
    time = data["time"]
    contact = data["contact"]
    roll = data["roll"]
    pitch = data["pitch"]
    yaw = data["yaw"]
    foot_z = data["foot_z"]
    base_vx = data["base_vx"]
    base_vz = data["base_vz"]
    base_ax = data["base_ax"]
    base_az = data["base_az"]

    # Apply time range filter if specified.
    if time_range is not None:
        mask = (time >= time_range[0]) & (time <= time_range[1])
        time = time[mask]
        contact = contact[mask]
        roll = roll[mask]
        pitch = pitch[mask]
        yaw = yaw[mask]
        foot_z = foot_z[mask]
        base_vx = base_vx[mask]
        base_vz = base_vz[mask]
        base_ax = base_ax[mask]
        base_az = base_az[mask]

    # --- Create figure with GridSpec for precise height ratios ---
        # --- Create figure with GridSpec for precise height ratios ---
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(
        5, 1,
        height_ratios=[1.2, 1.0, 1.0, 1.0, 1.0],
        hspace=0.08,
    )

    axes = []
    for i in range(5):
        ax = fig.add_subplot(gs[i])
        axes.append(ax)

    # Panel 1: Footfall.
    _add_footfall_panel(axes[0], time, contact)

    # Shade flight phases only on the footfall panel.
    _add_flight_phase_shading([axes[0]], time, contact)

    # Panel 2: Body orientation (roll, pitch, yaw).
    _add_orientation_panel(axes[1], time, roll, pitch, yaw)

    # Panel 3: Foot z-positions.
    _add_foot_position_panel(
        axes[2], time, foot_z, "z", "Foot z\n[m]"
    )

    # Panel 4: X velocity.
    _add_velocity_accel_panel(axes[3], time, base_vx, base_ax, "X")
    axes[3].tick_params(axis="x", labelbottom=False)

    # Panel 5: Z velocity.
    _add_velocity_accel_panel(axes[4], time, base_vz, base_az, "Z")
    axes[4].set_xlabel("Time [s]", fontsize=11, fontweight="bold")


    # Synchronize x-axis limits across all panels.
    for ax in axes:
        ax.set_xlim(time[0], time[-1])
        ax.grid(axis="x", alpha=0.3, linewidth=0.5)

    # Title.
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    fig.align_ylabels(axes)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved diagnostic plot to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Convenience: save/load metric data
# ---------------------------------------------------------------------------

def save_rollout_data(data: Dict[str, np.ndarray], path: str):
    """Save rollout metrics to an .npz file."""
    np.savez_compressed(path, **data)
    print(f"Saved rollout data to {path}")


def load_rollout_data(path: str) -> Dict[str, np.ndarray]:
    """Load rollout metrics from an .npz file."""
    loaded = np.load(path, allow_pickle=True)
    return dict(loaded)


# ---------------------------------------------------------------------------
# Synthetic demo for testing the plots without a trained policy
# ---------------------------------------------------------------------------

def generate_synthetic_demo(
    n_steps: int = 1000,
    dt: float = 0.02,
    gait_freq: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Generate synthetic trotting data for plot testing.

    Creates idealized trotting metrics without running MuJoCo —
    useful for testing the plot layout and style.
    """
    time = np.arange(n_steps) * dt
    phase = 2.0 * np.pi * gait_freq * time

    # Trot offsets: FL=0, FR=pi, HL=pi, HR=0
    offsets = np.array([0.0, np.pi, np.pi, 0.0])

    # Contact: leg on ground when sin(phase + offset) < 0 (stance phase).
    contact = np.zeros((n_steps, 4))
    for i in range(4):
        contact[:, i] = (np.sin(phase + offsets[i]) < 0).astype(float)

    # Orientation: small oscillations.
    roll = 0.02 * np.sin(2 * phase)
    pitch = 0.04 * np.sin(phase + 0.3)
    yaw = 0.01 * np.sin(0.5 * phase)

    # Foot positions.
    foot_z = np.zeros((n_steps, 4))
    foot_x = np.zeros((n_steps, 4))
    for i in range(4):
        leg_phase = phase + offsets[i]
        swing = (np.sin(leg_phase) >= 0).astype(float)
        foot_z[:, i] = swing * 0.06 * np.abs(np.sin(leg_phase))
        foot_x[:, i] = 0.03 * np.sin(leg_phase) + (0.05 if i < 2 else -0.05)

    # Base velocity.
    base_vx = 0.5 + 0.05 * np.sin(phase)
    base_vz = 0.02 * np.sin(2 * phase)

    # Acceleration (finite diff).
    base_ax = np.gradient(base_vx, dt)
    base_az = np.gradient(base_vz, dt)

    return {
        "time": time,
        "contact": contact,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "foot_x": foot_x,
        "foot_z": foot_z,
        "base_vx": base_vx,
        "base_vz": base_vz,
        "base_ax": base_ax,
        "base_az": base_az,
        "base_x": np.cumsum(base_vx) * dt,
        "base_z": 0.27 + 0.005 * np.sin(2 * phase),
        "dt": dt,
    }


# ---------------------------------------------------------------------------
# Main: generate a demo plot for visual verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic trotting demo data...")
    demo_data = generate_synthetic_demo(n_steps=1500, dt=0.02, gait_freq=2.0)

    print("Plotting diagnostics...")
    fig = plot_trotting_diagnostics(
        demo_data,
        title="Solo8 Trotting — Stage 1 Diagnostics (Synthetic Demo)",
        save_path="trotting_diagnostics_demo.png",
        dpi=200,
    )

    # Also show a time-zoomed version.
    fig2 = plot_trotting_diagnostics(
        demo_data,
        title="Solo8 Trotting — Stage 1 Diagnostics (Zoomed 5–10 s)",
        time_range=(5.0, 10.0),
        save_path="trotting_diagnostics_demo_zoomed.png",
        dpi=200,
    )

    print("Done. Saved trotting_diagnostics_demo.png and "
          "trotting_diagnostics_demo_zoomed.png")
