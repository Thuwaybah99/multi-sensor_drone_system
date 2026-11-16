# ============================================================
# UXO UAV EVENT-DRIVEN SIMULATION
# ============================================================

import os
import math
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import fusion_evaluate as fp


# ============================================================
# CONFIGURATION
# ============================================================

THERMAL_IMAGES_DIR = Path(
    r"C:\Users\hp\OneDrive\Desktop\Graduation Project"
    r"\Fyp-G2-Thermal-1\valid\images"
)

GPR_IMAGES_DIR = Path(
    r"C:\Users\hp\OneDrive\Desktop\Graduation Project"
    r"\Synthetic_GPR\images"
)

OUTPUT_DIR = Path(
    r"C:\Users\hp\OneDrive\Desktop\Graduation Project"
    r"\UAV_Simulation_Results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# UAV parameters
# ------------------------------------------------------------

UAV_ALTITUDE = 3.0          # meters
UAV_SPEED = 1.5             # meters/second
SIM_DT = 0.08               # simulation time step

FIELD_X_MIN = -10.0
FIELD_X_MAX = 10.0
FIELD_Y_MIN = -10.0
FIELD_Y_MAX = 10.0

SCAN_SPACING = 2.5          # lawnmower line spacing


# ------------------------------------------------------------
# Detection event parameters
# ------------------------------------------------------------

ACTIVATION_RADIUS = 1.5     # meters

# Time the pipeline remains active for one target
PROCESSING_HOLD_TIME = 1.0


# ============================================================
# DATASET PAIRING
# ============================================================

def build_dataset_pairs():
    """
    Automatically pairs Thermal images with synthetic GPR images.

    Example:

    Thermal:
        abc123.jpg

    GPR:
        abc123_synthetic_gpr.jpg
    """

    thermal_files = sorted(
        list(THERMAL_IMAGES_DIR.glob("*.jpg")) +
        list(THERMAL_IMAGES_DIR.glob("*.jpeg")) +
        list(THERMAL_IMAGES_DIR.glob("*.png"))
    )

    gpr_files = sorted(
        list(GPR_IMAGES_DIR.glob("*.jpg")) +
        list(GPR_IMAGES_DIR.glob("*.jpeg")) +
        list(GPR_IMAGES_DIR.glob("*.png"))
    )

    gpr_map = {}

    for gpr_path in gpr_files:
        stem = gpr_path.stem

        if stem.endswith("_synthetic_gpr"):
            base_stem = stem[:-len("_synthetic_gpr")]
        else:
            base_stem = stem

        gpr_map[base_stem] = gpr_path

    pairs = []

    for thermal_path in thermal_files:

        base_stem = thermal_path.stem

        if base_stem in gpr_map:
            pairs.append(
                (
                    thermal_path,
                    gpr_map[base_stem],
                    base_stem
                )
            )

    print("\n========================================")
    print("DATASET PAIRING")
    print("========================================")
    print("Thermal images :", len(thermal_files))
    print("GPR images     :", len(gpr_files))
    print("Valid pairs    :", len(pairs))

    if not pairs:
        raise RuntimeError(
            "No Thermal/GPR pairs were found.\n"
            "Check THERMAL_IMAGES_DIR and GPR_IMAGES_DIR."
        )

    for i, (t, g, name) in enumerate(pairs[:10]):
        print(f"\nPair {i + 1}")
        print(" Thermal:", t.name)
        print(" GPR    :", g.name)

    if len(pairs) > 10:
        print(f"\n... and {len(pairs) - 10} more pairs")

    return pairs


# ============================================================
# UAV LAWNMOWER PATH
# ============================================================

def create_lawnmower_path():
    """
    Creates a continuous lawnmower flight path.
    """

    path = []

    y_values = np.arange(
        FIELD_Y_MIN,
        FIELD_Y_MAX + SCAN_SPACING,
        SCAN_SPACING
    )

    for row, y in enumerate(y_values):

        if row % 2 == 0:
            x_values = np.linspace(
                FIELD_X_MIN,
                FIELD_X_MAX,
                100
            )
        else:
            x_values = np.linspace(
                FIELD_X_MAX,
                FIELD_X_MIN,
                100
            )

        for x in x_values:
            path.append(
                [float(x), float(y), UAV_ALTITUDE]
            )

    return np.array(path)


# ============================================================
# SIMULATED UNDERGROUND TARGETS
# ============================================================

def create_targets(number_of_targets):
    """
    Creates simulated underground UXO positions.

    IMPORTANT:
    These are simulation objects used only to trigger
    the acquisition event. They are not claimed as the
    physical ground truth coordinates of the dataset.
    """

    predefined_positions = [
        (-7.0, -7.0, -1.0),
        (2.0, -4.0, -1.3),
        (7.0, 0.0, -0.8),
        (-3.0, 4.5, -1.2),
        (6.0, 7.0, -1.0),
        (-7.0, 8.0, -1.4),
    ]

    number_of_targets = min(
        number_of_targets,
        len(predefined_positions)
    )

    targets = []

    for i in range(number_of_targets):

        x, y, z = predefined_positions[i]

        targets.append(
            {
                "id": i + 1,
                "x": x,
                "y": y,
                "z": z,
                "triggered": False,
                "completed": False,
            }
        )

    return targets


# ============================================================
# DISTANCE
# ============================================================

def distance_2d(uav, target):
    return math.sqrt(
        (uav[0] - target["x"]) ** 2 +
        (uav[1] - target["y"]) ** 2
    )


# ============================================================
# B-SCAN DISPLAY
# ============================================================

def show_bscan(gpr_path, target_id):
    """
    Displays the synthetic GPR image as the simulated B-scan.

    The current dataset provides an image/radargram rather than
    raw GPR traces, so we display that image as the B-scan
    representation used by the simulation.
    """

    gpr_img = cv2.imread(str(gpr_path), cv2.IMREAD_GRAYSCALE)

    if gpr_img is None:
        print("WARNING: Could not read GPR image.")
        return

    plt.figure(figsize=(9, 4))

    plt.imshow(
        gpr_img,
        cmap="gray",
        aspect="auto"
    )

    plt.title(
        f"GPR B-Scan — Target {target_id}"
    )

    plt.xlabel("Scan / Horizontal Position")
    plt.ylabel("Depth / Time")

    plt.tight_layout()

    output_path = OUTPUT_DIR / (
        f"target_{target_id:02d}_bscan.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show(block=False)
    plt.pause(0.01)

    print("B-scan saved:", output_path)


# ============================================================
# RUN EXISTING FUSION PIPELINE
# ============================================================

def run_detection_event(
    thermal_path,
    gpr_path,
    target_id
):
    """
    Activates the existing fusion pipeline.

    We deliberately do not modify the fusion algorithm here.
    """

    print("\n")
    print("========================================")
    print(f"TARGET {target_id} DETECTION EVENT")
    print("========================================")

    print("Pipeline status: ON")

    print("Thermal:")
    print(thermal_path)

    print("GPR:")
    print(gpr_path)

    thermal_img = cv2.imread(
        str(thermal_path)
    )

    gpr_img = cv2.imread(
        str(gpr_path)
    )

    if thermal_img is None:
        print("ERROR: Thermal image could not be loaded.")
        return None

    if gpr_img is None:
        print("ERROR: GPR image could not be loaded.")
        return None

    output_path = OUTPUT_DIR / (
        f"target_{target_id:02d}_fusion.png"
    )

    # --------------------------------------------------------
    # THIS IS YOUR EXISTING PIPELINE
    # --------------------------------------------------------

    predictions = fp.run_pipeline_single(
        thermal_img,
        gpr_img,
        thermal_path,
        str(output_path)
    )
    #show thermal detection results
    result_img = cv2.imread(str(output_path))
    if result_img is not None:
        cv2.imshow(f"Thermal + Fusion Detection - Target {target_id}", result_img)
        cv2.waitKey(1)
    else:
        print("ERROR: Fusion result image could not be loaded.")
    print("\nPipeline result:")
    print("Predictions:", predictions)

    print("Fusion image saved:")
    print(output_path)

    # --------------------------------------------------------
    # B-SCAN
    # --------------------------------------------------------

    #show_bscan( gpr_path, target_id)

    print("\nPipeline status: OFF")

    return predictions


# ============================================================
# SIMULATION STATE
# ============================================================

class SimulationState:

    def __init__(self, path, targets, pairs):

        self.path = path
        self.targets = targets
        self.pairs = pairs

        self.path_index = 0

        self.current_target = None

        self.pipeline_active = False

        self.processing_until = 0.0

        self.event_counter = 0

        self.results = []

        self.sim_time = 0.0

        self.finished = False


# ============================================================
# TARGET EVENT
# ============================================================

def process_target_event(
    state,
    target
):

    target["triggered"] = True

    state.pipeline_active = True
    state.current_target = target

    state.event_counter += 1

    event_index = state.event_counter - 1

    if event_index >= len(state.pairs):
        print(
            "No more dataset pairs available."
        )

        state.pipeline_active = False
        target["completed"] = True

        return

    thermal_path, gpr_path, sample_name = (
        state.pairs[event_index]
    )

    print("\n")
    print("########################################")
    print(
        f" EVENT {state.event_counter}: "
        f"TARGET {target['id']}"
    )
    print("########################################")

    print(
        f"UAV position: "
        f"X={state.path[state.path_index][0]:.2f}, "
        f"Y={state.path[state.path_index][1]:.2f}, "
        f"Z={state.path[state.path_index][2]:.2f}"
    )

    predictions = run_detection_event(
        thermal_path,
        gpr_path,
        target["id"]
    )

    state.results.append(
        {
            "event": state.event_counter,
            "target_id": target["id"],
            "uav_x": state.path[state.path_index][0],
            "uav_y": state.path[state.path_index][1],
            "uav_z": state.path[state.path_index][2],
            "sample": sample_name,
            "thermal_image": str(thermal_path),
            "gpr_image": str(gpr_path),
            "num_predictions": (
                len(predictions)
                if predictions is not None
                else 0
            ),
        }
    )

    target["completed"] = True

    state.pipeline_active = False

    print(
        f"\nTarget {target['id']} processing complete."
    )

    print("Pipeline status: OFF")


# ============================================================
# DRAW UAV
# ============================================================

def draw_uav(ax, position):

    x, y, z = position

    ax.scatter(
        [x],
        [y],
        [z],
        s=100,
        marker="^",
        label="UAV"
    )


# ============================================================
# DRAW TARGETS
# ============================================================

def draw_targets(ax, targets):

    for target in targets:

        ax.scatter(
            [target["x"]],
            [target["y"]],
            [target["z"]],
            s=150,
            marker="o"
        )

        ax.text(
            target["x"],
            target["y"],
            target["z"] - 0.25,
            f"UXO {target['id']}",
            fontsize=8
        )


# ============================================================
# DRAW GROUND
# ============================================================

def draw_ground(ax):

    x = np.linspace(
        FIELD_X_MIN,
        FIELD_X_MAX,
        20
    )

    y = np.linspace(
        FIELD_Y_MIN,
        FIELD_Y_MAX,
        20
    )

    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)

    ax.plot_surface(
        X,
        Y,
        Z,
        alpha=0.15
    )


# ============================================================
# UPDATE SIMULATION
# ============================================================

def update_simulation(
    frame,
    state,
    uav_marker,
    trajectory_line,
    status_text,
    target_markers
):

    if state.finished:
        return

    state.sim_time += SIM_DT

    # --------------------------------------------------------
    # Move UAV
    # --------------------------------------------------------

    if state.path_index < len(state.path) - 1:

        state.path_index += 1

    else:

        state.finished = True

        print("\n========================================")
        print("SIMULATION COMPLETE")
        print("========================================")

        return

    position = state.path[
        state.path_index
    ]

    # --------------------------------------------------------
    # EVENT DETECTION
    # --------------------------------------------------------

    if not state.pipeline_active:

        for target in state.targets:

            if target["triggered"]:
                continue

            distance = distance_2d(
                position,
                target
            )

            if distance <= ACTIVATION_RADIUS:

                process_target_event(
                    state,
                    target
                )

                break

    # --------------------------------------------------------
    # UAV graphics
    # --------------------------------------------------------

    uav_marker._offsets3d = (
        [position[0]],
        [position[1]],
        [position[2]]
    )

    # --------------------------------------------------------
    # Trajectory
    # --------------------------------------------------------

    trajectory = state.path[
        :state.path_index + 1
    ]

    trajectory_line.set_data(
        trajectory[:, 0],
        trajectory[:, 1]
    )

    trajectory_line.set_3d_properties(
        trajectory[:, 2]
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if state.pipeline_active:

        status = (
            f"PIPELINE: ON\n"
            f"Target: {state.current_target['id']}"
        )

    else:

        remaining = sum(
            not t["triggered"]
            for t in state.targets
        )

        status = (
            f"PIPELINE: OFF\n"
            f"Targets remaining: {remaining}"
        )

    status_text.set_text(status)

    # --------------------------------------------------------
    # Target colors / size
    # --------------------------------------------------------

    for i, target in enumerate(state.targets):

        if target["triggered"]:

            target_markers[i]._sizes = [250]

        else:

            target_markers[i]._sizes = [120]


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(state):

    import csv

    output_file = OUTPUT_DIR / "simulation_results.csv"

    if not state.results:

        print("No simulation results.")
        return

    fieldnames = list(
        state.results[0].keys()
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            state.results
        )

    print(
        "\nSimulation CSV saved:"
    )

    print(output_file)


# ============================================================
# MAIN SIMULATION
# ============================================================

def main():

    print("\n")
    print("========================================")
    print("       UAV UXO SIMULATION")
    print("========================================")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    pairs = build_dataset_pairs()

    # --------------------------------------------------------
    # UAV path
    # --------------------------------------------------------

    path = create_lawnmower_path()

    # --------------------------------------------------------
    # Underground targets
    # --------------------------------------------------------

    number_of_targets = min(
        5,
        len(pairs)
    )

    targets = create_targets(
        number_of_targets
    )

    # --------------------------------------------------------
    # Simulation state
    # --------------------------------------------------------

    state = SimulationState(
        path,
        targets,
        pairs
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(12, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.set_xlim(
        FIELD_X_MIN,
        FIELD_X_MAX
    )

    ax.set_ylim(
        FIELD_Y_MIN,
        FIELD_Y_MAX
    )

    ax.set_zlim(
        -2.5,
        UAV_ALTITUDE + 1
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.set_title(
        "UAV-Based UXO Detection Simulation"
    )

    # --------------------------------------------------------
    # Ground
    # --------------------------------------------------------

    draw_ground(ax)

    # --------------------------------------------------------
    # Targets
    # --------------------------------------------------------

    draw_targets(
        ax,
        targets
    )

    target_markers = []

    for target in targets:

        marker = ax.scatter(
            [target["x"]],
            [target["y"]],
            [target["z"]],
            s=120,
            marker="o"
        )

        target_markers.append(marker)

    # --------------------------------------------------------
    # UAV
    # --------------------------------------------------------

    initial_position = path[0]

    uav_marker = ax.scatter(
        [initial_position[0]],
        [initial_position[1]],
        [initial_position[2]],
        s=120,
        marker="^"
    )

    # --------------------------------------------------------
    # Flight trajectory
    # --------------------------------------------------------

    trajectory_line, = ax.plot(
        [initial_position[0]],
        [initial_position[1]],
        [initial_position[2]],
        linewidth=2,
        label="UAV trajectory"
    )

    # --------------------------------------------------------
    # Status display
    # --------------------------------------------------------

    status_text = ax.text2D(
        0.02,
        0.92,
        "PIPELINE: OFF",
        transform=ax.transAxes,
        fontsize=12
    )

    ax.legend()

    # --------------------------------------------------------
    # Animation
    # --------------------------------------------------------

    animation = FuncAnimation(
        fig,
        update_simulation,
        fargs=(
            state,
            uav_marker,
            trajectory_line,
            status_text,
            target_markers
        ),
        interval=SIM_DT * 1000,
        blit=False,
        repeat=False
    )

    plt.show()

    # Prevent animation from being garbage collected
    return animation, state


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    animation, simulation_state = main()

    save_results(
        simulation_state
    )