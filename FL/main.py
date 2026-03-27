import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np

import json 
import os
import sys
import argparse
import traceback

from .model import ThreeLevelGame
from .branch_and_bound import BranchAndBound
from .logging_setup import experiment_logger, experiment_log_path, BNB_status_logger

# -------------------------------------------------------------------------
# CSV Logging
# -------------------------------------------------------------------------

HEADER = ("BNB.SLURM_ARRAY_JOB_ID,BNB.SLURM_ARRAY_TASK_ID,"
        "V,"
        "U,"
        "K,"
        "R,"
        "b_a,"
        "b_p,"
        "structure_propagation,"
        "strategy_propagation,"
        "seed,"
        "TimeLimit,"
        "Anti-Symmetry,"
        "BNB.#Nodes,"
        "BNB.Time,"
        "BNB.P.Time,"
        "BNB.A.Time,"
        "BNB.D.Time,"
        "BNB.AR.Time,"
        "BNB.DR.Time,"
        "BNB.H.Time,"
        "BNB.C.Strat.Propagation.Time,"
        "BNB.LB,"
        "BNB.UB,"
        "BNB.Opt.Gap,"
        "BNB.C.Strats,"
        "BNB.C.Structs,"
        "BNB.A.calls,"
        "BNB.D.calls,"
        "BNB.H.calls,"
        "BNB.C.Strat.Propagation.Calls,"
        "BNB.Opt.Policy,"
        "BNB.Opt.Strategy,"
        "BNB.Opt.Allocation,"
        "BNB.Opt.Location," 
        "BNB.Error" 
    )

def _csv_safe(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple, dict)):
        return f'"{json.dumps(v)}"'
    return str(v)

def log_dicts_csv(logger, log_path, graph_config, bnb_stats):
    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    if write_header:
        logger.info(HEADER)

    merged = {
        **graph_config,
        **{f"BNB.{k}": v for k, v in bnb_stats.items()},
    }

    columns = HEADER.split(",")
    row = ",".join(_csv_safe(merged.get(col, "")) for col in columns)
    logger.info(row)


# -------------------------------------------------------------------------
# Graph Generation
# -------------------------------------------------------------------------
def generate_instance_data(num_facilities:int, num_survivors:int, seed:int):
    random.seed(seed)
    facility_locations = [(round(random.uniform(0,10),2),round(random.uniform(0,10),2)) for _ in range(num_facilities)]
    survivor_locations = [(round(random.uniform(0,10),2),round(random.uniform(0,10),2)) for _ in range(num_survivors)]

    return facility_locations, survivor_locations


# -------------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------------

def plot_colored_graph(best_feasible_solution:tuple, model:ThreeLevelGame):
    if best_feasible_solution:
        (obj_val, protector_policy, attacker_strategy, defender_allocation, defender_location) = best_feasible_solution 
    else:
        (obj_val, protector_policy, attacker_strategy, defender_allocation, defender_location) = float('nan'), [], [], [], []


    log_directory = os.getenv("EXPERIMENT_DIR", "FL/logs/local")
    
    fig, ax = plt.subplots(figsize=(8, 8))

    # Extract facility and customer coordinates
    facility_x, facility_y = zip(*model.facility_locations)
    survivor_x, survivor_y = zip(*model.survivor_locations)

    # Plot facilities and customers
    plt.scatter(facility_x, facility_y, facecolors='none', edgecolors='green', s=100, label='Facilities', marker='s')
    plt.scatter(survivor_x, survivor_y, facecolors='none', edgecolors='blue', s=100, label='Customers', marker='o')

    # Add customer numbers
    for i, (x, y) in enumerate(model.survivor_locations):
        plt.text(x, y, str(i), color="black", fontsize=8, ha="center", va="center")

    # Add facility numbers
    for i, (x, y) in enumerate(model.facility_locations):
        plt.text(x, y, str(i), color="black", fontsize=8, ha="center", va="center")

    # ====================================== Messy, don't worry about this 
    # Number of horizontal strips (rows)
    rows = math.floor(math.sqrt(model.num_regions))
    cols_per_row = []  # How many columns in each row

    remaining = model.num_regions
    for _ in range(rows):
        cols = math.ceil(remaining / (rows - len(cols_per_row)))
        cols_per_row.append(cols)
        remaining -= cols

    cell_id = 0
    y_step = 10 / rows
    region_bounds = []

    # Precompute region bounds (xmin, xmax, ymin, ymax) for each cell
    for row, cols in enumerate(cols_per_row):
        x_step = 10 / cols
        for col in range(cols):
            xmin = col * x_step
            xmax = (col + 1) * x_step
            ymin = row * y_step
            ymax = (row + 1) * y_step
            region_bounds.append((cell_id, xmin, xmax, ymin, ymax))
            cell_id += 1

    gap_ratio = 0.05  # 5% gap on each side (adjust if needed)
    # ====================================== Messy, don't worry about this 


    # Protector Policy Plot (Blue Shading of Regions)
    for (a, c) in protector_policy:
        cell_id, xmin, xmax, ymin, ymax = region_bounds[a]

        # for cell_id, xmin, xmax, ymin, ymax in region_bounds:
        width = xmax - xmin
        height = ymax - ymin

        # Shrink the rectangle a bit for spacing
        shrink_width = width * (1 - gap_ratio)
        shrink_height = height * (1 - gap_ratio)

        # Center the smaller rectangle in the cell
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        lower_left_x = center_x - shrink_width / 2
        lower_left_y = center_y - shrink_height / 2

        rect = patches.Rectangle(
            (lower_left_x, lower_left_y),
            shrink_width, shrink_height,
            facecolor='#008080', alpha=0.3, edgecolor='black'
        )
        ax.add_patch(rect)
        ax.text(center_x, center_y, str(c), color="black", fontsize=8, ha="center", va="center")

    # Attacker Strategy Plot (Red Dashed Lines)
    for facility_idx, customer_idx in attacker_strategy:
        fx, fy = model.facility_locations[facility_idx]
        cx, cy = model.survivor_locations[customer_idx]
        plt.plot([fx, cx], [fy, cy], color="red", linestyle="--", linewidth=1)

    # Defender Allocation (gray dashed lines)
    for facility_idx, customer_idx in defender_allocation:
        fx, fy = model.facility_locations[facility_idx]
        cx, cy = model.survivor_locations[customer_idx]
        plt.plot([fx, cx], [fy, cy], color="gray", linestyle="--", linewidth=1)

    # Adjust grid dynamically
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Facility and Customer Locations")
    plt.grid(True)

    grid_size = int(math.sqrt(model.num_regions))  # N for an N×N grid
    cell_size = 10 / grid_size  # Each quadrant size in (0,10) space

    # Dynamic grid ticks
    plt.xticks(np.arange(0, 10 + cell_size, cell_size))
    plt.yticks(np.arange(0, 10 + cell_size, cell_size))

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()

    # Append solution information at the bottom of the figure
    info_text = (
        f"Objective Value: {obj_val}\n"
        f"Protection Policy: {protector_policy} --- Max Patrols: {model.b_p}\n"
        f"Attack: {attacker_strategy} --- Att Budget: {model.b_a}\n"
        f"Crew Location: {defender_location}\n"
        f"Crew Allocation: {defender_allocation}"
    )

    # Adjust layout to make space for text
    plt.subplots_adjust(bottom=0.3)
    plt.figtext(0.25, 0.12, info_text, wrap=True, horizontalalignment='left', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    
    # Proxy artists for legend
    route_line = mlines.Line2D([], [], color="gray", linestyle="--", label="Routes")
    attack_line = mlines.Line2D([], [], color="red", linestyle="--", label="Attacks")
    protected_patch = mpatches.Patch(facecolor="#008080", alpha=0.3, edgecolor="black", label="Protected Region")

    # Existing handles from scatter plots
    handles, labels = ax.get_legend_handles_labels()

    # Add the proxies
    handles.extend([route_line, attack_line, protected_patch])

    ax.legend(handles=handles, loc="best")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Save the plot
    plt.savefig(f"{log_directory}/solution.png")
    plt.close()

# -------------------------------------------------------------------------
# Solvers
# -------------------------------------------------------------------------

def run_BNB(I:int, J:int, K:int, A:int, b_p:int, b_a:int, strategy_propagation:int, structure_propagation:int, seed:int, anti_symmetry:bool, time_limit:int=60*60*3):
    
    facility_locations, survivor_locations = generate_instance_data(I, J, seed)

    model = ThreeLevelGame(
        facility_locations=facility_locations,
        survivor_locations=survivor_locations,
        num_crews=K,
        num_regions=A,
        b_p=b_p,
        b_a=b_a,
        strategy_propagation=strategy_propagation,
        structure_propagation=structure_propagation,
        seed=seed,
        anti_symmetry=anti_symmetry,
    )

    bnb = BranchAndBound(model, time_limit)
    result = bnb.solveBNB()
    plot_colored_graph(result, model)
    return bnb.get_statistics()


# -------------------------------------------------------------------------
# SLURM Entry Point
# -------------------------------------------------------------------------

def run_SLURM(V:int, U:int, K:int, R:int, b_p:int, b_a:int, strategy_propagation:int, structure_propagation:int, seed:int, anti_symmetry:bool, time_limit:int):
    had_error = False 
    bnb_stats = {}

    try:
        bnb_stats = run_BNB(V, U, K, R, b_p, b_a, strategy_propagation, structure_propagation, seed, anti_symmetry, time_limit)
    except Exception as e:
        err = traceback.format_exc()
        BNB_status_logger.error(f"BNB failed:\n{err}")
        bnb_stats["Error"] = repr(e)
        had_error = True

    # Log SLURM run
    graph_config = {"V":V, "U":U, "K":K, "R":R ,"b_p":b_p,"b_a":b_a,"seed":seed,"structure_propagation":structure_propagation,"strategy_propagation":strategy_propagation,"TimeLimit":time_limit,"Anti-Symmetry":anti_symmetry}
    log_dicts_csv(experiment_logger, experiment_log_path, graph_config, bnb_stats)

    if had_error:
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--V", type=int, default=3, help="number of service facilities")
    parser.add_argument("--U", type=int, default=10, help="number of survivor locations")
    parser.add_argument("--K", type=int, default=10, help="number of crews")
    parser.add_argument("--R", type=int, default=4, help="number of regions")
    parser.add_argument("--b_p", type=int, default=2, help="protector budget")
    parser.add_argument("--b_a", type=int, default=5, help="attacker budget")
    parser.add_argument("--strategy_propagation",  type=int, choices=[-1,0]    ,default=0, help="Protector constr. propagation type")
    parser.add_argument("--structure_propagation", type=int, choices=[-1,0,1,2], default=0, help="Attacker constr. propagation type")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--anti_symmetry", action="store_true",help="Enable constraints that fight symmetry in the protector and defender")
    parser.add_argument("--time_limit", type=int, default=60*60*3)

    args = parser.parse_args()
    BNB_status_logger.info(args)

    run_SLURM(
        V=args.V,
        U=args.U,
        K=args.K,
        R=args.R,
        b_p=args.b_p,
        b_a=args.b_a,
        strategy_propagation=args.strategy_propagation,
        structure_propagation=args.structure_propagation,
        seed=args.seed,
        anti_symmetry=args.anti_symmetry,
        time_limit=args.time_limit,
    )
