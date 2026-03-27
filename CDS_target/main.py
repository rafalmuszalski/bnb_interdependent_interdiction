# -*- coding: utf-8 -*-

import os
import sys
import json
import random
import argparse
import traceback

import networkx as nx
import matplotlib.pyplot as plt

from .model import ThreeLevelGame
from .branch_and_bound import BranchAndBound
from .logging_setup import experiment_logger, experiment_log_path, BNB_status_logger


# -------------------------------------------------------------------------
# CSV Logging
# -------------------------------------------------------------------------

HEADER = (
    "BNB.SLURM_ARRAY_JOB_ID,BNB.SLURM_ARRAY_TASK_ID,"
    "N,K,b_a,b_p,structure_propagation,strategy_propagation,seed,TimeLimit,"
    "BNB.#Nodes,BNB.Time,BNB.P.Time,BNB.A.Time,BNB.D.Time,BNB.AR.Time,BNB.DR.Time,"
    "BNB.H.Time,BNB.C.Strat.Propagation.Time,"
    "BNB.LB,BNB.UB,BNB.Opt.Gap,"
    "BNB.C.Strats,BNB.C.Structs,BNB.A.calls,BNB.D.calls,BNB.H.calls,"
    "BNB.C.Strat.Propagation.Calls,"
    "BNB.Opt.Policy,BNB.Opt.Strategy,BNB.Opt.Structure,"
    "BNB.Error,"
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

def generate_small_world_clique(n: int, clique_size: int, seed: int):
    assert n % clique_size == 0, "clique_size must divide n exactly"

    random.seed(seed)
    G = nx.Graph()
    node_offset = 0
    cliques = []
    
    for _ in range(n//clique_size):
        clique_nodes = list(range(node_offset, node_offset + clique_size))
        G.add_nodes_from(clique_nodes)
        G.add_edges_from([(u, v) for u in clique_nodes for v in clique_nodes if u < v])
        cliques.append(clique_nodes)
        node_offset += clique_size


    # Connect every clique to every other clique
    num_bridges = 4  # how many edges to add between each pair of cliques
    for i, clique_a in enumerate(cliques):
        for clique_b in cliques[i+1:]:
            for _ in range(num_bridges):
                node_a = random.choice(clique_a)
                node_b = random.choice(clique_b)
                # Avoid duplicate edges
                if not G.has_edge(node_a, node_b):
                    G.add_edge(node_a, node_b)

    protectors_of = [[] for _ in G.nodes()]
    protected_by = [[] for _ in G.nodes()]

    for i in G.nodes():
        G.nodes[i]['price'] =1 
        G.nodes[i]['cost'] = 1 
        G.nodes[i]['weight'] = round(random.uniform(100,300),0)

        neighbors = list(G.neighbors(i))
        random.shuffle(neighbors)
        to_protect = neighbors[:2] + [i]
        protectors_of[i] = to_protect
        for j in to_protect:
            protected_by[j].append(i)

    # Q(i) is protected_by[i]
    # Q^{-1}(i) is protectors_of[i]
    return G, protected_by, protectors_of

# -------------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------------

def plot_colored_graph(G, protected_by, w_hat=[], x_hat=[], y_hat=[], name:str="test"):
    log_directory = os.getenv("EXPERIMENT_DIR", "CDS_target/logs/local")
    
    pos = nx.get_node_attributes(G,'pos')
    if not pos:
        pos = nx.spring_layout(G, seed=1)

    # Draw edges first
    nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.5)

    # Assign colors per node
    node_colors = []
    edge_colors = {e: "black" for e in G.edges}  # default edge color

    for n in G.nodes:
        if n in w_hat: # Fortified 
            node_colors.append("green")

            # Color edges between n and its protected neighbors
            for i in protected_by[n]:
                if G.has_edge(n, i):
                    edge_colors[(n, i)] = "green"
                    edge_colors[(i, n)] = "green"  # handle undirected edges

        elif n in x_hat: # Attacked 
            node_colors.append("red")
        elif n in y_hat: # DS 
            node_colors.append("blue")
        else: # Default 
            node_colors.append("none")


    # Convert edge_colors dict to a list in G.edges order
    edge_color_list = [edge_colors.get(e, "black") for e in G.edges]

    # Draw the graph
    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_color_list,
        node_size=300,
        edgecolors="black",  # use color for border instead
        font_color="black",
        font_weight="bold",
    )
    plt.savefig(os.path.join(log_directory,f"{name}.png"))
    plt.close()

# -------------------------------------------------------------------------
# Solvers
# -------------------------------------------------------------------------

def run_BNB(n: int, k: int, b_p: int, b_a: int, strategy_propagation: int,
            structure_propagation: int, seed: int, time_limit: int = 60 * 60 * 3):
    

    G, protected_by, protectors_of = generate_small_world_clique(n=n,clique_size=k,seed=seed)

    model = ThreeLevelGame(
        num_nodes=n,
        b_p=b_p,
        b_a=b_a,
        strategy_propagation=strategy_propagation,
        structure_propagation=structure_propagation,
        G=G,
        protected_by=protected_by,
        protectors_of=protectors_of,
        seed=seed,
    )

    bnb = BranchAndBound(model, time_limit)
    result = bnb.solveBNB()
    plot_colored_graph(G, protected_by, result[1], result[2], result[3], name="BNB")
    return bnb.get_statistics()

# -------------------------------------------------------------------------
# SLURM Entry Point
# -------------------------------------------------------------------------

def run_SLURM(n: int, k: int, b_p: int, b_a: int, strategy_propagation: int,
              structure_propagation: int, seed: int, time_limit: int):
    
    had_error = False
    bnb_stats= {}
    
    try:
        bnb_stats = run_BNB(n=n,
                        k=k, 
                        b_p=b_p,
                        b_a=b_a, 
                        strategy_propagation=strategy_propagation, 
                        structure_propagation=structure_propagation, 
                        seed=seed, 
                        time_limit=time_limit
                        )
    except Exception as e:
        err = traceback.format_exc()
        BNB_status_logger.error(f"BNB failed:\n{err}")
        bnb_stats["Error"] = repr(e)
        had_error = True
    
    graph_config = {
        "N": n, "K": k, "b_p": b_p, "b_a": b_a, "seed": seed,
        "structure_propagation": structure_propagation,
        "strategy_propagation": strategy_propagation,
        "TimeLimit": time_limit,
    }
    log_dicts_csv(experiment_logger, experiment_log_path, graph_config, bnb_stats)

    if had_error:
        sys.exit(1)
# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",                     type=int,   default=20,                  help="Number of nodes in the graph")
    parser.add_argument("--k",                     type=int,   default=5,                   help="Clique size")
    parser.add_argument("--b_p",                   type=int,   default=80,                  help="Protector threshold")
    parser.add_argument("--b_a",                   type=int,   default=3,                   help="Attacker budget")
    parser.add_argument("--strategy_propagation",  type=int,   default=0, choices=[-1, 0],  help="Protector constr. propagation type")
    parser.add_argument("--structure_propagation", type=int,   default=1, choices=[-1, 0, 1, 2], help="Attacker constr. propagation type") 
    parser.add_argument("--seed",                  type=int,   default=1)
    parser.add_argument("--time_limit",            type=float, default=60 * 60 * 3)

    args = parser.parse_args()
    BNB_status_logger.info(args)

    run_SLURM(
        n=args.n, k=args.k, b_p=args.b_p, b_a=args.b_a,
        strategy_propagation=args.strategy_propagation,
        structure_propagation=args.structure_propagation,
        seed=args.seed, time_limit=args.time_limit
    )
