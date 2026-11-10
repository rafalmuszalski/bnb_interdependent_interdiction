"""Problem data: the graph, who protects whom, and the budgets. No Gurobi here."""
import random
from dataclasses import dataclass

import networkx as nx


@dataclass(frozen=True)
class Instance:
    G: nx.Graph
    protected_by: list[list[int]]   # protected_by[i]: nodes that fortifying i protects  (Q(i) in the paper)
    protectors_of: list[list[int]]  # protectors_of[i]: nodes whose fortification protects i  (Q^{-1}(i))
    b_p: int                        # protector target, in percent (see ThreeLevelGame.init_protector_model)
    b_a: int                        # attacker budget
    seed: int

    @property
    def num_nodes(self) -> int:
        return self.G.number_of_nodes()

    @property
    def nodes(self):
        return self.G.nodes()

    def weight(self, i: int) -> float:
        return self.G.nodes[i]["weight"]


def small_world_clique(n: int, clique_size: int, protectors: int, bridges: int,
                       b_p: int, b_a: int, seed: int) -> Instance:
    """
    n / clique_size disjoint cliques, with up to `bridges` random edges between every
    pair of cliques. Each node i can be protected by itself plus `protectors` random
    neighbours.
    """
    if n % clique_size != 0:
        raise ValueError(f"clique_size ({clique_size}) must divide n ({n}) exactly")

    # A private RNG: identical sequence to the old random.seed(seed) + module-level calls,
    # but doesn't touch global random state.
    rng = random.Random(seed)
    G = nx.Graph()

    cliques = []
    for c in range(n // clique_size):
        nodes = list(range(c * clique_size, (c + 1) * clique_size))
        G.add_nodes_from(nodes)
        G.add_edges_from((u, v) for u in nodes for v in nodes if u < v)
        cliques.append(nodes)

    for a, clique_a in enumerate(cliques):
        for clique_b in cliques[a + 1:]:
            for _ in range(bridges):
                u, v = rng.choice(clique_a), rng.choice(clique_b)
                if not G.has_edge(u, v):
                    G.add_edge(u, v)

    protected_by = [[] for _ in G.nodes()]
    protectors_of = [[] for _ in G.nodes()]
    for i in G.nodes():
        G.nodes[i]["price"] = 1   # protector's cost to fortify i
        G.nodes[i]["cost"] = 1    # attacker's cost to attack i
        G.nodes[i]["weight"] = round(rng.uniform(100, 300), 0)

        neighbors = list(G.neighbors(i))
        rng.shuffle(neighbors)
        # i can be protected by itself or by `protectors` random neighbours
        protectors_of[i] = neighbors[:protectors] + [i]
        for j in protectors_of[i]:
            protected_by[j].append(i)

    return Instance(G=G, protected_by=protected_by, protectors_of=protectors_of,
                    b_p=b_p, b_a=b_a, seed=seed)
