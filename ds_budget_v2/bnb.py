"""
Branch-and-bound over protector-defender interdependencies.

Each node solves the three-level game with the defender INDEPENDENT of the protector
(ThreeLevelGame lets the DS use fortified nodes). The node's value is therefore a lower
bound. A node is classified as:
    infea   no feasible DS
    frac    the policy fortifies a node the DS uses  -> branch on it
    sopt    no overlap, but the attacker over-hedged against DSs that can't exist under
            the policy (the heuristic finds a higher true value)  -> branch
    int     the solution is valid for the interdependent game  -> candidate incumbent
    kill    the protector's bound reached the incumbent mid-solve (Gurobi Cutoff)
Branching on vertex v:
    D.Branch (fix=0): v may not be fortified
    P.Branch (fix=1): v is fortified and may not be in the DS
"""
import heapq
import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

from .constants import EPS
from .model import LimitReached, NodeCutoff, ThreeLevelGame

log = logging.getLogger("bnb")
node_log = logging.getLogger("nodes")   # one row per processed node, for debugging


@dataclass
class Node:
    name: int
    parent: Optional[int]
    depth: int
    bound: float                     # lower bound: parent's value until solved, then its own
    branch_var: Optional[int] = None
    branch_fix: Optional[int] = None  # 0 = D.Branch, 1 = P.Branch
    status: str = "open"
    policy: list = field(default_factory=list)
    attack: list = field(default_factory=list)
    structure: list = field(default_factory=list)
    to_branch_on: list = field(default_factory=list)

    def __lt__(self, other: "Node") -> bool:
        """Best-bound first; ties go to the D.Branch (fix=0) child."""
        if abs(self.bound - other.bound) < EPS:
            return self.branch_fix < other.branch_fix
        return self.bound < other.bound


@dataclass
class Solution:
    weight: float = float("inf")
    policy: list = field(default_factory=list)
    attack: list = field(default_factory=list)
    structure: list = field(default_factory=list)


class BranchAndBound:
    def __init__(self, game: ThreeLevelGame, time_limit: float):
        self.game = game
        self.time_limit = time_limit

        self.nodes: list[Node] = []   # every node, indexed by name
        self.queue: list[Node] = []   # open nodes, best bound first
        self.best = Solution()
        self.lower_bound = float("-inf")
        self.upper_bound = float("inf")
        self.gap = float("inf")
        self.status = "open"  # why solve() stopped: optimal / time_limit / mem_limit
        self.time = 0.0
        self.heuristic_time = 0.0  # wall time in evaluate_policy (its solves also count in game.solve_time)
        # Nodes explored: processing finished with a solve (infea / frac / sopt / int, or kill,
        # which stopped mid-solve on the Cutoff). Not counted: nodes pruned by bound without
        # a solve, nodes interrupted by the time/memory limit, and nodes never popped.
        self.nodes_explored = 0

        self._add_node(Node(name=0, parent=None, depth=0, bound=float("-inf")))

    def _add_node(self, node: Node):
        self.nodes.append(node)
        heapq.heappush(self.queue, node)

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    def solve(self) -> Solution:
        start = self.game.created  # the clock starts when the game's models are built
        log.info(f"{'Node':>6} {'Parent':>6} {'Depth':>5} {'Var':>4} {'Fix':>3} {'Status':>6} "
                 f"{'Value':>9} | {'LB':>9} {'UB':>9} {'Gap':>9} {'Open':>5} {'Time(s)':>9}")
        node_log.info(f"{'Node':>6} {'Parent':>6} {'Var':>4} {'Fix':>3} {'Status':>6} {'Value':>9} | "
                      f"{'Policy':<16} {'Strategy':<16} Structure")

        self.game.deadline = start + self.time_limit
        self.status = "loaded" 
        while self.queue:
            node = heapq.heappop(self.queue)
            try:
                self.process_node(node)
                if node.status != "prune":
                    self.nodes_explored += 1
            except LimitReached as e:
                heapq.heappush(self.queue, node)  # unfinished: its bound still counts in the LB
                self.update_bounds()
                self.status = e.reason
                log.info(f"Stopped at node {node.name}: {e.reason}")
                break
            self.update_bounds()
            self.log_node(node, perf_counter() - start)

            if self.gap <= EPS:
                self.status = "optimal"
                log.info("Optimality gap closed")
                break
            if perf_counter() - start >= self.time_limit:
                self.status = "time_limit"
                log.info(f"Time limit reached ({self.time_limit:.0f}s)")
                break
        else:                       # queue ran empty: every node was explored
            self.status = "optimal"

        self.time = perf_counter() - start
        log.info(f"Status: {self.status}  LB={self.lower_bound:g}  UB={self.upper_bound:g}  gap={self.gap:.4g}")
        log.info(f"Best: weight={self.best.weight} policy={self.best.policy} "
                 f"attack={self.best.attack} structure={self.best.structure}")
        return self.best

    def process_node(self, node: Node):
        g = self.game

        if node.bound >= self.upper_bound:
            node.status = "prune"
            return

        self.reset_models()
        fortified = self.apply_branching(node)
        g.set_structure_cuts(blocked=fortified)
        g.update_strategy_cuts()

        # Stop the protector solve as soon as its bound shows this node can't beat the incumbent
        g.protector_model.Params.Cutoff = self.upper_bound
        try:
            node.policy, node.attack, node.structure, node.bound = g.solve_three_level_game()
        except NodeCutoff:
            node.status = "kill"
            node.bound = self.upper_bound  # the bound reached at least this
            return

        if node.bound >= g.M:
            node.status = "infea"
            return

        overlap = sorted(set(node.policy) & set(node.structure))
        if overlap:
            node.status = "frac"
            node.to_branch_on = overlap
            self.branch(node)
            return

        # No overlap, but was the attacker's value right for this policy?
        heuristic_start = perf_counter()
        try:
            heuristic = self.evaluate_policy(node.policy, fortified)
        finally:
            self.heuristic_time += perf_counter() - heuristic_start
        if heuristic.weight > node.bound + EPS:
            # The attacker over-hedged; the heuristic found the policy's true value, which
            # is a feasible solution in its own right.
            improved = heuristic.weight < self.upper_bound
            if improved:
                self.set_incumbent(heuristic)
            log.info(f"{'':>6} {'':>6} {'':>5} {'':>4} {'':>3} {'heur':>6} {heuristic.weight:>9g} | "
                     f"{self.lower_bound:>9g} {self.upper_bound:>9g}")

            node.status = "sopt"
            node.to_branch_on = sorted(set(node.policy) - fortified)
            assert node.to_branch_on, f"Node {node.name}: attacker suboptimal but nothing to branch on"
            self.branch(node)
            return

        node.status = "int"
        if node.bound < self.upper_bound:
            self.set_incumbent(Solution(node.bound, node.policy, node.attack, node.structure))

    def branch(self, node: Node):
        var = node.to_branch_on[0]
        for fix in (0, 1):
            self._add_node(Node(name=len(self.nodes), parent=node.name, depth=node.depth + 1,
                                bound=node.bound, branch_var=var, branch_fix=fix))

    def set_incumbent(self, solution: Solution):
        self.best = solution
        self.upper_bound = solution.weight

    def update_bounds(self):
        # LB: best bound among open nodes (never above UB)
        self.lower_bound = min(self.queue[0].bound, self.upper_bound) if self.queue else self.upper_bound
        if self.upper_bound > EPS and self.upper_bound < float("inf"):
            self.gap = (self.upper_bound - self.lower_bound) / self.upper_bound

    # -------------------------------------------------------------------------
    # Model state
    # -------------------------------------------------------------------------

    def reset_models(self):
        """Clear solutions and variable bounds left over from the previous node."""
        g = self.game
        for m in (g.protector_model, g.attacker_model, g.defender_model):
            m.reset(1)  # discards MIP starts / solution info; does NOT reset bounds
        for i in g.inst.nodes:
            g.w[i].LB, g.w[i].UB = 0, 1
            g.x[i].LB, g.x[i].UB = 0, 1
            g.y[i].LB, g.y[i].UB = 0, 1
        for m in (g.protector_model, g.attacker_model, g.defender_model):
            m.update()

    def apply_branching(self, node: Node) -> set[int]:
        """Apply every branching decision from the node up to the root.
        Returns the vertices fixed as fortified (P.Branch)."""
        g = self.game
        fortified = set()
        current = node
        while current.parent is not None:
            v = current.branch_var
            if current.branch_fix == 1:   # P.Branch: must fortify, and not in the DS
                g.w[v].LB = 1
                g.y[v].UB = 0
                fortified.add(v)
            else:                         # D.Branch: may not fortify
                g.w[v].UB = 0
            current = self.nodes[current.parent]
        g.protector_model.update()
        g.defender_model.update()
        return fortified

    # -------------------------------------------------------------------------
    # Heuristic
    # -------------------------------------------------------------------------

    def evaluate_policy(self, policy: list[int], fortified: set[int]) -> Solution:
        """
        True value of `policy` in the interdependent game: forbid fortified nodes in the
        DS, then solve A-D. Doesn't touch the incumbent; the caller decides.
        Leaves y bounds and structure cuts set for this policy; the next node resets them.
        """
        g = self.game
        for i in policy:
            g.y[i].UB = 0
        g.defender_model.update()
        g.set_structure_cuts(blocked=fortified | set(policy))

        attack, value = g.solve_attacker(policy)
        structure, weight = g.solve_defender(attack)
        if abs(value - weight) > EPS:
            raise ValueError(f"Heuristic A-D mismatch: attacker={value}, defender={weight}")
        return Solution(weight, policy, attack, structure)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def log_node(self, node: Node, elapsed: float):
        var = "" if node.branch_var is None else node.branch_var
        fix = "" if node.branch_fix is None else node.branch_fix
        parent = "" if node.parent is None else node.parent
        log.info(f"{node.name:>6} {parent:>6} {node.depth:>5} {var:>4} {fix:>3} {node.status:>6} "
                 f"{node.bound:>9g} | {self.lower_bound:>9g} {self.upper_bound:>9g} {self.gap:>9.4g} "
                 f"{len(self.queue):>5} {elapsed:>9.2f}")
        node_log.info(f"{node.name:>6} {parent:>6} {var:>4} {fix:>3} {node.status:>6} {node.bound:>9g} | "
                      f"{str(node.policy):<16} {str(node.attack):<16} {node.structure}")
