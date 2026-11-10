"""
Branch-and-bound over protector-defender interdependencies (facility location).

The game is a maximization: the protector maximizes the defender's value, so a node's
value is an UPPER bound and the incumbent is the LOWER bound. Each node solves the game
with the defender INDEPENDENT of the protector (it may station patrolling crews).
A node is classified as:
    infea   the protector has no feasible policy under the branching
    frac    a crew both patrols (policy) and is stationed (defender)  -> branch on it
    sopt    no overlap, but the attacker over-hedged against defenders that can't exist
            under the policy (the heuristic finds a lower true value)  -> branch
    int     the solution is valid for the interdependent game  -> candidate incumbent
    kill    the protector's bound fell to the incumbent mid-solve (Gurobi Cutoff)
Branching on a patrol (region a, crew k):
    D.Branch (fix=0): crew k may not patrol region a
    P.Branch (fix=1): crew k patrols region a and may not be stationed
"""
import heapq
import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

from gurobipy import GRB

from solver_constants import EPS
from .model import LimitReached, NodeCutoff, ThreeLevelGame

log = logging.getLogger("bnb")
node_log = logging.getLogger("nodes")   # one row per processed node, for debugging


@dataclass
class Node:
    name: int
    parent: Optional[int]
    depth: int
    bound: float                     # upper bound: parent's value until solved, then its own
    branch_var: Optional[tuple] = None  # (region, crew)
    branch_fix: Optional[int] = None    # 0 = D.Branch, 1 = P.Branch
    status: str = "open"
    policy: list = field(default_factory=list)      # patrols (region, crew)
    attack: list = field(default_factory=list)      # cut edges (facility, survivor)
    allocation: list = field(default_factory=list)  # served edges (facility, survivor)
    location: list = field(default_factory=list)    # stationed crews (facility, crew)
    to_branch_on: list = field(default_factory=list)

    def __lt__(self, other: "Node") -> bool:
        """Best-bound first (highest); ties go to the D.Branch (fix=0) child."""
        if abs(self.bound - other.bound) < EPS:
            return self.branch_fix < other.branch_fix
        return self.bound > other.bound


@dataclass
class Solution:
    value: float = float("-inf")
    policy: list = field(default_factory=list)
    attack: list = field(default_factory=list)
    allocation: list = field(default_factory=list)
    location: list = field(default_factory=list)


class BranchAndBound:
    def __init__(self, game: ThreeLevelGame, time_limit: float):
        self.game = game
        self.time_limit = time_limit

        self.nodes: list[Node] = []   # every node, indexed by name
        self.queue: list[Node] = []   # open nodes, best (highest) bound first
        self.best = Solution()
        self.lower_bound = float("-inf")  # incumbent value
        self.upper_bound = float("inf")   # best bound among open nodes
        self.gap = float("inf")
        self.status = "open"  # why solve() stopped: optimal / time_limit / mem_limit
        self.time = 0.0
        self.heuristic_time = 0.0  # wall time in evaluate_policy (its solves also count in game.solve_time)
        # Nodes explored: processing finished with a solve (infea / frac / sopt / int, or kill,
        # which stopped mid-solve on the Cutoff). Not counted: nodes pruned by bound without
        # a solve, nodes interrupted by the time/memory limit, and nodes never popped.
        self.nodes_explored = 0

        self._add_node(Node(name=0, parent=None, depth=0, bound=float("inf")))

    def _add_node(self, node: Node):
        self.nodes.append(node)
        heapq.heappush(self.queue, node)

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    def solve(self) -> Solution:
        start = self.game.created  # the clock starts when the game's models are built
        log.info(f"{'Node':>6} {'Parent':>6} {'Depth':>5} {'Var':>8} {'Fix':>3} {'Status':>6} "
                 f"{'Value':>9} | {'LB':>9} {'UB':>9} {'Gap':>9} {'Open':>5} {'Time(s)':>9}")
        node_log.info(f"{'Node':>6} {'Parent':>6} {'Var':>8} {'Fix':>3} {'Status':>6} {'Value':>9} | "
                      f"Policy | Attack | Allocation | Location")

        self.game.deadline = start + self.time_limit
        self.status = "loaded"

        # Root heuristic: no patrols means no interdependency, so the empty policy's
        # A-D value is a feasible solution. Start with it as the incumbent.
        heuristic_start = perf_counter()
        try:
            self.set_incumbent(self.evaluate_policy([], set()))
            log.info(f"{'root':>6} {'':>6} {'':>5} {'':>8} {'':>3} {'heur':>6} {self.lower_bound:>9g} |")
        except LimitReached as e:  # nothing explored; the root's bound stays inf
            self.heuristic_time += perf_counter() - heuristic_start
            self.status = e.reason
            self.time = perf_counter() - start
            log.info(f"Stopped in the root heuristic: {e.reason}")
            return self.best
        self.heuristic_time += perf_counter() - heuristic_start

        while self.queue:
            node = heapq.heappop(self.queue)
            try:
                self.process_node(node)
                if node.status != "prune":
                    self.nodes_explored += 1
            except LimitReached as e:
                heapq.heappush(self.queue, node)  # unfinished: its bound still counts in the UB
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
        log.info(f"Best: value={self.best.value} policy={self.best.policy} attack={self.best.attack} "
                 f"allocation={self.best.allocation} location={self.best.location}")
        return self.best

    def process_node(self, node: Node):
        g = self.game

        if node.bound <= self.lower_bound:
            node.status = "prune"
            return

        self.reset_models()
        patrols, blocked_crews = self.apply_branching(node)
        g.set_structure_cuts(blocked=blocked_crews)
        g.update_strategy_cuts()

        # Stop the protector solve as soon as its bound shows this node can't beat the incumbent
        g.protector_model.Params.Cutoff = max(self.lower_bound, -GRB.INFINITY)
        try:
            node.policy, node.attack, node.allocation, node.location, node.bound = g.solve_three_level_game()
        except NodeCutoff:
            node.status = "kill"
            node.bound = self.lower_bound  # the bound fell to at most this
            return

        if node.bound <= -g.M:
            node.status = "infea"
            return

        stationed = {k for (_, k) in node.location}
        overlap = sorted((p for p in node.policy if p[1] in stationed), key=lambda p: p[1])  # by crew
        if overlap:
            node.status = "frac"
            node.to_branch_on = overlap
            self.branch(node)
            return

        # No overlap, but was the attacker's value right for this policy?
        heuristic_start = perf_counter()
        try:
            heuristic = self.evaluate_policy(node.policy, blocked_crews)
        finally:
            self.heuristic_time += perf_counter() - heuristic_start
        if heuristic.value < node.bound - EPS:
            # The attacker over-hedged; the heuristic found the policy's true value, which
            # is a feasible solution in its own right.
            if heuristic.value > self.lower_bound:
                self.set_incumbent(heuristic)
            log.info(f"{'':>6} {'':>6} {'':>5} {'':>8} {'':>3} {'heur':>6} {heuristic.value:>9g} | "
                     f"{self.lower_bound:>9g} {self.upper_bound:>9g}")

            node.status = "sopt"
            node.to_branch_on = sorted(set(node.policy) - patrols)
            assert node.to_branch_on, f"Node {node.name}: attacker suboptimal but nothing to branch on"
            self.branch(node)
            return

        node.status = "int"
        if node.bound > self.lower_bound:
            self.set_incumbent(Solution(node.bound, node.policy, node.attack, node.allocation, node.location))

    def branch(self, node: Node):
        var = node.to_branch_on[0]
        for fix in (0, 1):
            self._add_node(Node(name=len(self.nodes), parent=node.name, depth=node.depth + 1,
                                bound=node.bound, branch_var=var, branch_fix=fix))

    def set_incumbent(self, solution: Solution):
        self.best = solution
        self.lower_bound = solution.value

    def update_bounds(self):
        # UB: best bound among open nodes (never below LB)
        self.upper_bound = max(self.queue[0].bound, self.lower_bound) if self.queue else self.lower_bound
        if EPS < self.upper_bound < float("inf"):
            self.gap = (self.upper_bound - self.lower_bound) / self.upper_bound

    # -------------------------------------------------------------------------
    # Model state
    # -------------------------------------------------------------------------

    def reset_models(self):
        """Clear solutions and variable bounds left over from the previous node."""
        g = self.game
        for m in (g.protector_model, g.attacker_model, g.defender_model):
            m.reset(1)  # discards MIP starts / solution info; does NOT reset bounds
        for var in (*g.w.values(), *g.omega.values(), *g.x.values(),
                    *g.y.values(), *g.z.values(), *g.zeta.values()):
            var.LB, var.UB = 0, 1
        for m in (g.protector_model, g.attacker_model, g.defender_model):
            m.update()

    def apply_branching(self, node: Node) -> tuple[set, set]:
        """Apply every branching decision from the node up to the root.
        Returns the patrols fixed on (P.Branch) and their crews, which the defender can't use."""
        g = self.game
        patrols, crews = set(), set()
        current = node
        while current.parent is not None:
            a, k = current.branch_var
            if current.branch_fix == 1:   # P.Branch: crew k must patrol region a, and not be stationed
                g.w[a, k].LB = 1
                g.omega[k].LB = 1
                g.zeta[k].UB = 0
                for i in g.inst.facilities:
                    g.z[i, k].UB = 0
                patrols.add((a, k))
                crews.add(k)
            else:                         # D.Branch: crew k may not patrol region a
                g.w[a, k].UB = 0
            current = self.nodes[current.parent]
        g.protector_model.update()
        g.defender_model.update()
        return patrols, crews

    # -------------------------------------------------------------------------
    # Heuristic
    # -------------------------------------------------------------------------

    def evaluate_policy(self, policy: list, blocked_crews: set) -> Solution:
        """
        True value of `policy` in the interdependent game: patrolling crews can't be
        stationed, then solve A-D. Doesn't touch the incumbent; the caller decides.
        Leaves defender bounds and structure cuts set for this policy; the next node resets them.
        """
        g = self.game
        policy_crews = {k for (_, k) in policy}
        for k in policy_crews:
            g.zeta[k].UB = 0
            for i in g.inst.facilities:
                g.z[i, k].UB = 0
        g.defender_model.update()
        g.set_structure_cuts(blocked=blocked_crews | policy_crews)

        attack, value = g.solve_attacker(policy)
        allocation, location, d_value = g.solve_defender(attack)
        if abs(value - d_value) > EPS:
            raise ValueError(f"Heuristic A-D mismatch: attacker={value}, defender={d_value}")
        return Solution(d_value, policy, attack, allocation, location)

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def log_node(self, node: Node, elapsed: float):
        var = "" if node.branch_var is None else str(node.branch_var)
        fix = "" if node.branch_fix is None else node.branch_fix
        parent = "" if node.parent is None else node.parent
        log.info(f"{node.name:>6} {parent:>6} {node.depth:>5} {var:>8} {fix:>3} {node.status:>6} "
                 f"{node.bound:>9g} | {self.lower_bound:>9g} {self.upper_bound:>9g} {self.gap:>9.4g} "
                 f"{len(self.queue):>5} {elapsed:>9.2f}")
        node_log.info(f"{node.name:>6} {parent:>6} {var:>8} {fix:>3} {node.status:>6} {node.bound:>9g} | "
                      f"{node.policy} | {node.attack} | {node.allocation} | {node.location}")
