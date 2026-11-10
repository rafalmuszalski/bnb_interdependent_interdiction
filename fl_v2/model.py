"""The three Gurobi models of the protector-attacker-defender facility location game.

    Protector  max pi: assigns crews to patrol regions (w[a, k]); patrolled survivors can't be cut off
    Attacker   min alpha: cuts facility->survivor edges (x[i, j]) to minimize the defender's value
    Defender   max value: stations crews at facilities (z[i, k]) and serves survivors over
               uncut edges (y[i, j]); value = DISTANCE_PENALTY * served - total distance

Interdependency: a crew that patrols (protector) can't also be stationed (defender).
ThreeLevelGame ignores it (the defender may use any crew); BranchAndBound enforces it.
"""
from time import perf_counter

import gurobipy as gp
from gurobipy import GRB

from solver_constants import (BIG_M, BINARY_THRESHOLD, DEFENDER_MIP_GAP, DISTANCE_PENALTY, EPS,
                        GUROBI_SEED, SOFT_MEM_LIMIT_GB, THREADS)
from .instance import Instance


class LimitReached(Exception):
    """A solve hit the time limit or Gurobi's memory limit. `reason` is "time_limit" or "mem_limit"."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class NodeCutoff(Exception):
    """The protector's bound fell to its Cutoff: nothing in this BnB node beats the incumbent."""


class ThreeLevelGame:
    def __init__(self, inst: Instance, threads: int = THREADS, soft_mem_limit_gb: float = SOFT_MEM_LIMIT_GB,
                 structure_propagation: int = -1, strategy_propagation: int = -1, anti_symmetry: bool = False):
        self.created = perf_counter()
        self.inst = inst
        self.M = BIG_M
        self.deadline = float("inf")  # perf_counter() time by which every solve must finish
        self.anti_symmetry = anti_symmetry

        # Propagation: keep cuts found by the callbacks as permanent constraints.
        #   structure_propagation: -1 = off, 0 = normal constraint, 1 = lazy constraint
        #   strategy_propagation:  -1 = off, 0 = normal constraint
        self.structure_propagation = structure_propagation
        self.strategy_propagation = strategy_propagation
        self.kept_structures = {}  # frozenset(allocation) -> (value, crews stationed, constr) in the attacker model
        self.kept_strategies = {}  # frozenset(attack) -> constr in the protector model

        # The defender's result depends only on the cut edges and the crews it can't use
        self.defender_cache = {}      # (frozenset(attack), frozenset(blocked crews)) -> (allocation, location, value)
        self.defender_calls = 0       # every solve_defender() call
        self.defender_cache_hits = 0  # calls answered from defender_cache (no Gurobi solve)

        # Statistics. Solve times are exclusive: time spent in nested solves (inside a
        # callback) goes to the nested model, not its parent.
        self.solve_time = {"protector": 0.0, "attacker": 0.0, "defender": 0.0}
        self._nested_time = []         # stack: time of solves nested in each running solve
        self.strategies_added = 0      # critical strategy cuts (protector), every time one is added
        self.structures_added = 0      # critical structure cuts (attacker), every time one is added
        self.distinct_strategies = set()
        self.distinct_structures = set()

        self.env = gp.Env(empty=True)
        self.env.setParam("LogToConsole", 0)
        self.env.setParam("Seed", GUROBI_SEED)
        self.env.setParam("Threads", threads)
        self.env.setParam("SoftMemLimit", soft_mem_limit_gb)
        self.env.start()

        self.init_defender_model()
        self.init_attacker_model()
        self.init_protector_model()

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------

    def init_defender_model(self):
        """Defender: station crews and serve survivors, maximizing DISTANCE_PENALTY * served - distance."""
        inst = self.inst
        m = gp.Model("Defender", env=self.env)
        m.Params.MIPGap = DEFENDER_MIP_GAP

        self.y = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}") for (i, j) in inst.edges}  # serve j from i
        self.z = {(i, k): m.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}")                              # crew k at facility i
                  for i in inst.facilities for k in inst.crews}
        self.zeta = {k: m.addVar(vtype=GRB.BINARY, name=f"zeta_{k}") for k in inst.crews}            # crew k stationed

        for k in inst.crews:
            m.addConstr(self.zeta[k] == gp.quicksum(self.z[i, k] for i in inst.facilities), name="crew_stationed")
            m.addConstr(gp.quicksum(self.z[i, k] for i in inst.facilities) <= 1, name="crew_at_most_one_facility")
        # each crew serves one survivor (capacity 1)
        for i in inst.facilities:
            m.addConstr(gp.quicksum(self.y[i, j] for j in inst.facility_to_survivors[i])
                        <= gp.quicksum(self.z[i, k] for k in inst.crews), name="facility_capacity")
        for j in inst.survivors:
            m.addConstr(gp.quicksum(self.y[i, j] for i in inst.survivor_to_facilities[j]) <= 1,
                        name="survivor_served_once")

        m.setObjective(DISTANCE_PENALTY * gp.quicksum(self.y.values())
                       - gp.quicksum(inst.distances[e] * self.y[e] for e in inst.edges), GRB.MAXIMIZE)
        m.update()
        self.defender_model = m

    def init_attacker_model(self):
        """
        Attacker: min alpha, the defender's best-response value.
        Defender responses arrive later as lazy cuts: alpha >= value(Y) - M * sum_{(i,j) in Y} x_ij
        """
        inst = self.inst
        m = gp.Model("Attacker", env=self.env)
        m.Params.PreCrush = 0
        m.Params.Presolve = 0
        m.Params.Heuristics = 0
        m.Params.LazyConstraints = 1

        self.x = {(i, j): m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}") for (i, j) in inst.edges}  # edge cut
        self.alpha = m.addVar(vtype=GRB.CONTINUOUS, obj=1, name="alpha")

        m.addConstr(gp.quicksum(self.x.values()) <= inst.b_a, name="attack_budget")
        m.addConstr(self.alpha >= -1)  # keeps alpha bounded before any cuts exist

        m.ModelSense = GRB.MINIMIZE
        m.update()
        self.attacker_model = m

    def init_protector_model(self):
        """
        Protector: max pi, the worst attack value the policy allows.
        Critical attacks arrive later as lazy cuts: pi <= value(A) + M * sum_{regions of A, crews} w_ak
        """
        inst = self.inst
        m = gp.Model("Protector", env=self.env)
        m.Params.PreCrush = 0
        m.Params.Presolve = 0
        m.Params.Heuristics = 0
        m.Params.LazyConstraints = 1

        self.w = {(a, k): m.addVar(vtype=GRB.BINARY, name=f"w_{a}_{k}")  # crew k patrols region a
                  for a in inst.regions for k in inst.crews}
        self.omega = {k: m.addVar(vtype=GRB.BINARY, name=f"omega_{k}") for k in inst.crews}  # crew k patrols
        self.pi = m.addVar(vtype=GRB.CONTINUOUS, obj=1, name="pi")

        for k in inst.crews:
            m.addConstr(self.omega[k] == gp.quicksum(self.w[a, k] for a in inst.regions), name="crew_patrols")
        m.addConstr(gp.quicksum(self.w.values()) <= inst.num_crews, name="patrols_at_most_crews")
        m.addConstr(gp.quicksum(self.w.values()) <= inst.b_p, name="protector_budget")
        for a in inst.regions:
            m.addConstr(gp.quicksum(self.w[a, k] for k in inst.crews) <= 1, name="one_crew_per_region")
        for k in inst.crews:
            m.addConstr(gp.quicksum(self.w[a, k] for a in inst.regions) <= 1, name="one_region_per_crew")
        if self.anti_symmetry:  # use crews in index order
            for k in inst.crews[:-1]:
                m.addConstr(self.omega[k] >= self.omega[k + 1], name="anti_symmetry")

        # Initial bound: the attacker can do no better than no attack at all
        _, _, no_attack_value = self.solve_defender(attack=[])
        m.addConstr(self.pi <= no_attack_value)

        m.ModelSense = GRB.MAXIMIZE
        m.update()
        self.protector_model = m

    def optimize(self, m: gp.Model, name: str, callback=None):
        """
        m.optimize() under the deadline, timed into solve_time[name].
        Raises LimitReached on the time or memory limit.
        Gurobi ignores exceptions raised in a callback, so we catch them, stop the
        solve, and re-raise them here (this is how a limit in a nested solve gets out).
        """
        def guarded(model, where):
            if model._error is None:
                try:
                    callback(model, where)
                except Exception as e:
                    model._error = e
                    model.terminate()

        m._error = None
        m.Params.TimeLimit = max(0.0, self.deadline - perf_counter())
        start = perf_counter()
        self._nested_time.append(0.0)
        try:
            m.optimize(guarded if callback else None)
        finally:
            elapsed = perf_counter() - start
            self.solve_time[name] += elapsed - self._nested_time.pop()
            if self._nested_time:
                self._nested_time[-1] += elapsed

        if m._error is not None:
            raise m._error
        if m.Status == GRB.TIME_LIMIT:
            raise LimitReached("time_limit")
        if m.Status == GRB.MEM_LIMIT:
            raise LimitReached("mem_limit")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def critical_structure_callback(self, model: gp.Model, where: int):
        """
        Attacker lazy cuts. For each incumbent attack, solve the defender. If the defender
        does better than the attacker's alpha, add its allocation Y as a cut:
            alpha >= value(Y) - M * sum_{(i,j) in Y} x_ij
        """
        if where != GRB.Callback.MIPSOL:
            return

        x_sol = model.cbGetSolution(self.x)
        attack = [e for e in self.inst.edges if x_sol[e] > BINARY_THRESHOLD]

        allocation, location, value = self.solve_defender(attack)
        if value > model.cbGet(GRB.Callback.MIPSOL_OBJ):
            model.cbLazy(self.alpha >= value - self.M * gp.quicksum(self.x[e] for e in allocation))
            self.structures_added += 1
            self.distinct_structures.add(frozenset(allocation))
            crews = frozenset(k for (_, k) in location)
            self.new_structures.append((value, allocation, crews))  # so callers can keep the cut

    def critical_strategy_callback(self, model: gp.Model, where: int):
        """
        Protector lazy cuts. For each incumbent policy, solve the A-D subgame. If the best
        attack leaves less than the protector's pi, the protector must patrol one of its regions:
            pi <= value(A) + M * sum_{regions of A, crews} w_ak
        """
        if where != GRB.Callback.MIPSOL:
            return

        w_sol = model.cbGetSolution(self.w)
        policy = [(a, k) for (a, k) in self.w if w_sol[a, k] > BINARY_THRESHOLD]

        attack, attack_value = self.solve_attacker(policy)
        if attack_value < model.cbGet(GRB.Callback.MIPSOL_OBJ):
            model.cbLazy(self.pi <= attack_value + self.M * self._cover(attack))
            self.strategies_added += 1
            self.distinct_strategies.add(frozenset(attack))
            self.new_strategies.append((attack_value, attack))  # kept after the solve, if propagating

    def _cover(self, attack: list) -> gp.LinExpr:
        """Sum of the w's whose patrol would protect at least one survivor of `attack`."""
        regions = {self.inst.survivor_to_region[j] for (_, j) in attack}
        return gp.quicksum(self.w[a, k] for a in sorted(regions) for k in self.inst.crews)

    # -------------------------------------------------------------------------
    # Propagation
    # -------------------------------------------------------------------------

    def keep_structures(self, structures: list):
        """Add critical structures to the attacker model as permanent constraints."""
        for value, allocation, crews in structures:
            if frozenset(allocation) in self.kept_structures:
                continue
            constr = self.attacker_model.addConstr(
                self.alpha >= value - self.M * gp.quicksum(self.x[e] for e in allocation))
            constr.Lazy = self.structure_propagation
            self.kept_structures[frozenset(allocation)] = (value, crews, constr)
        self.attacker_model.update()

    def keep_strategies(self, strategies: list):
        """Add critical strategies to the protector model as permanent constraints."""
        for value, attack in strategies:
            if frozenset(attack) in self.kept_strategies:
                continue
            constr = self.protector_model.addConstr(self.pi <= value + self.M * self._cover(attack))
            self.kept_strategies[frozenset(attack)] = constr
        self.protector_model.update()

    def set_structure_cuts(self, blocked: set):
        """
        Switch off kept structures whose allocation needs a crew the defender can't use
        (`blocked`), and switch the rest on. RHS is set from scratch, so there is nothing to undo.
        """
        for value, crews, constr in self.kept_structures.values():
            constr.RHS = value - self.M if crews & blocked else value
        self.attacker_model.update()

    def update_strategy_cuts(self):
        """
        A kept attack's value is the defender's best response to it, which depends on the
        crews BnB branching blocks. Re-solve the defender for each and set the RHS.
        """
        for attack, constr in self.kept_strategies.items():
            _, _, value = self.solve_defender(list(attack))
            constr.RHS = value
        self.protector_model.update()

    # -------------------------------------------------------------------------
    # Solves
    # -------------------------------------------------------------------------

    def solve_defender(self, attack: list):
        """
        Best response to the cut edges `attack`. Doesn't apply the protector's policy
        (that's BnB's job): crews blocked by branching have zeta.UB = 0. Cut edges are
        restored to their previous bound afterwards.
        Results are cached by (cut edges, blocked crews).
        Returns (allocation, location, value).
        """
        self.defender_calls += 1
        blocked = frozenset(k for k in self.inst.crews if self.zeta[k].UB < BINARY_THRESHOLD)
        key = (frozenset(attack), blocked)
        if key in self.defender_cache:
            self.defender_cache_hits += 1
            return self.defender_cache[key]

        original_ub = {e: self.y[e].UB for e in attack}
        for e in attack:
            self.y[e].UB = 0
        self.defender_model.update()

        self.optimize(self.defender_model, "defender")

        status = self.defender_model.Status
        if status != GRB.OPTIMAL:  # y = 0, z = 0 is always feasible
            raise ValueError(f"Unexpected defender status: {status}")
        allocation = [e for e in self.inst.edges if self.y[e].X > BINARY_THRESHOLD]
        location = [(i, k) for (i, k) in self.z if self.z[i, k].X > BINARY_THRESHOLD]
        value = self.defender_model.ObjVal

        for e in attack:
            self.y[e].UB = original_ub[e]
        self.defender_model.update()

        self.defender_cache[key] = (allocation, location, value)
        return allocation, location, value

    def solve_attacker(self, policy: list):
        """A-D subgame for a fixed policy: edges into patrolled regions can't be cut."""
        inst = self.inst
        for (a, _) in policy:
            for j in inst.region_to_survivors[a]:
                for i in inst.survivor_to_facilities[j]:
                    self.x[i, j].UB = 0
        self.attacker_model.update()

        self.new_structures = []  # (value, allocation, crews) of each cut the callback adds during this solve
        self.optimize(self.attacker_model, "attacker", self.critical_structure_callback)

        status = self.attacker_model.Status
        if status != GRB.OPTIMAL:
            raise ValueError(f"Unexpected attacker status: {status}")
        attack = [e for e in inst.edges if self.x[e].X > BINARY_THRESHOLD]
        value = self.attacker_model.ObjVal

        for e in inst.edges:
            self.x[e].UB = 1
        self.attacker_model.update()

        # After reading the solution: adding constraints discards it
        if self.structure_propagation >= 0:
            self.keep_structures(self.new_structures)

        return attack, value

    def solve_protector(self):
        self.new_strategies = []  # (value, attack) of each cut the callback adds during this solve
        self.optimize(self.protector_model, "protector", self.critical_strategy_callback)

        status = self.protector_model.Status
        if status == GRB.OPTIMAL:
            policy = [(a, k) for (a, k) in self.w if self.w[a, k].X > BINARY_THRESHOLD]
            value = self.protector_model.ObjVal
        elif status == GRB.INFEASIBLE:
            policy, value = [], -self.M
        elif status == GRB.CUTOFF:
            raise NodeCutoff()
        else:
            raise ValueError(f"Unexpected protector status: {status}")

        # After reading the solution: adding constraints discards it
        if self.strategy_propagation >= 0:
            self.keep_strategies(self.new_strategies)

        return policy, value

    def solve_three_level_game(self):
        """
        Solve P, then re-solve A-D and D for the optimal policy to recover the attack,
        allocation and location, and check that all three values agree.
        """
        policy, p_value = self.solve_protector()
        if p_value <= -self.M:  # infeasible: nothing to recover
            return policy, [], [], [], p_value
        attack, a_value = self.solve_attacker(policy)
        allocation, location, d_value = self.solve_defender(attack)

        values = (p_value, a_value, d_value)
        if max(values) - min(values) > EPS:
            raise ValueError(f"Objective mismatch: p={p_value}, a={a_value}, d={d_value}")

        return policy, attack, allocation, location, d_value
