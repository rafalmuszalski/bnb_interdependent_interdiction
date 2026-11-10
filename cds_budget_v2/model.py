"""The three Gurobi models of the protector-attacker-defender game."""
from time import perf_counter

import gurobipy as gp
from gurobipy import GRB

from solver_constants import BINARY_THRESHOLD, EPS, GUROBI_SEED, SOFT_MEM_LIMIT_GB, THREADS
from .instance import Instance


class LimitReached(Exception):
    """A solve hit the time limit or Gurobi's memory limit. `reason` is "time_limit" or "mem_limit"."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class NodeCutoff(Exception):
    """The protector's bound reached its Cutoff: nothing in this BnB node beats the incumbent."""


class ThreeLevelGame:
    def __init__(self, inst: Instance, threads: int = THREADS, soft_mem_limit_gb: float = SOFT_MEM_LIMIT_GB,
                 structure_propagation: int = -1, strategy_propagation: int = -1):
        self.created = perf_counter()
        self.inst = inst
        # Value of an infeasible defender (no CDS exists): above any feasible CDS weight.
        # Also the largest value alpha/pi can take, which is what the cut coefficients are built from.
        self.M = sum(inst.weight(i) for i in inst.nodes) + 1
        self.pi_lb = 0.0  # lower bound on pi (no-attack CDS weight), set in init_protector_model
        self.deadline = float("inf")  # perf_counter() time by which every solve must finish

        # Propagation: keep cuts found by the callbacks as permanent constraints.
        #   structure_propagation: -1 = off, 0 = normal constraint, 1 = lazy constraint
        #   strategy_propagation:  -1 = off, 0 = normal constraint
        self.structure_propagation = structure_propagation
        self.strategy_propagation = strategy_propagation
        self.kept_structures = {}  # frozenset(ds) -> (weight, constr) in the attacker model
        self.kept_strategies = {}  # frozenset(attack) -> constr in the protector model

        # The defender's result depends only on which vertices it can't use
        self.defender_cache = {}   # frozenset(unavailable vertices) -> (ds, weight)
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

    def add_cds(self, m: gp.Model, tag: str = "", slack=0):
        """
        Add variables and constraints to `m` that make y a connected dominating set
        (single-commodity flow from a dummy root). Returns the dicts (y, h, g).

        Used by the defender model, and once per scenario by the CCG master problem,
        which passes a `tag` for unique names and a `slack` variable that switches
        domination off when that scenario has no feasible CDS.
        """
        G = self.inst.G
        n = self.inst.num_nodes
        root = n  # index of the dummy root node; flow starts here

        y = {}  # y[i]    = 1 if node i is in the dominating set
        h = {}  # h[i, j] = 1 if arc (i, j) is used;  h[root, i]: arc from the root
        g = {}  # g[i, j] = flow on arc (i, j);       g[root, i]: flow from the root
        for i in G.nodes():
            y[i] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}{tag}")
            h[root, i] = m.addVar(vtype=GRB.BINARY, name=f"h_{root}_{i}{tag}")
            for j in G.neighbors(i):
                h[i, j] = m.addVar(vtype=GRB.BINARY, name=f"h_{i}_{j}{tag}")
                g[i, j] = m.addVar(vtype=GRB.CONTINUOUS, name=f"g_{i}_{j}{tag}")
            g[root, i] = m.addVar(vtype=GRB.CONTINUOUS, name=f"g_{root}_{i}{tag}")

        # (2) every node is dominated
        for i in G.nodes():
            m.addConstr(y[i] + gp.quicksum(y[j] for j in G.neighbors(i)) + slack >= 1)

        # flow conservation: each DS node consumes one unit of flow
        for i in G.nodes():
            m.addConstr(
                gp.quicksum(g[j, i] for j in G.neighbors(i)) + g[root, i]
                - gp.quicksum(g[i, j] for j in G.neighbors(i))
                == y[i]
            )

        # (6) each edge used in at most one direction
        for i in G.nodes():
            for j in G.neighbors(i):
                m.addConstr(h[i, j] + h[j, i] <= 1)

        # (7), (8) flow only on selected arcs
        for i in G.nodes():
            m.addConstr(g[root, i] <= h[root, i] * n)
            for j in G.neighbors(i):
                m.addConstr(g[i, j] <= h[i, j] * n)

        # (9) exactly one arc leaves the root
        m.addConstr(gp.quicksum(h[root, i] for i in G.nodes()) == 1)

        # (10)-(12) arcs only between DS nodes
        for i in G.nodes():
            for j in G.neighbors(i):
                m.addConstr(h[i, j] <= y[i])
                m.addConstr(h[i, j] <= y[j])
            m.addConstr(h[root, i] <= y[i])

        return y, h, g


    def init_defender_model(self):
        """ Defender: minimum-weight connected dominating set """
        m = gp.Model("Defender", env=self.env)
        self.y, self.h, self.g = self.add_cds(m)
        m.setObjective(gp.quicksum(self.inst.weight(i) * self.y[i] for i in self.inst.nodes), GRB.MINIMIZE)
        m.update()
        self.defender_model = m

    def init_attacker_model(self):
        """
        Attacker: max alpha, the defender's best-response value.
        Defender responses arrive later as lazy cuts: alpha <= w(D) + (M - w(D)) * sum_{i in D} x_i
        """
        G = self.inst.G

        m = gp.Model("Attacker", env=self.env)
        m.Params.PreCrush = 0
        m.Params.Presolve = 0
        m.Params.Heuristics = 0
        m.Params.LazyConstraints = 1

        self.x = [m.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in G.nodes()]  # x[i] = 1 if i is attacked
        self.alpha = m.addVar(vtype=GRB.CONTINUOUS, obj=1, name="alpha")

        m.addConstr(gp.quicksum(G.nodes[i]["cost"] * self.x[i] for i in G.nodes()) <= self.inst.b_a,
                    name="attack_budget")
        m.addConstr(self.alpha <= self.M)  # alpha can't exceed the infeasible value; bounded before any cuts

        m.ModelSense = GRB.MAXIMIZE
        m.update()
        self.attacker_model = m

    def init_protector_model(self):
        """
        Protector: min pi, the worst attack value the policy leaves open.
        Critical attacks arrive later as lazy cuts: pi >= v(A) - (v(A) - pi_lb) * sum_{i covering A} w_i
        """
        G = self.inst.G

        m = gp.Model("Protector", env=self.env)
        m.Params.PreCrush = 0
        m.Params.Presolve = 0
        m.Params.Heuristics = 0
        m.Params.LazyConstraints = 1

        self.w = [m.addVar(vtype=GRB.BINARY, name=f"w_{i}") for i in G.nodes()]  # w[i] = 1 if i is fortified
        self.pi = m.addVar(vtype=GRB.CONTINUOUS, obj=1, name="pi")

        m.addConstr(gp.quicksum(G.nodes[i]["price"] * self.w[i] for i in G.nodes()) <= self.inst.b_p,
                    name="protector_budget")

        # Initial bound: no attack at all is always an option for the attacker
        # Initial bound: no attack at all is always an option for the attacker. BnB only
        # tightens y bounds, which can only raise the defender's value, so this stays valid.
        _, no_attack_weight = self.solve_defender(attack=[])
        self.pi_lb = no_attack_weight
        m.addConstr(self.pi >= no_attack_weight)

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
    # Cuts
    # -------------------------------------------------------------------------

    def structure_cut(self, weight: float, ds):
        """
        alpha <= w(D) + (M - w(D)) * sum_{i in D} x_i
        Attacking any node of D lets alpha reach M (its largest possible value), so the
        coefficient M - w(D) is the smallest valid one.
        """
        return self.alpha <= weight + (self.M - weight) * gp.quicksum(self.x[i] for i in ds)

    def strategy_coef(self, value: float) -> float:
        """Smallest valid coefficient for a strategy cut of `value`: covering it drops pi to pi_lb."""
        return max(0.0, value - self.pi_lb)

    def strategy_cut(self, value: float, cover):
        """pi >= v(A) - (v(A) - pi_lb) * sum_{i in cover} w_i"""
        return self.pi >= value - self.strategy_coef(value) * gp.quicksum(self.w[i] for i in cover)

    @staticmethod
    def strategy_cover(inst, attack):
        """Nodes whose fortification would protect at least one attacked node."""
        return set(attack).union(*(inst.protectors_of[i] for i in attack))

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def critical_structure_callback(self, model: gp.Model, where: int):
        """
        Attacker lazy cuts. For each incumbent attack, solve the defender. If the best
        DS is cheaper than the attacker's alpha, add it as a cut:
            alpha <= w(D) + (M - w(D)) * sum_{i in D} x_i
        """
        if where != GRB.Callback.MIPSOL:
            return

        x_sol = model.cbGetSolution(self.x)
        attack = [i for i in self.inst.nodes if x_sol[i] > BINARY_THRESHOLD]

        ds, ds_weight = self.solve_defender(attack)
        if ds_weight < model.cbGet(GRB.Callback.MIPSOL_OBJ):
            model.cbLazy(self.structure_cut(ds_weight, ds))
            self.structures_added += 1
            self.distinct_structures.add(frozenset(ds))
            self.new_structures.append((ds_weight, ds))  # so callers can keep the cut

    def critical_strategy_callback(self, model: gp.Model, where: int):
        """
        Protector lazy cuts. For each incumbent policy, solve the A-D subgame. If the
        best attack is worth more than the protector's pi, the protector must cover it:
            pi >= v(A) - (v(A) - pi_lb) * sum_{i covering A} w_i
        """
        if where != GRB.Callback.MIPSOL:
            return

        w_sol = model.cbGetSolution(self.w)
        policy = [i for i in self.inst.nodes if w_sol[i] > BINARY_THRESHOLD]

        attack, attack_value = self.solve_attacker(policy)
        if attack_value > model.cbGet(GRB.Callback.MIPSOL_OBJ):
            cover = self.strategy_cover(self.inst, attack)
            model.cbLazy(self.strategy_cut(attack_value, cover))
            self.strategies_added += 1
            self.distinct_strategies.add(frozenset(attack))
            self.new_strategies.append((attack_value, attack))  # kept after the solve, if propagating

    # -------------------------------------------------------------------------
    # Propagation
    # -------------------------------------------------------------------------

    def keep_structures(self, structures: list):
        """Add critical structures to the attacker model as permanent constraints."""
        for weight, ds in structures:
            if frozenset(ds) in self.kept_structures:
                continue
            constr = self.attacker_model.addConstr(self.structure_cut(weight, ds))
            constr.Lazy = self.structure_propagation
            self.kept_structures[frozenset(ds)] = (weight, constr)
        self.attacker_model.update()

    def keep_strategies(self, strategies: list):
        """Add critical strategies to the protector model as permanent constraints."""
        for value, attack in strategies:
            if frozenset(attack) in self.kept_strategies:
                continue
            cover = self.strategy_cover(self.inst, attack)
            constr = self.protector_model.addConstr(self.strategy_cut(value, cover))
            self.kept_strategies[frozenset(attack)] = (constr, cover)
        self.protector_model.update()

    def set_structure_cuts(self, blocked: set):
        """
        Switch off kept structures that use a vertex the defender can't use (`blocked`),
        and switch the rest on. RHS is set from scratch, so there is nothing to undo.
        """
        for ds, (weight, constr) in self.kept_structures.items():
            constr.RHS = self.M if ds & blocked else weight  # alpha <= M is no restriction
        self.attacker_model.update()

    def update_strategy_cuts(self):
        """
        A kept attack's value is the cheapest DS without it, which depends on the
        current y bounds (BnB branching). Re-solve the defender for each and set the RHS,
        and the cover coefficients to match (the tight coefficient depends on the value).
        """
        for attack, (constr, cover) in self.kept_strategies.items():
            _, value = self.solve_defender(list(attack))
            constr.RHS = value
            coef = self.strategy_coef(value)
            for i in cover:
                self.protector_model.chgCoeff(constr, self.w[i], coef)
        self.protector_model.update()

    # -------------------------------------------------------------------------
    # Solves
    # -------------------------------------------------------------------------

    def solve_defender(self, attack: list[int]):
        """
        Min-weight CDS with the attacked nodes removed. Doesn't apply the protector's
        policy (that's BnB's job). y bounds may be set by BnB branching, so attacked
        nodes are restored to their previous bound afterwards, not to 1.
        Results are cached by the set of unavailable vertices (attacked or y.UB = 0).
        """
        unavailable = frozenset(attack) | {i for i in self.inst.nodes if self.y[i].UB < BINARY_THRESHOLD}
        self.defender_calls += 1
        if unavailable in self.defender_cache:
            self.defender_cache_hits += 1
            return self.defender_cache[unavailable]

        original_ub = {i: self.y[i].UB for i in attack}
        for i in attack:
            self.y[i].UB = 0
        self.defender_model.update()

        self.optimize(self.defender_model, "defender")

        status = self.defender_model.Status
        if status == GRB.OPTIMAL:
            ds = [i for i in self.inst.nodes if self.y[i].X > BINARY_THRESHOLD]
            weight = sum(self.inst.weight(i) for i in ds)
        elif status == GRB.INFEASIBLE:
            ds, weight = [], self.M
        else:
            raise ValueError(f"Unexpected defender status: {status}")

        for i in attack:
            self.y[i].UB = original_ub[i]
        self.defender_model.update()

        self.defender_cache[unavailable] = (ds, weight)
        return ds, weight

    def solve_attacker(self, policy: list[int]):
        """A-D subgame for a fixed policy: nodes protected by the policy can't be attacked."""
        for i in policy:
            for j in self.inst.protected_by[i]:
                self.x[j].UB = 0
        self.attacker_model.update()

        self.new_structures = []  # (weight, ds) of each cut the callback adds during this solve
        self.optimize(self.attacker_model, "attacker", self.critical_structure_callback)

        status = self.attacker_model.Status
        if status != GRB.OPTIMAL:
            raise ValueError(f"Unexpected attacker status: {status}")
        attack = [i for i in self.inst.nodes if self.x[i].X > BINARY_THRESHOLD]
        value = self.attacker_model.ObjVal

        for i in self.inst.nodes:
            self.x[i].UB = 1
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
            policy = [i for i in self.inst.nodes if self.w[i].X > BINARY_THRESHOLD]
            value = self.protector_model.ObjVal
        elif status == GRB.INFEASIBLE:
            policy, value = [], self.M
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
        Solve P, then re-solve A-D and D for the optimal policy to recover the attack
        and structure, and check that all three values agree.
        """
        policy, p_value = self.solve_protector()
        attack, a_value = self.solve_attacker(policy)
        structure, d_value = self.solve_defender(attack)

        values = (p_value, a_value, d_value)
        if max(values) - min(values) > EPS:
            raise ValueError(f"Objective mismatch: p={p_value}, a={a_value}, d={d_value}")

        return policy, attack, structure, d_value


