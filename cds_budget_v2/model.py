"""The three Gurobi models of the protector-attacker-defender game."""
import gurobipy as gp
from gurobipy import GRB

from .instance import Instance


class ThreeLevelGame:
    def __init__(self, inst: Instance, threads: int = 8, soft_mem_limit_gb: float = 16):
        self.inst = inst
        self.M = 1e5

        self.env = gp.Env(empty=True)
        self.env.setParam("LogToConsole", 0)
        self.env.setParam("Seed", inst.seed)
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
        Defender responses arrive later as lazy cuts: alpha <= w(D) + M * sum_{i in D} x_i
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
        m.addConstr(self.alpha <= self.M + 1)  # keeps alpha bounded before any cuts exist

        m.ModelSense = GRB.MAXIMIZE
        m.update()
        self.attacker_model = m

    def init_protector_model(self):
        """
        Protector: min pi, the worst attack value the policy leaves open.
        Critical attacks arrive later as lazy cuts: pi >= w(A) - M * sum_{i covering A} w_i
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
        _, no_attack_weight = self.solve_defender(attack=[])
        m.addConstr(self.pi >= no_attack_weight)

        m.update()
        self.protector_model = m

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def critical_structure_callback(self, model: gp.Model, where: int):
        """
        Attacker lazy cuts. For each incumbent attack, solve the defender. If the best
        DS is cheaper than the attacker's alpha, add it as a cut:
            alpha <= w(D) + M * sum_{i in D} x_i
        """
        if where != GRB.Callback.MIPSOL:
            return

        x_sol = model.cbGetSolution(self.x)
        attack = [i for i in self.inst.nodes if x_sol[i] > 0.5]

        ds, ds_weight = self.solve_defender(attack)
        if ds_weight < model.cbGet(GRB.Callback.MIPSOL_OBJ):
            model.cbLazy(self.alpha <= ds_weight + self.M * gp.quicksum(self.x[i] for i in ds))
            self.new_structures.append((ds_weight, ds))  # so callers can keep the cut

    def critical_strategy_callback(self, model: gp.Model, where: int):
        """
        Protector lazy cuts. For each incumbent policy, solve the A-D subgame. If the
        best attack is worth more than the protector's pi, the protector must cover it:
            pi >= w(A) - M * sum_{i covering A} w_i
        """
        if where != GRB.Callback.MIPSOL:
            return

        w_sol = model.cbGetSolution(self.w)
        policy = [i for i in self.inst.nodes if w_sol[i] > 0.5]

        attack, attack_value = self.solve_attacker(policy)
        if attack_value > model.cbGet(GRB.Callback.MIPSOL_OBJ):
            # nodes whose fortification would protect at least one attacked node
            cover = set(attack).union(*(self.inst.protectors_of[i] for i in attack))
            model.cbLazy(self.pi >= attack_value - self.M * gp.quicksum(self.w[i] for i in cover))

    # -------------------------------------------------------------------------
    # Solves
    # -------------------------------------------------------------------------

    def solve_defender(self, attack: list[int]):
        """
        Min-weight CDS with the attacked nodes removed. Doesn't apply the protector's
        policy (that's BnB's job). y bounds may be set by BnB branching, so attacked
        nodes are restored to their previous bound afterwards, not to 1.
        """
        original_ub = {i: self.y[i].UB for i in attack}
        for i in attack:
            self.y[i].UB = 0
        self.defender_model.update()

        self.defender_model.optimize()

        status = self.defender_model.Status
        if status == GRB.OPTIMAL:
            ds = [i for i in self.inst.nodes if self.y[i].X > 0.5]
            weight = sum(self.inst.weight(i) for i in ds)
        elif status == GRB.INFEASIBLE:
            ds, weight = [], self.M
        else:
            raise ValueError(f"Unexpected defender status: {status}")

        for i in attack:
            self.y[i].UB = original_ub[i]
        self.defender_model.update()

        return ds, weight

    def solve_attacker(self, policy: list[int]):
        """A-D subgame for a fixed policy: nodes protected by the policy can't be attacked."""
        for i in policy:
            for j in self.inst.protected_by[i]:
                self.x[j].UB = 0
        self.attacker_model.update()

        self.new_structures = []  # (weight, ds) of each cut the callback adds during this solve
        self.attacker_model.optimize(self.critical_structure_callback)

        status = self.attacker_model.Status
        if status != GRB.OPTIMAL:
            raise ValueError(f"Unexpected attacker status: {status}")
        attack = [i for i in self.inst.nodes if self.x[i].X > 0.5]
        value = self.attacker_model.ObjVal

        for i in self.inst.nodes:
            self.x[i].UB = 1
        self.attacker_model.update()

        return attack, value

    def solve_protector(self):
        self.protector_model.optimize(self.critical_strategy_callback)

        status = self.protector_model.Status
        if status == GRB.OPTIMAL:
            policy = [i for i in self.inst.nodes if self.w[i].X > 0.5]
            value = self.protector_model.ObjVal
        elif status == GRB.INFEASIBLE:
            policy, value = [], self.M
        else:
            raise ValueError(f"Unexpected protector status: {status}")

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
        if max(values) - min(values) > 1e-5:
            raise ValueError(f"Objective mismatch: p={p_value}, a={a_value}, d={d_value}")

        return policy, attack, structure, d_value


