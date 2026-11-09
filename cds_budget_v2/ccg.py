"""
Column-and-constraint generation (CCG) for the protector-attacker-defender game.

Unlike BnB, interdependency is enforced directly: the defender may not use fortified
nodes. Each iteration:
    1. A-D subgame for the current policy  ->  a critical attack
    2. Add that attack as a scenario to the master (protector) problem: a fresh copy of
       the CDS variables/constraints in which the defender reacts to that attack
    3. Re-solve the master  ->  new policy and a lower bound
until the defender's value for the current policy matches the master's lower bound.

Reuses the defender, attacker and callbacks from ThreeLevelGame; only the protector
(master problem) and the main loop are CCG-specific.
"""
import logging
from time import perf_counter

import gurobipy as gp
from gurobipy import GRB

from .instance import Instance
from .model import ThreeLevelGame

log = logging.getLogger("ccg")


class CCG(ThreeLevelGame):
    def __init__(self, inst: Instance, time_limit: float, **kwargs):
        self.policy = []        # current protection policy; the defender respects it
        self.structure_cuts = []  # (weight, ds, constr): attacker cuts kept across iterations
        self.num_scenarios = 0
        self.time_limit = time_limit
        super().__init__(inst, **kwargs)

    # -------------------------------------------------------------------------
    # Master problem
    # -------------------------------------------------------------------------

    def init_protector_model(self):
        """
        Master problem: min pi over fortification policies w. Starts with no scenarios;
        add_scenario() adds one per critical attack.
        """
        G = self.inst.G
        m = gp.Model("Protector", env=self.env)

        self.pi = m.addVar(vtype=GRB.CONTINUOUS, name="pi")
        self.w = {}  # w[i] = 1 if i is fortified
        self.u = {}  # u[i] = 1 if i is protected (by itself or a fortified neighbour)
        for i in G.nodes():
            self.w[i] = m.addVar(vtype=GRB.BINARY, name=f"w_{i}")
            self.u[i] = m.addVar(vtype=GRB.BINARY, name=f"u_{i}")

        m.setObjective(self.pi, GRB.MINIMIZE)

        # (12) fortification budget
        m.addConstr(gp.quicksum(G.nodes[i]["price"] * self.w[i] for i in G.nodes()) <= self.inst.b_p,
                    name="protector_budget")

        # (13)-(14) i is protected iff at least one of its protectors is fortified
        for i in G.nodes():
            for j in self.inst.protectors_of[i]:
                m.addConstr(self.u[i] >= self.w[j], name=f"protection_lower_{i}")
            m.addConstr(self.u[i] <= gp.quicksum(self.w[j] for j in self.inst.protectors_of[i]),
                        name=f"protection_upper_{i}")

        m.update()
        self.protector_model = m

    def add_scenario(self, attack: list[int]):
        """
        Add a scenario for `attack`: a copy of the defender's CDS whose weight bounds pi.
        Attacked nodes are usable only if the policy protects them; fortified nodes never.
        """
        m = self.protector_model
        k = self.num_scenarios
        self.num_scenarios += 1

        infeasible = m.addVar(vtype=GRB.BINARY, name=f"infeasible_{k}")  # 1 if no CDS exists
        y, _, _ = self.add_cds(m, tag=f"_s{k}", slack=infeasible)

        # (23) pi bounds this scenario's DS weight (or M if infeasible)
        m.addConstr(self.pi >= gp.quicksum(self.inst.weight(i) * y[i] for i in self.inst.nodes)
                    + self.M * infeasible)

        for i in self.inst.nodes:
            m.addConstr(y[i] <= 1 - self.w[i])     # (38) fortified nodes can't be in the DS
            if i in attack:
                m.addConstr(y[i] <= self.u[i])     # (39) attacked nodes only if protected

    # -------------------------------------------------------------------------
    # Interdependent defender and A-D subgame
    # -------------------------------------------------------------------------

    def solve_defender(self, attack: list[int]):
        """CCG's defender is interdependent: fortified nodes are unavailable too."""
        blocked = list(dict.fromkeys(self.policy + attack))  # union, order kept
        return super().solve_defender(blocked)

    def solve_attacker(self, policy: list[int]):
        """
        A-D subgame for `policy`, keeping every DS cut found so far. Cuts using a fortified
        node can't be realised under this policy, so they're relaxed by +M for this solve.
        """
        self.policy = policy
        fortified = set(policy)
        relaxed = [c for _, ds, c in self.structure_cuts if fortified & set(ds)]
        for c in relaxed:
            c.RHS += self.M
        self.attacker_model.update()

        attack, value = super().solve_attacker(policy)

        for c in relaxed:
            c.RHS -= self.M
        for weight, ds in self.new_structures:
            c = self.attacker_model.addConstr(
                self.alpha <= weight + self.M * gp.quicksum(self.x[i] for i in ds))
            self.structure_cuts.append((weight, ds, c))
        self.attacker_model.update()

        return attack, value

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    def solve(self):
        """Returns (weight, policy, attack, structure) of the best solution found."""
        start = perf_counter()
        lower_bound = float("-inf")
        best = (float("inf"), [], [], [])
        policy = []
        self.iterations = 0

        log.info(f"{'It':>4} | {'P.Cols':^6}  {'P.Rows':^6} | {'LB':^8}  {'D.Value':^8} | "
                 f"{'UB':^8} {'Gap':^10} {'Time(s)':>9}")

        while perf_counter() - start < self.time_limit:
            attack, _ = self.solve_attacker(policy)
            structure, weight = self.solve_defender(attack)

            if weight < best[0]:
                best = (weight, policy, attack, structure)

            gap = (best[0] - lower_bound) / abs(lower_bound) if abs(lower_bound) not in (0, float("inf")) \
                else float("inf")
            log.info(f"{self.iterations:4d} | {self.protector_model.NumVars:^6}  "
                     f"{self.protector_model.NumConstrs:^6} | {lower_bound:>8g}  {weight:>8g} | "
                     f"{best[0]:^8g} {gap:^10.5g} {perf_counter() - start:>9.3f}")

            # Converged: the policy's true value matches the master's lower bound
            if weight <= lower_bound + 1e-5:
                break

            self.add_scenario(attack)

            remaining = self.time_limit - (perf_counter() - start)
            if remaining <= 0:
                break
            self.protector_model.Params.TimeLimit = remaining
            self.protector_model.optimize()

            status = self.protector_model.Status
            if status == GRB.TIME_LIMIT:
                log.info("Time limit reached in master problem")
                break
            if status != GRB.OPTIMAL:
                raise ValueError(f"Unexpected master problem status: {status}")
            policy = [i for i in self.inst.nodes if self.w[i].X > 0.5]
            lower_bound = self.protector_model.ObjVal
            self.iterations += 1

        self.lower_bound = lower_bound
        self.time = perf_counter() - start
        return best
