# -*- coding: utf-8 -*-
import os
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from time import perf_counter
from dataclasses import dataclass
from typing import List

from .logging_setup import CG_status_logger, CG_details_logger


@dataclass(frozen=True)
class DefenderCallbackRecord:
    ds_weight: float
    ds: List[int]
    ds_constr: gp.Constr


class Interdependent_Nested_CnCG:
    def __init__(self, n: int, b_a: int, b_p: int, G: nx.Graph,
                 protected_by, protectors_of, seed: int, time_limit: int = 60 * 60 * 3):
        self.BIG_M = 1e5
        self.epsilon = 1e-8
        self.time_limit = time_limit
        self.n = n
        self.b_a = b_a
        self.b_p = b_p
        self.G = G
        self.protected_by = protected_by
        self.protectors_of = protectors_of
        self.seed = seed

        self.propagated_structures: list[DefenderCallbackRecord] = []

        # Protector variables
        self.w_var = {}  # Fortification variables
        self.u_var = {}  # Protection variables
        self.y_hat_var = {} # DS variables
        self.dummy_hat_var = {}  # Dummy Var (takes 1 if DS is infeasible)
        self.pi_var = None  # Value Func Var

        # Attacker variables
        self.x_var = {}
        self.gamma_var = None

        # Defender variables
        self.y_var = {}

        # Initialize Gurobi Models
        self.env = gp.Env(empty=True)
        self.env.setParam("LogToConsole", 0)
        self.env.setParam("Threads",8)
        self.env.setParam("SoftMemLimit",16)
        self.env.start()
        
        self.defender_model  = self.init_defender()
        self.attacker_model  = self.init_attacker()
        self.protector_model = self.init_protector()

        self.best_feasible_solution = (float('inf'), [], [], [])

        self.A_calls = 0
        self.D_calls = 0
        self.time = 0
        self.time_solving_attacker = 0
        self.time_solving_defender = 0
        self.time_solving_protector = 0
        self.num_crit_att = 0
        self.num_crit_structures = 0
        self.protector_obj_val = float('-inf')
        self.protector_iterations = None
        self.protector_columns = None
        self.protector_rows = None
        self.defender_obj_val = float('-inf')
        self.protection_policy = []
        self.attack_strategy = []
        self.defender_structure = []

    # -------------------------------------------------------------------------
    # Model Initialization
    # -------------------------------------------------------------------------
    
    def init_defender(self):
        m = gp.Model("Defender", env=self.env)

        self.y_var = [
            m.addVar(vtype=GRB.BINARY, obj=self.G.nodes[i]['weight'], name=f"y_{i}")
            for i in self.G.nodes()
        ]

        for i in self.G.nodes():
            m.addConstr(
                self.y_var[i] + gp.quicksum(self.y_var[j] for j in self.G.neighbors(i)) >= 1,
                name=f"ds_{i}"
            )

        m.update()
        return m

    def init_attacker(self):
        m = gp.Model("Attacker", env=self.env)
        m.setParam("LazyConstraints", 1)

        for i in self.G.nodes():
            self.x_var[i] = m.addVar(vtype=GRB.BINARY, name=f'x_{i}')

        self.gamma_var = m.addVar(vtype=GRB.CONTINUOUS, name='gamma')

        m.setObjective(self.gamma_var, GRB.MAXIMIZE)
        m.addConstr(
            gp.quicksum(self.G.nodes[i]['cost'] * self.x_var[i] for i in self.G.nodes()) <= self.b_a,
            name='attack_budget'
        )
        m.addConstr(self.gamma_var <= self.BIG_M)  # prevent unboundedness on first iteration

        m.update()
        return m

    def init_protector(self):
        m = gp.Model("Protector", env=self.env)

        self.pi_var = m.addVar(vtype=GRB.CONTINUOUS, name='pi')
        for i in self.G.nodes():
            self.w_var[i] = m.addVar(vtype=GRB.BINARY, name=f'w_{i}')
            self.u_var[i] = m.addVar(vtype=GRB.BINARY, name=f'u_{i}')

        m.setObjective(self.pi_var, GRB.MINIMIZE)

        m.addConstr(
            gp.quicksum(self.G.nodes[i]["price"] * self.w_var[i] for i in self.G.nodes()) <= self.b_p,
            name='protector_budget'
        )

        # Constraint (13-14): Protection-Fortification relationship
        for i in self.G.nodes():
            for j in self.protectors_of[i]:
                # if any w_j is fortified then u_i is protected
                m.addConstr(self.u_var[i] >= self.w_var[j], name=f'protection_lower_{i}')
            # if no w_j is fortified then u_i is not protected, where j's are protectors of i
            m.addConstr(
                self.u_var[i] <= gp.quicksum(self.w_var[j] for j in self.protectors_of[i]),
                name=f'protection_upper_{i}'
            )

        m.update()
        return m

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    
    def cut_crit_struct_callback(self, attacker_model:gp.Model, where:int, w_hat:list[int]):
        """
        Callback that adds critical structures (dominating sets) available to the
        defender for a given attack strategy. These value-function constraints are
        introduced into the attacker model as Gurobi lazy cuts.
        
        Notice that the defender problem is parametrized by the strategy and policy. This problem
        has all interdependencies enforced, unlike the BNB method. 
        """
        if where != GRB.Callback.MIPSOL:
            return

        x_sol = attacker_model.cbGetSolution(self.x_var)
        attack = [i for i in self.G.nodes() if x_sol[i] > 0.5]

        # Solve defender response given attack strategy
        ds, ds_weight = self.solve_one_level(w_hat=w_hat, x_hat=attack)

        if ds_weight < attacker_model.cbGet(GRB.Callback.MIPSOL_OBJ):
            attacker_model.cbLazy(
                self.gamma_var <= ds_weight + self.BIG_M*gp.quicksum(self.x_var[i] for i in ds)
            )
            self.num_crit_structures +=1 
            self.policy_local_critical_structures.append((ds_weight, ds))

    # -------------------------------------------------------------------------
    # Column & Row Generation
    # -------------------------------------------------------------------------
    
    def add_new_columns(self, attack):
        """
        Add columns (and corresponding rows) to the protector model for a new critical attack.
        A column for each vertex in the graph, hence this method adds multiple columns per sub-problem solve.
        In addition to this, for the new columns we also add the rows that correspond to the resulting Defender subproblem. 
        """
        k = self.num_crit_att

        for i in self.G.nodes():
            self.y_hat_var[i, k] = self.protector_model.addVar(vtype=GRB.BINARY, name=f"y_hat_{i}_{k}")

        self.dummy_hat_var[k] = self.protector_model.addVar(vtype=GRB.BINARY, name=f"dummy_hat_{k}")

        # Value function lower bound (Constraint 23)
        self.protector_model.addConstr(
            self.pi_var >= gp.quicksum(
                self.G.nodes[i]['weight'] * self.y_hat_var[i, k] for i in self.G.nodes()
            ) + self.BIG_M * self.dummy_hat_var[k]
        )

        # Dominating set feasibility (Constraint 27)
        for i in self.G.nodes():
            neighbors_set = set(self.G.neighbors(i)) | {i}
            self.protector_model.addConstr(
                gp.quicksum(self.y_hat_var[j, k] for j in neighbors_set) + self.dummy_hat_var[k] >= 1
            )

        # Fortified nodes excluded from DS (Constraint 38)
        for i in self.G.nodes():
            self.protector_model.addConstr(self.y_hat_var[i, k] <= 1 - self.w_var[i])

        # Attacked nodes require protection to be in DS (Constraint 39)
        for i in attack:
            self.protector_model.addConstr(self.y_hat_var[i, k] <= self.u_var[i])

    # -------------------------------------------------------------------------
    # Subproblem Solvers
    # -------------------------------------------------------------------------
    
    def solve_one_level(self, w_hat:list[int], x_hat:list[int]):
        """
        Solve the defender subproblem parametrized by protection policy and attack strategy; interdependent.
        """
        self.D_calls += 1
        t_def = perf_counter()

        u_hat = [j for i in w_hat for j in self.protected_by[i]]

        for i in w_hat:
            self.y_var[i].ub = 0
        for i in x_hat:
            self.y_var[i].ub = 0
            if i in u_hat:
                raise ValueError("Protected node is attacked.")

        # Update the model to reflect bound changes
        self.defender_model.update()
        self.defender_model.optimize()

        # Check optimization status
        if self.defender_model.Status == GRB.OPTIMAL:
            ds = [i for i in self.G.nodes() if self.y_var[i].x > 0.5]
            ds_weight = self.defender_model.ObjVal
        elif self.defender_model.Status == GRB.INFEASIBLE:
            ds = []
            ds_weight = float('inf')
        else:
            raise ValueError(f"Unknown Defender Status: {self.defender_model.Status}")

        # Reset bounds
        for i in self.G.nodes():
            self.y_var[i].ub = 1

        self.time_solving_defender += perf_counter() - t_def
        return ds, ds_weight

    def solve_two_level(self, time_remaining:float, w_hat=[]):
        """
        Find the critical attack for a given protection policy (A–D subgame).
        Please note that in this A-D subgame the defender is also restricted by the policy.
        """
        self.A_calls += 1

        u_hat = [j for i in w_hat for j in self.protected_by[i]]
        for i in u_hat:
            self.x_var[i].ub = 0

        # Temporarily relax cuts involving fortified nodes
        for record in self.propagated_structures:
            if set(record.ds) & set(w_hat):
                record.ds_constr.RHS += self.BIG_M
        self.attacker_model.update()

        self.policy_local_critical_structures = []
        self.attacker_model.Params.TimeLimit = time_remaining
        self.attacker_model.optimize(
            lambda model, where: self.cut_crit_struct_callback(model, where, w_hat)
        )

        if self.attacker_model.Status == GRB.TIME_LIMIT:
            CG_status_logger.info("Attacker Model Timed Out!")
            return None, float('inf')
        if self.attacker_model.Status != GRB.OPTIMAL:
            raise ValueError(f"Attacker Model Status is {self.attacker_model.Status}")

        # Extract x_hat (attack strategy)
        x_hat = [i for i in self.G.nodes if self.x_var[i].x > 0.5]
        aObjVal = self.attacker_model.ObjVal

        # Restore relaxed cuts
        for record in self.propagated_structures:
            if set(record.ds) & set(w_hat):
                record.ds_constr.RHS -= self.BIG_M

        # Propagate local critical structures
        for ds_weight, ds in self.policy_local_critical_structures:
            constr = self.attacker_model.addConstr(
                self.gamma_var <= ds_weight + self.BIG_M * gp.quicksum(self.x_var[i] for i in ds)
            )
            # Must store them, so that we can enable/disable them based on relevant protection polcies
            self.propagated_structures.append(
                DefenderCallbackRecord(ds_weight=ds_weight, ds=ds, ds_constr=constr)
            )

        for i in u_hat: #Reset bounds
            self.x_var[i].ub = 1
        self.attacker_model.update()

        return x_hat, aObjVal

    def solve_three_level(self):
        """
        Solve the three-stage P–A–D game via column-and-row generation.
        The interdependencies are incorporated via scenarios generated
        by the attacker–defender (A–D) subroutine. The generated scenarios
        are introduced into the main protector problem through column and row generation.
        """
        CG_status_logger.info(
            f"{'It':>4} | {'P.Cols':^6}  {'P.Rows':^6} | {'P.ObjVal':^8}  {'D.ObjVal':^8} | "
            f"{'D.Calls':^7} | {'LB':^6} {'UB':^6} {'Gap':^12} {'Time(s)':^9}"
        )

        w_hat = []
        iter = 0
        start_time = perf_counter()

        while True:
            t_iter = perf_counter()
            time_remaining = self.time_limit - (t_iter - start_time)
            if time_remaining < 0:
                break

            attack, aObjVal= self.solve_two_level(time_remaining, w_hat)
            if attack is None: # Attacker Timed Out
                break

            self.num_crit_att += 1
            ds, ds_weight = self.solve_one_level(w_hat, attack)
            self.defender_obj_val = ds_weight
            self.time_solving_attacker += perf_counter() - t_iter

            # Store Best Candidate
            if ds_weight < self.best_feasible_solution[0]:
                self.best_feasible_solution = (ds_weight, w_hat, attack, ds)

            self.gap = (self.best_feasible_solution[0] - self.protector_obj_val) / abs(self.protector_obj_val) \
                if self.protector_obj_val != 0 else float('inf')

            elapsed = perf_counter() - start_time
            CG_status_logger.info(
                f"{iter:4d} | {self.protector_model.NumVars:^6g}  {self.protector_model.NumConstrs:^6g} | "
                f"{self.protector_obj_val:>8g}  {self.defender_obj_val:>8g} | {self.D_calls:^7g} | "
                f"{self.protector_obj_val:^6g} {self.best_feasible_solution[0]:^6g} {self.gap:^12.5g} {elapsed:>10.5f}"
            )
            CG_details_logger.info(
                f"Iter -- {iter}:   Time: {elapsed}\n"
                f"Protection Policy: {w_hat}\n"
                f"Protector Obj Val: {self.protector_obj_val}\n"
                f"Attacker Strategy: {attack}\n"
                f"Attacker Obj Val:  {aObjVal}\n"
                f"Defender Structure: {ds}\n"
                f"Defender Obj Val:  {ds_weight}\n"
            )

            # Stopping Condition: 
            # if the attacker-defender loop terminated, and the resulting ds weight is 
            # less (or EQUAL) to what the protector thought then we are optimal; can stop!
            if self.defender_obj_val <= self.protector_obj_val + self.epsilon:
                break

            self.add_new_columns(attack)

            time_remaining = self.time_limit - (perf_counter() - start_time)
            if time_remaining <0:
                break

            self.protector_model.update()
            self.protector_model.Params.TimeLimit = time_remaining
            t_p = perf_counter()
            self.protector_model.optimize()
            self.time_solving_protector += perf_counter() - t_p
            
            if self.protector_model.Status == GRB.OPTIMAL:
                w_hat = [i for i in self.G.nodes() if self.w_var[i].x > 0.5]
                self.protector_obj_val = self.protector_model.ObjVal
            elif self.protector_model.Status == GRB.TIME_LIMIT:
                CG_status_logger.info("Time Limit Reached in Protector Model!")
                break

            iter += 1

        # FINAL SOLUTION
        self.defender_obj_val   = self.best_feasible_solution[0]
        self.protection_policy  = self.best_feasible_solution[1]
        self.attack_strategy    = self.best_feasible_solution[2]
        self.defender_structure = self.best_feasible_solution[3]
        self.time = perf_counter() - start_time
        self.protector_iterations = iter
        self.protector_columns = self.protector_model.NumVars
        self.protector_rows    = self.protector_model.NumConstrs

        return self.best_feasible_solution
    
    # -------------------------------------------------------------------------
    # Reporting
    # -------------------------------------------------------------------------
    
    def get_statistics(self):
        return {
            "SLURM_ARRAY_TASK_ID": os.environ.get('SLURM_ARRAY_TASK_ID', 'local'),
            "Iters":      self.protector_iterations,
            "Time":       round(self.time, 2),
            "P.Time":     round(self.time_solving_protector, 2),
            "P.Cols":     self.protector_columns,
            "P.Rows":     self.protector_rows,
            "A.Calls":    self.A_calls,
            "D.Calls":    self.D_calls,
            "D.Time":     round(self.time_solving_defender, 2),
            "A.Time":     round(self.time_solving_attacker, 2),
            "Gap":        f"{self.gap:.5f}",
            "Weight":     f"{self.defender_obj_val:.5f}",
            "Opt.Policy":    self.protection_policy,
            "Opt.Strategy":  self.attack_strategy,
            "Opt.Structure": self.defender_structure,
        }

