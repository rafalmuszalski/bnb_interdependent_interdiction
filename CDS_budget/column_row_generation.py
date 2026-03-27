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
        self.h_hat_var = {} # DS edge variables
        self.g_hat_var = {} # DS flow variables
        self.dummy_hat_var = {}  # Dummy Var (takes 1 if DS is infeasible)
        self.pi_var = None  # Value Func Var

        # --Attacker
        self.x_var = {}
        self.gamma_var = None

        # --Defender
        self.y_var = {}
        self.g_var = {}
        self.h_var = {}

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
        """Initialize Defender Model"""
        m = gp.Model("Defender", env=self.env)

        # Create variables
        for i in self.G.nodes():
            self.y_var[i] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}")
            self.h_var[self.n,i] = m.addVar(vtype=GRB.BINARY, name=f"h_{self.n}_{i}") #arcs connecting the dummy node with each vertex           
            for j in self.G.neighbors(i):
                self.h_var[i,j] = m.addVar(vtype=GRB.BINARY, name=f"h_{i}_{j}") #arcs between vertices        
                self.g_var[i,j] = m.addVar(vtype=GRB.CONTINUOUS, name=f"g_{i}_{j}") # flow of commodity k on arc (i,j)

            self.g_var[self.n,i] = m.addVar(vtype=GRB.CONTINUOUS, name=f"g_{self.n}_{i}") # flow of commodity k on arc (n,i) from dummy node

        # Objective: Minimize total weight of dominating set
        m.setObjective(gp.quicksum(self.G.nodes[i]['weight'] * self.y_var[i] for i in self.G.nodes()), GRB.MINIMIZE)

        # Constraint (2): Each node must be dominated
        for i in self.G.nodes():
            m.addConstr(self.y_var[i] + gp.quicksum(self.y_var[j] for j in self.G.neighbors(i)) >= 1)


        for i in self.G.nodes():
            m.addConstr(gp.quicksum(self.g_var[j,i] for j in self.G.neighbors(i)) + self.g_var[self.n,i] - gp.quicksum(self.g_var[i,j] for j in self.G.neighbors(i)) == self.y_var[i]) # flow conservation for commodity k at node i
                    
        for i in self.G.nodes():
            for j in self.G.neighbors(i):
                # Constraint 6
                m.addConstr(self.h_var[i,j] + self.h_var[j,i] <= 1) # only one direction (i,j) or (j,i) can be used


        for i in self.G.nodes():
            # Constraint 8
            m.addConstr(self.g_var[self.n,i] <= self.h_var[self.n,i]*self.n) # flow can only be sent on selected arcs
            for j in self.G.neighbors(i):
                # Constraint 7
                m.addConstr(self.g_var[i,j] <= self.h_var[i,j]*self.n) # flow can only be sent on selected arcs

        # Constraint 9
        m.addConstr(gp.quicksum(self.h_var[self.n,i] for i in self.G.nodes()) == 1) # only one arc from dummy node is selected
        
        for i in self.G.nodes():          
            for j in self.G.neighbors(i):
                # C10, C11
                m.addConstr(self.h_var[i,j] <= self.y_var[i]) # arc (i,j) can only be selected if i is in DS
                m.addConstr(self.h_var[i,j] <= self.y_var[j]) # arc (i,j) can only be selected if j is in DS
            # C12
            m.addConstr(self.h_var[self.n,i] <= self.y_var[i]) # arc (n,i) can only be selected if i is in DS

        m.update()
        return m
  

    def init_attacker(self):
        """Initialize Attacker Model"""
        m = gp.Model("Attacker",env=self.env)
        m.setParam("LazyConstraints", 1)

        # Create x variables for each node
        for i in self.G.nodes():
            self.x_var[i] = m.addVar(vtype=GRB.BINARY, name=f'x_{i}')

        self.gamma_var = m.addVar(vtype=GRB.CONTINUOUS, name='gamma')

        # Objective: Maximize gamma
        m.setObjective(self.gamma_var, GRB.MAXIMIZE)

        m.addConstr(gp.quicksum(self.G.nodes[i]['cost']*self.x_var[i] for i in self.G.nodes()) <= self.b_a,name='attack_budget')

        # dummy variable to prevent unboundedness in first iteration
        m.addConstr(self.gamma_var <= self.BIG_M)

        m.update()
        return m

    def init_protector(self):
        """Initialize Protector Model"""
        m = gp.Model("Protector",env=self.env)

        self.pi_var = m.addVar(vtype=GRB.CONTINUOUS, name='pi')
        for i in self.G.nodes():
            self.w_var[i] = m.addVar(vtype=GRB.BINARY, name=f'w_{i}')
            self.u_var[i] = m.addVar(vtype=GRB.BINARY, name=f'u_{i}')

        # Objective: Minimize pi
        m.setObjective(self.pi_var, GRB.MINIMIZE)

        # Constraint (12): Fortification budget constraint
        m.addConstr(gp.quicksum(self.G.nodes[i]["price"]*self.w_var[i] for i in self.G.nodes()) <= self.b_p,name='protector_budget')

        # Constraint (13-14): Protection-Fortification relationship
        for i in self.G.nodes():
            for j in self.protectors_of[i]:
                m.addConstr( # if any w_j is fortified then u_i is protected
                    self.u_var[i] >= self.w_var[j],
                    name=f'protection_lower_{i}'
                )
            m.addConstr( # if no w_j is fortified then u_i is not protected, where j's are protectors of i
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
        For each iteration of critical attack, we neeed to add columns to the main (protector) problem.
        It's actually a column for each vertex in the graph, hence this method adds multiple columns per sub-problem solve.

        In addition to this, for the new columns we also add the rows that correspond to the resulting D subproblem. 
        """
        # Generate new columns (plural) (its one col per each node in the graph)
        for i in self.G.nodes():
            self.y_hat_var[i,self.num_crit_att] = self.protector_model.addVar(vtype=GRB.BINARY, name=f"y_hat_{i}_{self.num_crit_att}")
        
        self.dummy_hat_var[self.num_crit_att] = self.protector_model.addVar(vtype=GRB.BINARY, name=f"dummy_hat_{self.num_crit_att}")

        for i in self.G.nodes():
            self.y_hat_var[i,self.num_crit_att] = self.protector_model.addVar(vtype=GRB.BINARY, name=f"y_hat_{i}_{self.num_crit_att}")
            self.h_hat_var[self.n,i,self.num_crit_att] = self.protector_model.addVar(vtype=GRB.BINARY, name=f"h_hat{self.n}_{i}_{self.num_crit_att}")            
            for j in self.G.neighbors(i):
                self.h_hat_var[i,j,self.num_crit_att] = self.protector_model.addVar(vtype=GRB.BINARY, name=f"h_hat{i}_{j}_{self.num_crit_att}")            
                
                self.g_hat_var[i,j,self.num_crit_att] = self.protector_model.addVar(vtype=GRB.CONTINUOUS, name=f"g_hat{i}_{j}_{self.num_crit_att}") 
            
            self.g_hat_var[self.n,i,self.num_crit_att] = self.protector_model.addVar(vtype=GRB.CONTINUOUS, name=f"g_hat{self.n}_{i}_{self.num_crit_att}")

        # Add Constraint 23
        self.protector_model.addConstr(self.pi_var >= gp.quicksum(self.G.nodes[i]['weight']*self.y_hat_var[i,self.num_crit_att] for i in self.G.nodes())+ self.BIG_M*self.dummy_hat_var[self.num_crit_att])

        # Add Constraint 27
        for i in self.G.nodes():
            neighbors_set = set(self.G.neighbors(i))
            neighbors_set.add(i)
            self.protector_model.addConstr(gp.quicksum(self.y_hat_var[j, self.num_crit_att] for j in neighbors_set) +self.dummy_hat_var[self.num_crit_att] >= 1)

        # Add Constraint 28-30
        for i in self.G.nodes():
            self.protector_model.addConstr(gp.quicksum(self.g_hat_var[j,i,self.num_crit_att] for j in self.G.neighbors(i)) + self.g_hat_var[self.n,i,self.num_crit_att] - gp.quicksum(self.g_hat_var[i,j,self.num_crit_att] for j in self.G.neighbors(i)) == self.y_hat_var[i,self.num_crit_att]) # flow conservation for commodity k at node i
            
            # Add Constraint 31
        for i in self.G.nodes():
            for j in self.G.neighbors(i):
                self.protector_model.addConstr(self.h_hat_var[i,j,self.num_crit_att] + self.h_hat_var[j,i,self.num_crit_att] <= 1)

            # Add Constraint 32, 33

        for i in self.G.nodes():
            self.protector_model.addConstr(self.g_hat_var[self.n,i,self.num_crit_att] <= self.h_hat_var[self.n,i,self.num_crit_att]*self.n)
            for j in self.G.neighbors(i):
                self.protector_model.addConstr(self.g_hat_var[i,j,self.num_crit_att] <= self.h_hat_var[i,j,self.num_crit_att]*self.n)
    
            # Add Constraint 34-37
        self.protector_model.addConstr(gp.quicksum(self.h_hat_var[self.n,i,self.num_crit_att] for i in self.G.nodes()) == 1)
        for i in self.G.nodes():          
            for j in self.G.neighbors(i):
                self.protector_model.addConstr(self.h_hat_var[i,j,self.num_crit_att] <= self.y_hat_var[i,self.num_crit_att])
                self.protector_model.addConstr(self.h_hat_var[i,j,self.num_crit_att] <= self.y_hat_var[j,self.num_crit_att])
            self.protector_model.addConstr(self.h_hat_var[self.n,i,self.num_crit_att] <= self.y_hat_var[i,self.num_crit_att])

            # Add Constraint 38
        for i in self.G.nodes():
            self.protector_model.addConstr(self.y_hat_var[i,self.num_crit_att] <= 1 - self.w_var[i])

            # Add Constraint 39
        for i in self.G.nodes():
            if i in attack:
                self.protector_model.addConstr(self.y_hat_var[i,self.num_crit_att] <= self.u_var[i])


    # -------------------------------------------------------------------------
    # Subproblem Solvers
    # -------------------------------------------------------------------------
    
    def solve_one_level(self, w_hat:list[int], x_hat:list[int]):
        """
        We solve the defender subproblem parametrized by the policy and strategy, ie interdependent.
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
        Finds critical attacks for a given protection policy. This is the A-D subgame. 
        Please note that in this A-D subgame the defender is also restricted by the policy.
        """
        self.A_calls += 1

        u_hat = [j for i in w_hat for j in self.protected_by[i]]
        for i in u_hat:
            self.x_var[i].ub = 0

        # Disable (relax) relevant propagated structures
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

        # Enable (restrict) relevant propagated structures
        for record in self.propagated_structures:
            if set(record.ds) & set(w_hat):
                record.ds_constr.RHS -= self.BIG_M

        # Propagate Temporary Critical Structures to Propagated cuts
        for ds_weight, ds in self.policy_local_critical_structures:
            constr = self.attacker_model.addConstr(
                self.gamma_var <= ds_weight + self.BIG_M*gp.quicksum(self.x_var[i] for i in ds)
            )

            # Must store them, so that we can enable/disable them based on relevant protection polcies
            self.propagated_structures.append(
                DefenderCallbackRecord(
                    ds_weight= ds_weight,
                    ds = ds,
                    ds_constr = constr
                )
            )

        for i in u_hat: #Reset bounds
            self.x_var[i].ub = 1
        self.attacker_model.update()

        return x_hat, aObjVal

    def solve_three_level(self):
        """
        Solve the three-stage Protector–Attacker–Defender (P–A–D) game,
        where interdependencies are incorporated via scenarios generated
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
            if time_remaining <0:
                break

            attack, aObjVal = self.solve_two_level(time_remaining, w_hat)
            if attack is None: #Means Attacker Timed Out
                break

            self.num_crit_att += 1
            ds, ds_weight = self.solve_one_level(w_hat,attack)
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
            # less (or EQUAL) to what hte protector was solving then we are good!
            if self.defender_obj_val <= self.protector_obj_val + 1e-5:
                break

            self.add_new_columns(attack)

            time_remaining = self.time_limit - (perf_counter() - start_time)
            if time_remaining < 0:
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

