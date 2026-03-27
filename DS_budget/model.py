# -*- coding: utf-8 -*-
import os
import random
from time import perf_counter
from dataclasses import dataclass
from typing import List

import gurobipy as gp
from gurobipy import GRB
import networkx as nx

from .logging_setup import BNB_status_logger


@dataclass(frozen=True)
class DefenderCallbackRecord:
    ds_weight: float
    ds: List[int]
    ds_constr: gp.Constr

@dataclass(frozen=False)
class AttackerCallbackRecord:
    attack_weight: float
    attack: List[int]
    attack_constr: gp.Constr

class EarlyProtectorTermination(Exception):
    """
    When the protector model lower bound exceeds the BnB-interdependent solution we can terminate the node solve, saving quite a bit of time.
    We don't have to do A-D subgames to further refine the policy, since we already know we wont find a better solution in this node. 
    """
    def __init__(self, lb:float, message="Protector Terminating Early, P.LB >= BnB.UB"):
        self.lb = lb
        super().__init__(f"{message} (P.LB={lb})")

class TimeOutTermination(Exception):
    """
    When we timeout in the protector or attacker solves, we raise this exception, to stop the BnB. 
    """
    def __init__(self, message="BnB TimeOut"):
        super().__init__(message)


class ThreeLevelGame:
    def __init__(self, num_nodes: int, b_p: int, b_a: int,
                 strategy_propagation: int, structure_propagation: int,
                 G: nx.Graph, protected_by: list[list[int]],
                 protectors_of: list[list[int]], seed: int):
        self.num_nodes = num_nodes
        self.seed = seed
        self.b_p = b_p
        self.b_a = b_a
        self.structure_propagation = structure_propagation
        self.strategy_propagation = strategy_propagation
        self.G = G
        self.protected_by = protected_by
        self.protectors_of =  protectors_of

        self.M = 1e5
        self.epsilon = 1e-5
        random.seed(seed)

        self.UB = self.M

        # Statistics
        self.D_calls = 0
        self.A_calls = 0
        self.num_critical_structures_added = 0
        self.num_critical_strategies_added = 0
        self.time_solving_protector          = 0
        self.time_solving_attacker           = 0
        self.time_solving_defender           = 0
        self.time_solving_attacker_recourse  = 0
        self.time_solving_defender_recourse  = 0

        # Propagated cuts
        self.propagated_structures: list[DefenderCallbackRecord] = []
        self._seen_structures = set()
        self.propagated_strategies: list[AttackerCallbackRecord] = []
        self._seen_strategies = set()
        self.root_heuristic = None

        # Variables
        self.w_vars = None
        self.x_vars = None
        self.y_vars = {}

        
        # Gurobi environment
        self.env = gp.Env(empty=True)
        self.env.setParam("LogToConsole", 0)
        self.env.setParam("Seed", seed)
        self.env.setParam("Threads", 8)
        self.env.setParam("SoftMemLimit", 16)
        self.env.start()

        self.defender_model  = self.init_defender_model()
        self.attacker_model  = self.init_attacker_model()
        self.protector_model = self.init_protector_model()


    # -------------------------------------------------------------------------
    # Model Initialization
    # -------------------------------------------------------------------------

    def init_defender_model(self):
        m = gp.Model("Defender", env=self.env)

        self.y_vars = [
            m.addVar(vtype=GRB.BINARY, obj=self.G.nodes[i]['weight'], name=f"y_{i}")
            for i in self.G.nodes()
        ]

        for i in self.G.nodes():
            m.addConstr(
                self.y_vars[i] + gp.quicksum(self.y_vars[j] for j in self.G.neighbors(i)) >= 1,
                name=f"ds_{i}"
            )

        m.update()
        return m
    
    def init_attacker_model(self):
        m = gp.Model("Attacker", env=self.env)
        m.Params.PreCrush = 0
        m.Params.Presolve = 0
        m.Params.Heuristics = 0
        m.Params.LazyConstraints = 1

        self.x_vars = [m.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in self.G.nodes()]
        self.alpha_vars = m.addVar(vtype=GRB.CONTINUOUS, obj=1, name="alpha")

        m.addConstr(
            gp.quicksum(self.G.nodes[i]['cost'] * self.x_vars[i] for i in self.G.nodes()) <= self.b_a,
            name="budget"
        )
        m.addConstr(self.alpha_vars <= self.M +1)  # prevent unboundedness
        m.ModelSense = GRB.MAXIMIZE
        m.update()
        return m
    
    def init_protector_model(self):
        m = gp.Model("Protector",env=self.env)
        m.Params.PreCrush = 0
        m.Params.Presolve = 0
        m.Params.Heuristics = 0
        m.Params.LazyConstraints = 1

        self.w_vars = [m.addVar(vtype=GRB.BINARY, name=f"w_{i}") for i in self.G.nodes()]
        self.pi_vars = m.addVar(vtype=GRB.CONTINUOUS, obj=1, name="pi")

        m.addConstr(
            gp.quicksum(self.G.nodes[i]['price'] * self.w_vars[i] for i in self.G.nodes()) <= self.b_p,
            name="budget"
        )

        defender_structure, defender_weight, d_time = self.solve_defender(attacker_strategy=[])
        self.time_solving_defender += d_time
        m.addConstr(self.pi_vars >= defender_weight)
        
        m.update()

        return m 

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def cut_crit_struct_callback(self, attacker_model:gp.Model, where:int, timer:bool):
        """
        Lazy cut callback: adds defender dominating sets as value-function constraints
        into the attacker model until it converges to the strategy that maximizes
        the minimum DS weight.
        """
        if where != GRB.Callback.MIPSOL:
            return
        
        self.D_calls += 1 
        x_sol = attacker_model.cbGetSolution(self.x_vars)
        attack = [i for i in self.G.nodes() if x_sol[i] > 0.5]

        ds, ds_weight, d_time = self.solve_defender(attacker_strategy=attack)
        if timer: self.time_solving_defender +=  d_time

        # For the given attack, the resulting dominating set is better than attacker's choice
        # Therefore this DS needs to be added to the attacker sub-problem as a cut
        if ds_weight < self.attacker_model.cbGet(GRB.Callback.MIPSOL_OBJ):
            self.num_critical_structures_added += 1 
            attacker_model.cbLazy(
                self.alpha_vars <= ds_weight + self.M*gp.quicksum(self.x_vars[i] for i in ds)
            )

            # Store DS for propagation as Model Constraints marked Lazy, 
            # not in other nodes but even within the same node, for different policies w_bar
            candidate = tuple(ds)
            if candidate not in self._seen_structures:
                self.policy_local_critical_structures.append((ds_weight, ds))
                self._seen_structures.add(candidate)

    def cut_crit_attack_callback(self,protector_model:gp.Model, where:int, timer:bool):
        """
        Lazy cut callback: adds critical attack strategies into the protector model
        when a strategy results in a ds weight exceeding the value currently impleid by the protector.

        Such strategies indicate that the protector’s policy is insufficient. The
        protector must therefore account for (i.e., cover) these most critical attacks
        to ensure the defender can achieve the lowest possible dominating set weight.
        """

        if where == GRB .Callback.MIP:
            best_bound = protector_model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            if best_bound >= self.UB:
                self.protector_model.terminate()

        if where == GRB.Callback.MIPSOL:
            self.A_calls += 1 
            w_sol = protector_model.cbGetSolution(self.w_vars)
            protection = [i for i in self.G.nodes() if w_sol[i] > 0.5]

            attack, attacker_obj_val, a_time = self.solve_attacker(protector_policy=protection, timer=timer)
            if timer: self.time_solving_attacker += a_time

            # For the given protection, the resulting attack causes the dominating set weight
            # to be worse (higher) than what the protector thought. This is a critical attack
            # Therefore this attack needs to be added to the protector to protect against. 
            if attacker_obj_val > self.protector_model.cbGet(GRB.Callback.MIPSOL_OBJ):
                self.num_critical_strategies_added += 1 

                # Nodes that must be covered to block the critical attack
                must_cover_attack = set(attack)
                for i in attack:
                    must_cover_attack |= set(self.protectors_of[i])

                #NOTE self.M can be replaced with attacker_obj_val
                self.protector_model.cbLazy(
                    self.pi_vars >= attacker_obj_val - self.M*gp.quicksum(self.w_vars[i] for i in sorted(must_cover_attack))
                )

                # Store Critical Strategies for Propagation Later
                candidate = tuple(attack)
                if candidate not in self._seen_strategies:
                    self.local_critical_strategies.append((attacker_obj_val, attack))
                    self._seen_strategies.add(candidate)
    
    # -------------------------------------------------------------------------
    # Cut Propagation
    # -------------------------------------------------------------------------

    def add_global_strategy_cuts(self):
        """
        Propagate locally found critical attacks as global protector constraints for future BnB nodes.

        This helps cut down on computation time, because rather than (in each node) doing the full
        P-A-D enumeration, the protector already comes with many critical strategies. This allows us to 
        hopefully skip a lot of (A-D) subgames to find critical attacks. 
        """
        for weight, attack in self.local_critical_strategies:
            must_cover = set(attack).union(*(self.protectors_of[i] for i in attack))
            rhs = weight - self.M * gp.quicksum(self.w_vars[i] for i in must_cover)

            # For C.Strat popagation, we always mark constraints as constr.Lazy = 0
            constr = self.protector_model.addConstr(self.pi_vars >= rhs)

            # Store them so that they can be enabled/disabled based on future nodes
            self.propagated_strategies.append(
                AttackerCallbackRecord(
                    attack_weight=weight,
                    attack=attack,
                    attack_constr=constr,
                )
            )
        self.protector_model.update()

    def add_global_structure_cuts(self):
        """
        Propagate locally found defender structures as global attacker constraints for future BnB nodes and P-games.
        
        For each A-D sub-game, we are repeatedly generating defender structures. The issue with that
        is that a lot of them have already been discovered. We are finding structures we identified before. So
        to cut down on this wasted effort, we can store all defender structures found and add them to the Attacker model
        every time it is re-run. 
        """
        for ds_weight, ds in self.policy_local_critical_structures:
            constr = self.attacker_model.addConstr(
                self.alpha_vars <= ds_weight + self.M*gp.quicksum(self.x_vars[i] for i in ds)
                )
            constr.Lazy = self.structure_propagation # -1=off, 0=regular, 1/2=lazy
            # must store them, so that we can enable/disable them based on relevant branching conditions
            self.propagated_structures.append(
                DefenderCallbackRecord(
                    ds_weight = ds_weight,
                    ds = ds,
                    ds_constr = constr
                )
            )
        self.attacker_model.update()

    # -------------------------------------------------------------------------
    # Subproblem Solvers
    # -------------------------------------------------------------------------

    def solve_protector(self, timelimit:int):
        p_time = perf_counter()
        self.protector_model.Params.TimeLimit = timelimit

        # Array for storing temporary attacks found 
        self.local_critical_strategies = []

        self.protector_model.optimize(lambda model, where: self.cut_crit_attack_callback(model, where, True))

        if self.protector_model.Status == GRB.OPTIMAL:
            protection = [i for i in self.G.nodes() if self.w_vars[i].x > 0.5]
            prot_obj_val = self.protector_model.ObjVal
        elif self.protector_model.Status == GRB.INFEASIBLE:
            protection = []
            prot_obj_val = self.M
        elif self.protector_model.Status == GRB.TIME_LIMIT:
            # Cannot extract any solution, since it will be suboptimal and so it cannot be used by following players
            raise TimeOutTermination("Protector Model TimeLimit Reached.")
        elif self.protector_model.Status == GRB.INTERRUPTED:
            raise EarlyProtectorTermination(self.protector_model.ObjBound)
        else:
            raise ValueError(f"Unexpected Protector Model Status: {self.protector_model.Status}")
        
        # Propagate temporary critical strategies to global cuts
        if self.strategy_propagation >= 0:
            self.add_global_strategy_cuts()

        p_time = perf_counter() - p_time
        return protection, prot_obj_val, p_time

    def solve_attacker(self, protector_policy=[], timer=False):
        """Solve the A–D subgame with the defender independent of the protection policy."""
        a_time = perf_counter()

        # Reset all attacker var bounds, since those are not managed by the BNB 
        for i in self.G.nodes():
            self.x_vars[i].ub = 1 

        # Apply protector_policy to attacker 
        for i in protector_policy:
            for j in self.protected_by[i]:
                self.x_vars[j].ub = 0 
        self.attacker_model.update()

        # Apply protector_policy to defender
        # No! We are solving independent game.
        # Only bnb enforces interdependencies

        # Array for storing temporary ds found for a given policy (w_bar) 
        self.policy_local_critical_structures = []

        self.attacker_model.optimize(
            lambda model, where: self.cut_crit_struct_callback(model, where, timer)
        )

        if self.attacker_model.Status != GRB.OPTIMAL:
            raise ValueError(f"Unexpected Attacker Model Status: {self.attacker_model.Status}")
        
        attack = [i for i in self.G.nodes() if self.x_vars[i].x > 0.5]
        attacker_obj_val = self.attacker_model.ObjVal

        ds, ds_weight, d_time = self.solve_defender(attacker_strategy=attack)

        spread = abs(ds_weight - attacker_obj_val)
        if spread > self.epsilon:
            raise ValueError(
                f"A-D subgame verification failed. "
                f"a={attacker_obj_val:.9}, d={ds_weight:.9} "
                f"diff={abs(ds_weight - attacker_obj_val):.9}"
            )

        # We ran a free heuristic at root node, so lets store it and use that.
        if protector_policy == [] and self.root_heuristic is None: 
            self.root_heuristic = ("RootHeuristic", protector_policy, attack, ds, ds_weight, perf_counter()-a_time)

        # propagate the temporary critical structures to propagated, global(all nodes) cuts
        if self.structure_propagation >= 0: #-1=no propagation, 0=regular constr, 1=lazy, 2=lazy. 
            self.add_global_structure_cuts()

        # Reset attacker variable bounds. Not managed by the BNB
        for i in self.G.nodes():
            self.x_vars[i].ub = 1 
        self.attacker_model.update()

        a_time = perf_counter() - a_time
        return attack, attacker_obj_val, a_time

    def solve_defender(self, attacker_strategy=[]):
        """Solve the independent defender relaxation (not constrained by the protector)."""
        d_time = perf_counter()

        # Reset defender variable bounds
        # NOTE: No! These bounds are managed by the bnb branching decisions.
        # You cannot reset them here. 

        # Apply attacker_strategy on defender
        # Store original bounds so BnB branching decisions are preserved after the solve, trickery due to prop_strats
        original_bound = {} 
        for i in attacker_strategy:
            original_bound[i] = self.y_vars[i].ub
            self.y_vars[i].ub = 0 
        self.defender_model.update()

        self.defender_model.optimize()

        if self.defender_model.Status == GRB.OPTIMAL:
            defender_structure = [i for i in self.G.nodes() if self.y_vars[i].x >0.5]
            defender_weight = sum(self.G.nodes[i]['weight'] for i in defender_structure)
        elif self.defender_model.Status == GRB.INFEASIBLE:
            defender_structure = []
            defender_weight = self.M 
        else:
            raise ValueError(f"Unexpected Defender Model Status: {self.defender_model.Status}")
        
        # Reset attacker_strategy on defender
        for i in attacker_strategy:
            self.y_vars[i].ub = original_bound[i] 
        self.defender_model.update()

        d_time = perf_counter() - d_time
        return defender_structure, defender_weight, d_time
    
    def solve_recourse(self, protector_policy=[], attacker_strategy=[]):
        """
        Solve the defender's interdependent response given fixed policy and attack.
        WARNING: destroys the defender model after use — call only for debugging/final verification.

        NOTE: `solve_defender` is NOT analogous, as it ignores the P-D interaction. 
        That coupling is handled explicitly within the BnB branching scheme.
        """
        for i in self.G.nodes():
            self.y_vars[i].ub = 1 
        for i in protector_policy:
            self.y_vars[i].ub = 0 
        for i in attacker_strategy:
            self.y_vars[i].ub = 0

        # Verify the Attacker isn't targeting some protected vertex
        attacked_set = set(attacker_strategy)
        for i in protector_policy:
            overlap = attacked_set & set(self.protected_by[i])
            if overlap:
                raise ValueError(f"Attack targeted a protected node: {overlap}")
            
        self.defender_model.optimize()

        if self.defender_model.Status == GRB.OPTIMAL:
            ds = [i for i in self.G.nodes() if self.y_vars[i].x > 0.5]
            obj_val = sum(self.G.nodes[i]['weight'] for i in self.G.nodes() if self.y_vars[i].x > 0.5)
        elif self.defender_model.Status == GRB.INFEASIBLE:
            ds = []
            obj_val = self.M
        else:
            raise ValueError(f"Unexpected Recourse Defender Status: {self.defender_model.Status}")
        
        del self.defender_model
        del self.y_vars
        BNB_status_logger.warning("DEFENDER MODEL DESTROYED FOR YOUR OWN SAFETY. USE THIS FUNC CAREFULLY")

        return ds, obj_val
            
    # -------------------------------------------------------------------------
    # Main Solve
    # -------------------------------------------------------------------------

    def solve_three_level_game(self, timelimit:int):
        """
        Solve the relaxed P–A–D game. Interdependencies are enforced by the BranchAndBound class.
        After we solve the protector problem, we need to recalculate recourses to get the full triplet of solutions
        """

        protection, p_obj, p_time = self.solve_protector(timelimit)
        attack,     a_obj, a_time = self.solve_attacker(protection, timer=False)
        structure,  d_obj, d_time = self.solve_defender(attack)

        self.time_solving_protector         += p_time
        self.time_solving_attacker_recourse += a_time
        self.time_solving_defender_recourse += d_time

  
        spread = max(p_obj, a_obj, d_obj) - min(p_obj, a_obj, d_obj)
        if spread > self.epsilon:
            raise ValueError(
                f"Objective mismatch: p={p_obj:.9}, a={a_obj:.9}, d={d_obj:.9}, "
                f"spread={spread:.9}, eps={self.epsilon:.9}"
            )


        return (protection, attack, structure, d_obj)
