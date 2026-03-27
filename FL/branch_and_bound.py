import heapq
import os
from time import perf_counter
from dataclasses import dataclass, field
from typing import Any, Optional, List

from .logging_setup import BNB_status_logger, BNB_details_logger
from .model import ThreeLevelGame


@dataclass
class Node:
    """ Node within the BNB Tree. """
    name: int
    parent: Optional[int]
    status: str
    objective_value: float
    depth: int
    decision_var: int
    decision_var_fix: int
    left_child: Optional[int] = None
    right_child: Optional[int] = None
    interdependency: list = None
    fortification_crews: List[int] = field(default_factory=list) # P.Branch Decision made upto this node
    policy:list = None
    strategy: list = None
    allocation: list = None
    location: list = None

    def __lt__(self, other: "Node") -> bool:
        """ Sorting Nodes in descending objective value. """
        if abs(self.objective_value - other.objective_value) < 1e-9:
            return self.decision_var_fix < other.decision_var_fix
        return self.objective_value > other.objective_value



class BranchAndBound:
    def __init__(self, model:ThreeLevelGame, timelimit:int):
        self.model = model
        self.time_limit = timelimit
        self.max_num_nodes = 1e6
        self.epsilon = 1e-3


        self.delayed_log_of_heuristic_solution = None
        self.time_solving_heuristic = 0
        self.H_calls = 0
        self.time_propagating_strategies = 0
        self.propagating_strategies_calls = 0

        self.num_nodes = 1 
        self.node_queue = []
        self.node_storage = []
        self.tree_depth = 0 


        # Information on bounds and current incumbent
        self.best_feasible_solution = (float('-inf'),[],[],[],[])
        self.lower_bound_node_name = -1
        self.lower_bound = float('-inf')
        self.upper_bound = float('inf')
        self.gap = float('nan')

        # Generate root node and populate into the PQ
        root_node = Node(
            name=0,
            parent=None,
            objective_value=float('-inf'),
            decision_var=None,
            decision_var_fix=None,
            depth=0,
            status='root'
        )
        heapq.heappush(self.node_queue, root_node)
        self.node_storage.append(root_node) #Used for accessing information about nodes

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    
    def print_status(self, node:Node):
        if node.name == 0:
            header = (
                f"{'# Nodes':^10}{'Open':^10}{'T.Depth':^10}| {'Node':^10}{'Status':^10}"
                f"{'P.Obj':^10}{'Var':^10}{'Var_Fix':^12}{'N.Depth':^10}{'N.Parent':^10}|"
                f"{'LB':^10}{'UB':^10}{'Gap':^12}{'C.Strats':^10}{'C.Structs':^10}{'Time':^10}"
            )
            BNB_status_logger.info(header)

        status_text = (
            f"{self.num_nodes:^10}{len(self.node_queue):^10}{self.tree_depth:^10}|"
            f" {node.name:^10}{node.status:^10}{node.objective_value:^10.6g}"
            f"{str(node.decision_var) if node.decision_var is not None else 'None':^10}"
            f"{node.decision_var_fix if node.decision_var is not None else 'None':^12}"
            f"{node.depth:^10}{str(node.parent) if node.parent is not None else 'None':^10}|"
            f"{self.lower_bound:^10.6g}{self.upper_bound:^10.6g}{self.gap:^12.5g}"
            f"{self.model.num_critical_strategies_added:^10}{self.model.num_critical_structures_added:^10}"
            f"{perf_counter()-self.start_time:>10.5f}"
        )
        BNB_status_logger.info(status_text)

        node_details = (
            f"Node Solution -- Node Name: {node.name}\n"
            f"Protection Policy w_(a,k): {node.policy}\n"
            f"Attack Strategy x_(i,j): {node.strategy}\n"
            f"Defender Allocation: y_(i,j) {node.allocation}\n"
            f"Defender Location z_(i,k): {node.location}\n"
            f"Interdependency: {node.interdependency}\n"
            f"Crews Patroling: {[k for (a,k) in node.policy]}\n"
            f"Crews Stationed: {[k for (i,k) in node.location]} \n"
            f"Node Value: {node.objective_value}, Node Status: {node.status}\n"
            f"Decision Var: {node.decision_var}, Fix: {node.decision_var_fix},"
            f" Parent: {node.parent}\n"
        )
        BNB_details_logger.info(node_details)

    def print_status_heuristic(self):
        """ Logging heuristic solution into the log file and console. """
        (hname, hpolicy, hattack, hdefender_allocation, hdefender_location, hdefender_weight, h_time) = self.delayed_log_of_heuristic_solution
        if hdefender_weight > self.lower_bound + self.epsilon:
            self.lower_bound = hdefender_weight
            self.lower_bound_node_name = "Heuristic"
            self.best_feasible_solution = (hdefender_weight, hpolicy, hattack, hdefender_allocation, hdefender_location)

        if self.upper_bound > self.epsilon: #Avoid div0
            self.gap = (self.upper_bound - self.lower_bound) / self.upper_bound

        time_fromatted = f"({h_time:.5f})"
        status_text = (
            f"{'  HEURISTIC SOLUTION ':^30}|"
            f" {'*':^10}{'RH' if hname =='RootHeuristic' else 'H':^10}{hdefender_weight:^10.6g}"
            f"{'':^10}"
            f"{'':^12}"
            f"{'':^10}"
            f"{hname if hname !='RootHeuristic' else 'RH':^10}|"
            f"{self.lower_bound:^10.6g}{self.upper_bound:^10.6g}{self.gap:^12.5g}"
            f"{'':^10}{'':^10}"
            f"{time_fromatted:>11}"
        )
        node_details = (
            f" -- HEURISTIC SOLUTION --\n"
            f"Protection Policy: {hpolicy}\n"
            f"Protector Price: NOT-TRACKED \n" #{self.model.protector_obj_val}
            f"Attack Strategy: {hattack}\n"
            f"Attacker Cost: NOT-TRACKED \n" #{self.model.attacker_obj_val}
            f"Defender Allocation: {hdefender_allocation}\n"
            f"Defender Location: {hdefender_location}\n"
            f"Defender Weight {hdefender_weight}\n"
            f"Run at Node: {hname}\n"
        )
        BNB_status_logger.info(status_text)
        BNB_details_logger.info(node_details)

    # -------------------------------------------------------------------------
    # Bounds & Statistics
    # -------------------------------------------------------------------------

    def update_lb_ub_gap(self):
        if self.node_queue:  # Update the upper bound from the front of PQ (if any)
            # UB is the best upper bound among unexplored nodes: Ensure UB >= LB
            self.upper_bound = max(self.node_queue[0].objective_value, self.lower_bound)
        else:
            self.upper_bound = self.lower_bound

        if self.upper_bound > self.epsilon:
            self.gap = (self.upper_bound - self.lower_bound) / self.upper_bound
        else:
            self.gap = float('inf')

    def get_time_remaining(self):
        return self.time_limit - (perf_counter() - self.start_time)
    
    def get_statistics(self):
        return {
            "SLURM_ARRAY_JOB_ID": os.environ.get('SLURM_ARRAY_JOB_ID','local'),
            "SLURM_ARRAY_TASK_ID": os.environ.get('SLURM_ARRAY_TASK_ID', 'local'),
            "Time": round(self.end_time - self.start_time,2),
            "#Nodes": self.num_nodes,
            "C.Strats": self.model.num_critical_strategies_added,
            "C.Structs": self.model.num_critical_structures_added,
            "P.Time": round(self.model.time_solving_protector,5),
            "A.Time": round(self.model.time_solving_attacker,5),
            "D.Time": round(self.model.time_solving_defender,5),
            "AR.Time": round(self.model.time_solving_attacker_recourse,5), 
            "DR.Time": round(self.model.time_solving_defender_recourse,5),
            "H.Time": round(self.time_solving_heuristic,5),
            "C.Strat.Propagation.Time": round(self.time_propagating_strategies,5),
            "C.Strat.Propagation.Calls": self.propagating_strategies_calls,
            "UB": f"{self.upper_bound:.5f}",
            "LB": f"{self.lower_bound:.5f}",
            "Opt.Gap": f"{self.gap}",
            "D.calls": self.model.D_calls,
            "A.calls": self.model.A_calls,
            "H.calls": self.H_calls,
            "Opt.Policy": self.best_feasible_solution[1],
            "Opt.Strategy": self.best_feasible_solution[2],
            "Opt.Allocation": self.best_feasible_solution[3],
            "Opt.Location": self.best_feasible_solution[4],
        }

    def save_instance_info(self):
        BNB_details_logger.info(
            f" ----- Instance Info Snapshot ----- \n"
            f"Optimal Node: {self.lower_bound_node_name}\n"
            f"Optimal Node Obj Val: {self.best_feasible_solution[0]} \n"
            f"Optimal Policy: {self.best_feasible_solution[1]} \n"
            f"Optimal Strategy: {self.best_feasible_solution[2]} \n"
            f"Optimal Allocation: {self.best_feasible_solution[3]} \n"
            f"Optimal Location: {self.best_feasible_solution[4]} \n"
        )

    # -------------------------------------------------------------------------
    # Model State Management
    # -------------------------------------------------------------------------

    def reset_variable_bound_attributes(self):
        # model.reset(1) does NOT restore variable bounds (e.g. x.ub = 0 persists),
        # so we reset them explicitly here.

        # Protector 
        for a in self.model.regions:
            for k in self.model.crews:
                self.model.w_vars[a,k].lb = 0
                self.model.w_vars[a,k].ub = 1 
        for k in self.model.crews:
            self.model.omega_vars[k].lb = 0
            self.model.omega_vars[k].ub = 1 

        # Attacker 
        for (i,j) in self.model.edges:
            self.model.x_vars[i,j].lb = 0
            self.model.x_vars[i,j].ub = 1

        # Defender
        for (i,j) in self.model.edges:
            self.model.y_vars[i,j].lb = 0
            self.model.y_vars[i,j].ub = 1
        for i in self.model.facilities:
            for k in self.model.crews:
                self.model.z_vars[i,k].lb = 0
                self.model.z_vars[i,k].ub = 1 
        for k in self.model.crews:
            self.model.zeta_vars[k].lb = 0
            self.model.zeta_vars[k].ub = 1 

        self.model.protector_model.update()
        self.model.attacker_model.update()
        self.model.defender_model.update()

    def apply_branching_constraints(self, node:Node):
        """Walk up the tree from node to root, applying each branching decision to the model."""
        node.fortification_crews = []
        current = node

        while current.name != 0: 
            a, k = current.decision_var

            if current.decision_var_fix == 1: # P.Branch
                node.fortification_crews.append(k)
                self.model.w_vars[a,k].lb = 1 # force fortification assignment
                self.model.zeta_vars[k].ub = 0 #crew cannot be used in defender
                self.model.omega_vars[k].lb = 1 #crew must be used in protector
                for i in self.model.facilities:
                    self.model.z_vars[i,k].ub = 0 # Fix and Force Crew Assignment

            elif current.decision_var_fix == 0: # D.Branch
                self.model.w_vars[a, k].ub = 0 # forbid fotification
                # self.model.omega_vars[k].ub = 0 #crew cannot be used in protector

            current = self.node_storage[current.parent]

        # Apply the bound changes to both models
        self.model.protector_model.update()
        self.model.defender_model.update()

    def set_propagated_critical_structures(self, node:Node, action:str):
        assert action in ("relax", "restrict"), f"Invalid action: {action}"

        delta = -self.model.M if action == "relax" else self.model.M
        for record in self.model.propagated_structures:
            record.alloc_constr.Lazy = self.model.structure_propagation
            record_crews_used_for_distribution = [k for (i,k) in record.facility_crew_placement]
            if set(record_crews_used_for_distribution) & set(node.fortification_crews):
                record.alloc_constr.RHS += delta
        self.model.attacker_model.update()

    def set_propagated_critical_strategies(self, node:Node):
        t = perf_counter()

        for record in self.model.propagated_strategies:
            _, _, new_weight, _ = self.model.solve_defender(attacker_strategy=record.attack)
            record.attack_weight = new_weight
            record.attack_constr.RHS = new_weight
            self.propagating_strategies_calls += 1 

        self.model.protector_model.update()
        self.time_propagating_strategies += perf_counter() - t

    # -------------------------------------------------------------------------
    # Feasibility Checks
    # -------------------------------------------------------------------------

    def is_interdependent(self, node: Node):
        crews_placed_in_facilities = [k for (i,k) in node.location]

        # Inline filtering — find interdependent (region, crew) pairs directly
        interdependency = [
            (region, crew)
            for region, crew in node.policy
            if crew in crews_placed_in_facilities
        ]
        node.interdependency = sorted(interdependency, key=lambda x: x[1])

        return len(node.interdependency) > 0
    
    def is_attacker_subopt(self, node:Node):
        attacker_obj_val, t_heuristic = self.run_heuristic(policy=node.policy, ran_at_bnb_node=node.name)
        self.time_solving_heuristic += t_heuristic

        if attacker_obj_val < node.objective_value - self.epsilon:
            node.interdependency = sorted(set(node.policy) - set(node.fortification_crews))
            assert node.interdependency, f"interdependency Set is empty SOPT in node {node.name} -- {attacker_obj_val} -- {node.objective_value}"
            return True
        
        return False 
    
    def run_heuristic(self, policy=[], ran_at_bnb_node:str="*"):
        self.H_calls += 1 
        t = perf_counter()

        # We need to solve the A-D where the defender is parametrized by the protector
        # We apply this parametrization OUTSIDE of the function, because solve attacker is used by independent game too
        for (a,k) in policy:
            self.model.zeta_vars[k].ub = 0
            for i in self.model.facilities:
                self.model.z_vars[i,k].ub = 0 
        self.model.defender_model.update()

        # Relax Propagated Structures that are invalid because of protection policy
        for record in self.model.propagated_structures:
            record_crews_used_for_distribution = [k for (i,k) in record.facility_crew_placement]
            crews_used_for_patrol = [k for (a,k) in policy]
            if set(record_crews_used_for_distribution) & set(crews_used_for_patrol):
                record.alloc_constr.RHS -= self.model.M
        self.model.attacker_model.update()

        attack, attacker_obj_val, a_time = self.model.solve_attacker(policy,timer=False)

        # Undo that relaxation not to mess things up later 
        for record in self.model.propagated_structures:
            record_crews_used_for_distribution = [k for (i,k) in record.facility_crew_placement]
            crews_used_for_patrol = [k for (a,k) in policy]
            if set(record_crews_used_for_distribution) & set(crews_used_for_patrol):
                record.alloc_constr.RHS += self.model.M
        self.model.attacker_model.update()   

        allocation, location, defender_obj_val, _ = self.model.solve_defender(attack)

        spread = abs(attacker_obj_val - defender_obj_val)
        if spread > self.epsilon:
            raise ValueError(
                f"Heuristic A-D verification failure at node: {ran_at_bnb_node}. "
                f"a={attacker_obj_val:.8}, d={defender_obj_val:.8}, "
                f"Tol={self.epsilon:.8}."
            )

        t_heuristic = perf_counter() - t
        self.delayed_log_of_heuristic_solution = (
            ran_at_bnb_node, policy, attack, allocation, location, defender_obj_val, t_heuristic
        )
        return attacker_obj_val, t_heuristic
    
    # -------------------------------------------------------------------------
    # Tree Operations
    # -------------------------------------------------------------------------
    
    def spawn_children(self, node:Node):
        # Select the first interdependent variable (region, crew) to branch on
        var = next(iter(node.interdependency))

        child_l = Node(
            name=self.num_nodes,
            parent=node.name,
            objective_value=node.objective_value,
            decision_var=var,
            decision_var_fix=0,
            depth=node.depth+1,
            status="unprocessed"
        )
        self.num_nodes += 1

        child_r = Node(
            name=self.num_nodes,
            parent=node.name,
            objective_value=node.objective_value,
            decision_var=var,
            decision_var_fix=1,
            depth=node.depth+1,
            status="unprocessed"
        )
        self.num_nodes += 1

        # Assign children to parent node
        node.left_child = child_l
        node.right_child = child_r

        # Push children into the processing queue
        heapq.heappush(self.node_queue, child_l)
        heapq.heappush(self.node_queue, child_r)

        # Store nodes for global access
        self.node_storage.extend([child_l, child_r])

        # Update max tree depth if necessary
        if self.tree_depth < node.depth + 1:
            self.tree_depth = node.depth + 1

    def process_node(self, node:Node, timelimit:float):
        """Process a single BnB node: reset, branch, solve, classify."""

        self.model.protector_model.reset(1)
        self.model.attacker_model.reset(1)
        self.model.defender_model.reset(1)
        self.reset_variable_bound_attributes()

        # Prune if the node objective (inherited from the parent) is worse (lower) than
        # the best known lower bound
        if node.objective_value < self.lower_bound:
            node.status = 'prune'
            return 
        
        # Apply current node's branching constraints
        self.apply_branching_constraints(node)

        # fix propagated critical structures identified in previous nodes
        self.set_propagated_critical_structures(node, action="relax")

        # fix propagated critical strategies identified in previous nodes
        self.set_propagated_critical_strategies(node)

        # Solve the full 3-level model
        (
            node.policy,
            node.strategy,
            node.allocation,
            node.location,
            node.objective_value
        ) = self.model.solve_three_level_game(timelimit)

        # unfix propagated critical structures identified in previous nodes
        self.set_propagated_critical_structures(node, action="restrict")

        if node.name == 0 and self.model.root_heuristic:
            self.delayed_log_of_heuristic_solution = self.model.root_heuristic

        # Infeasibility, no DS was found
        if node.objective_value > self.model.M:
            node.status = "infea"

        # w intersects y is non-empty (fractional solution)
        elif self.is_interdependent(node):
            node.status = "frac"
            self.spawn_children(node)
            
        # The attacker behaved suboptimally (fractional solution)
        elif self.is_attacker_subopt(node):
            node.status = 'sopt'
            self.spawn_children(node) 

        # Feasible Solution to the INTERDEPENDENT problem (integer solution)
        else: 
            node.status = 'int'
            if node.objective_value > self.lower_bound: # New lower bound
                self.lower_bound = node.objective_value
                self.lower_bound_node_name = node.name
                self.best_feasible_solution = (node.objective_value, node.policy, node.strategy, node.allocation, node.location)
        
    # -------------------------------------------------------------------------
    # Main Solve
    # -------------------------------------------------------------------------   
    
    def solveBNB(self):
        self.start_time = perf_counter()
        time_remaining = self.time_limit

        while self.node_queue and self.num_nodes < self.max_num_nodes:
            node = heapq.heappop(self.node_queue)
            
            try:
                self.process_node(node, time_remaining)
            except RuntimeError:
                BNB_status_logger.info("TimeLimit Reached In Node of BNB. Terminating Sub-Optimal")
                break

            self.update_lb_ub_gap()
            self.print_status(node)

            if self.delayed_log_of_heuristic_solution:
                self.print_status_heuristic()
                self.delayed_log_of_heuristic_solution = None 

            if node.objective_value >= self.upper_bound + self.epsilon:
                BNB_status_logger.info(f"Objective value {node.objective_value} is greater than upper bound {self.upper_bound}, Node: {node.name}")
                raise ValueError("Node Objective Value is greater than upper bound") 

            if self.gap <= 1e-5:
                BNB_status_logger.info("Optimality Gap 0%: Terminating Early")
                break

            time_remaining = self.get_time_remaining()
            if time_remaining < 0:
                BNB_status_logger.info(f"TIME LIMIT EXCEEDED: {self.time_limit:.2f} seconds")
                break

        self.end_time = perf_counter()
        self.save_instance_info()
        self.model.attacker_model.update()
        return self.best_feasible_solution
