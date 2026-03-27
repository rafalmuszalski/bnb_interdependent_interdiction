import heapq
import os
from time import perf_counter
from dataclasses import dataclass, field
from typing import Any, Optional, List

from .logging_setup import BNB_status_logger, BNB_details_logger
from .model import ThreeLevelGame


@dataclass
class Node:
    name: int
    parent: Optional[int]
    status: str
    objective_value: float
    attacker_value: float
    depth: int
    decision_var: int
    decision_var_fix: int
    left_child: Optional[int] = None
    right_child: Optional[int] = None
    interdependency: list = None
    fortified_verticies: List[int] = field(default_factory=list) # P.Branch Decision made upto this node
    policy:list = None
    strategy: list = None
    structure: list = None

    def __lt__(self, other: "Node") -> bool:
        """ Sorting Nodes in ascending objective value. """
        if abs(self.objective_value - other.objective_value) < 1e-9:
            return self.decision_var_fix < other.decision_var_fix
        return self.objective_value < other.objective_value



class BranchAndBound:
    def __init__(self, model: ThreeLevelGame, timelimit: int = 60*60*3):
        self.model = model
        self.time_limit = timelimit
        self.max_num_nodes = 1e6
        self.epsilon = 1e-5

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
        self.best_feasible_solution = (float('inf'),[],[],[])
        self.upper_bound_node_name = -1
        self.lower_bound = float('-inf')
        self.upper_bound = float('inf')
        self.gap = float('nan')

        # Generate root-node and populate into PQ
        root_node = Node(
            name=0,
            parent=None,
            objective_value=float('inf'),
            attacker_value=float('inf'),
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
    
    def print_status(self, node: Node):
        if node.name == 0:
            header = (
                f"{'# Nodes':^10}{'Open':^10}{'T.Depth':^10}| {'Node':^10}{'Status':^10}"
                f"{'P.Obj':^10}{'Var':^10}{'Var_Fix':^12}{'N.Depth':^10}{'N.Parent':^10}|"
                f"{'LB':^10}{'UB':^10}{'Gap':^12}{'C.Strats':^10}{'C.Structs':^10}{'Time':^10}"
            )
            BNB_status_logger.info(header)

        var_str = str(node.decision_var) if node.decision_var is not None else 'None'
        fix_str = node.decision_var_fix if node.decision_var is not None else 'None'
        parent_str = str(node.parent) if node.parent is not None else 'None'

        BNB_status_logger.info(
            f"{self.num_nodes:^10}{len(self.node_queue):^10}{self.tree_depth:^10}|"
            f" {node.name:^10}{node.status:^10}{node.objective_value:^10.6g}"
            f"{var_str:^10}{fix_str:^12}{node.depth:^10}{parent_str:^10}|"
            f"{self.lower_bound:^10.6g}{self.upper_bound:^10.6g}{self.gap:^12.5g}"
            f"{self.model.num_critical_strategies_added:^10}{self.model.num_critical_structures_added:^10}"
            f"{perf_counter() - self.start_time:>10.5f}"
        )
        BNB_details_logger.info(
            f"Node Solution -- Node Name: {node.name}\n"
            f"Protection Policy: {node.policy}\n"
            f"Protector Price: {node.objective_value}\n"
            f"Attack Strategy: {node.strategy}\n"
            f"Defender Structure: {node.structure}\n"
            f"Defender Weight: {node.attacker_value}\n"
            f"Interdependency: {node.interdependency}\n"
            f"Status: {node.status} | Decision Var: {node.decision_var}, "
            f"Fix: {node.decision_var_fix}, Parent: {node.parent}\n"
        )

    def print_status_heuristic(self):
        name, policy, attack, structure, weight, h_time = self.delayed_log_of_heuristic_solution

        if weight < self.upper_bound + self.epsilon:
            self.upper_bound = weight
            self.upper_bound_node_name = "Heuristic"
            self.best_feasible_solution = (weight, policy, attack, structure)

        if self.upper_bound > self.epsilon:
            self.gap = (self.upper_bound - self.lower_bound) / self.upper_bound

        label = 'RH' if name == 'RootHeuristic' else 'H'
        BNB_status_logger.info(
            f"{'  HEURISTIC SOLUTION ':^30}|"
            f" {'*':^10}{label:^10}{weight:^10.6g}{'':^10}{'':^12}{'':^10}"
            f"{name if name != 'RootHeuristic' else 'RH':^10}|"
            f"{self.lower_bound:^10.6g}{self.upper_bound:^10.6g}{self.gap:^12.5g}"
            f"{'':^10}{'':^10}{f'({h_time:.5f})':>11}"
        )
        BNB_details_logger.info(
            f" -- HEURISTIC SOLUTION --\n"
            f"Protection Policy: {policy}\n"
            f"Protector Price: NOT-TRACKED\n"
            f"Attack Strategy: {attack}\n"
            f"Attacker Cost: NOT-TRACKED\n"
            f"Defender Structure: {structure}\n"
            f"Defender Weight: {weight}\n"
            f"Run at Node: {name}\n"
        )

    # -------------------------------------------------------------------------
    # Bounds & Statistics
    # -------------------------------------------------------------------------

    def update_lb_ub_gap(self):
        if self.node_queue: # Update the lower bound from the front of PQ (if any)
            # LB is the best lower bound among unexplored nodes: Ensure LB <= UB
            self.lower_bound = min(self.node_queue[0].objective_value, self.upper_bound)
        else:
            self.lower_bound = self.upper_bound

        if self.upper_bound > self.epsilon:
            self.gap = (self.upper_bound - self.lower_bound) / self.upper_bound
        else:
            self.gap = float('-inf')

    def get_time_remaining(self):
        return self.time_limit - (perf_counter() - self.start_time)

    def get_statistics(self):
        return {
            "SLURM_ARRAY_JOB_ID":  os.environ.get('SLURM_ARRAY_JOB_ID', 'local'),
            "SLURM_ARRAY_TASK_ID": os.environ.get('SLURM_ARRAY_TASK_ID', 'local'),
            "Time":    round(self.end_time - self.start_time, 2),
            "#Nodes":  self.num_nodes,
            "C.Strats":  self.model.num_critical_strategies_added,
            "C.Structs": self.model.num_critical_structures_added,
            "P.Time":  round(self.model.time_solving_protector, 5),
            "A.Time":  round(self.model.time_solving_attacker, 5),
            "D.Time":  round(self.model.time_solving_defender, 5),
            "AR.Time": round(self.model.time_solving_attacker_recourse, 5),
            "DR.Time": round(self.model.time_solving_defender_recourse, 5),
            "H.Time":  round(self.time_solving_heuristic, 5),
            "C.Strat.Propagation.Time":  round(self.time_propagating_strategies, 5),
            "C.Strat.Propagation.Calls": self.propagating_strategies_calls,
            "UB": f"{self.upper_bound:.5f}",
            "LB": f"{self.lower_bound:.5f}",
            "Opt.Gap": f"{self.gap}",
            "D.calls": self.model.D_calls,
            "A.calls": self.model.A_calls,
            "H.calls": self.H_calls,
            "Opt.Policy":    self.best_feasible_solution[1],
            "Opt.Strategy":  self.best_feasible_solution[2],
            "Opt.Structure": self.best_feasible_solution[3],
        }

    def save_instance_info(self):
        BNB_details_logger.info(
            f" ----- Instance Info Snapshot -----\n"
            f"Optimal Node: {self.upper_bound_node_name}\n"
            f"Optimal Node Obj Val: {self.best_feasible_solution[0]}\n"
            f"Optimal Policy: {self.best_feasible_solution[1]}\n"
            f"Optimal Strategy: {self.best_feasible_solution[2]}\n"
            f"Optimal Structure: {self.best_feasible_solution[3]}\n"
        )

    # -------------------------------------------------------------------------
    # Model State Management
    # -------------------------------------------------------------------------

    def reset_variable_bound_attributes(self):
        # model.reset(1) does NOT restore variable bounds (e.g. x.ub = 0 persists),
        # so we reset them explicitly here.
        for i in self.model.G.nodes():
            self.model.w_vars[i].ub = 1
            self.model.w_vars[i].lb = 0
            self.model.x_vars[i].ub = 1
            self.model.x_vars[i].lb = 0
            self.model.y_vars[i].ub = 1
            self.model.y_vars[i].lb = 0

        self.model.protector_model.update()
        self.model.attacker_model.update()
        self.model.defender_model.update()

    def apply_branching_constraints(self, node: Node):
        """Walk up the tree from node to root, applying each branching decision to the model."""
        node.fortified_verticies = []
        current = node

        while current.name != 0:
            a = current.decision_var

            if current.decision_var_fix == 1: # P.Branch
                self.model.w_vars[a].lb = 1   # force fortification
                self.model.y_vars[a].ub = 0   # disallow from DS
                node.fortified_verticies.append(a)

            elif current.decision_var_fix == 0: # D.Branch
                self.model.w_vars[a].ub = 0   # forbid fortification

            current = self.node_storage[current.parent]

        self.model.protector_model.update()
        self.model.defender_model.update()  

    def set_propagated_critical_structures(self, node:Node, action:str):
        """
        Activate or deactivate propagated DS cuts based on the current node's
        fortified vertices. A cut involving a fortified vertex is invalid at
        this node (the attacker need not hedge against it).

        This is necessary because a protector branch can invalidate structures.
        For example, if a branch fixes w_0 = 1, then y_0 = 0, and any critical
        structure involving vertex 0 becomes infeasible and must be removed
        from the attacker model, otherwise the attacker is unnecessarily worried 
        about structures with vertex 0: which cannot happen because of BNB branching.
        """

        assert action in ("relax", "restrict"), f"Invalid action: {action}"

        delta = self.model.M if action == "relax" else -self.model.M
        for record in self.model.propagated_structures:
            record.ds_constr.Lazy = self.model.structure_propagation
            if set(record.ds) & set(node.fortified_verticies):
                record.ds_constr.RHS += delta

        self.model.attacker_model.update()

    def set_propagated_critical_strategies(self, node:Node):
        """
        Similar to propagating critical structures, the same idea can be done for critical strategies.
        Instead of resolving the P-(A-D) game from scratch, we can store and propagate some critical strategies
        identified in the earlier nodes of the game. Therefore the P model begins with a few critical strategies, avoiding many (A-D) subgames.
        However, this is not as straight forward since the branching decisions affect the 
        protector and the defender and, as such, the values of the dominating sets changes. So a critical strategy in one node is valid, and then
        in the next node it may also be valid, however with a different objective value for the protector, beacuase due to branching
        that critical strategy actually results in a different dominating set with different weight... tricky 

        Please note that this propagation only happens once per node, so we never relax/restrict. We just change the RHS of the constraints.
        """
        t = perf_counter()

        for record in self.model.propagated_strategies:
            _, new_weight, _ = self.model.solve_defender(attacker_strategy=record.attack)
            record.attack_weight = new_weight
            if new_weight > self.model.t_p:
                record.attack_constr.RHS = 1
            else:
                record.attack_constr.RHS = 0    
            self.propagating_strategies_calls += 1

        self.model.protector_model.update()
        self.time_propagating_strategies += perf_counter() - t

    # -------------------------------------------------------------------------
    # Feasibility Checks
    # -------------------------------------------------------------------------
    
    def is_interdependent(self, node: Node) -> bool:
        """Check whether the protection policy and defender structure overlap."""
        node.interdependency = sorted(set(node.policy) & set(node.structure))
        return len(node.interdependency) > 0
    
    def is_attacker_subopt(self, node: Node) -> bool:
        """
        Check whether the attacker behaved suboptimally due to the relaxed defender.
        If fixing the policy and re-solving A–D yields a strictly higher objective,
        the attacker was over-hedging against structures that don't exist under
        strict interdependence.
        """
        new_obj, t_heuristic = self.run_heuristic(policy=node.policy, ran_at_bnb_node=node.name)
        self.time_solving_heuristic += t_heuristic

        if new_obj > node.attacker_value + self.epsilon:
            node.interdependency = sorted(set(node.policy) - set(node.fortified_verticies))
            assert node.interdependency, "Interdependency set is empty (SOPT)"
            return True

        return False
    
    def run_heuristic(self, policy=[], ran_at_bnb_node: str = "*"):
        """
        Compute a heuristic upper bound by fixing the given policy and solving
        the A–D subgame with full P-D interdependency enforced.

        Note: does not reset model state — should be the last solve in any node,
        as the full reset happens at the start of the next node.
        """
        self.H_calls += 1 
        t = perf_counter()

        # We need to solve attacker-defender where defender is parametrized by protector.
        # We apply this parametrization OUTSIDE of solve_attacker(), because that function is reused by independent solution
        for i in policy:
            self.model.y_vars[i].ub = 0 
        self.model.defender_model.update()
        
        # Temporarily relax cuts that are invalid under this policy
        # if the ds intersects with policy, the attacker does not have to worry about this critical struct, hence relax
        for record in self.model.propagated_structures:
            if set(record.ds) & set(policy):
                record.ds_constr.RHS += self.model.M
        self.model.attacker_model.update()

        new_attack, new_attacker_obj_val, _ = self.model.solve_attacker(policy, timer=False)

        # Undo that relaxation in order not to mess things up later.
        for record in self.model.propagated_structures:
            if set(record.ds) & set(policy):
                record.ds_constr.RHS -= self.model.M
        self.model.attacker_model.update()

        defender_structure, defender_weight, _ = self.model.solve_defender(new_attack)

        spread = abs(new_attacker_obj_val - defender_weight)
        if spread > self.epsilon:
            raise ValueError(
                f"Heuristic A-D verification failure at node: {ran_at_bnb_node}. "
                f"a={new_attacker_obj_val:.8}, d={defender_weight:.8}, "
                f"Tol={self.epsilon:.8}."
            )

        t_elapsed = perf_counter() - t
        
        return new_attacker_obj_val, t_elapsed

    # -------------------------------------------------------------------------
    # Tree Operations
    # -------------------------------------------------------------------------
    
    def spawn_children(self, node:Node):
        var = next(iter(node.interdependency))

        # P.Branch - protector assigned
        child_l = Node(
            name=self.num_nodes,
            parent=node.name,
            objective_value=node.objective_value,
            attacker_value=node.attacker_value,
            decision_var=var,
            decision_var_fix=0,
            depth=node.depth+1,
            status="unprocessed"
        )
        self.num_nodes += 1

        # D.Branch - protector forbidden, defender free
        child_r = Node(
            name=self.num_nodes,
            parent=node.name,
            objective_value=node.objective_value,
            attacker_value=node.attacker_value,
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
        
    def process_node(self, node:Node, time_limit:float):
        """Process a single BnB node: reset, branch, solve, classify."""
        
        # Reset MIP tree and solution (does not affect bounds or lazy constraints)
        self.model.protector_model.reset(1)
        self.model.attacker_model.reset(1)
        self.model.defender_model.reset(1)
        self.reset_variable_bound_attributes()

        # Prune if the node objective (inherited from the parent) is worse (higher) than UB
        if node.objective_value > self.upper_bound:
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
            node.structure,
            node.objective_value,
            node.attacker_value
        ) = self.model.solve_three_level_game(time_limit)

        # unfix propagated critical structures identified in previous nodes
        self.set_propagated_critical_structures(node, action="restrict")

        # Infeasibility, no DS was found
        if node.objective_value >= self.model.M:
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
            if node.objective_value < self.upper_bound: # New upper bound
                self.upper_bound = node.objective_value
                self.upper_bound_node_name = node.name
                self.best_feasible_solution = (
                    node.objective_value, node.policy, node.strategy, node.structure
                )

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

            if node.objective_value <= self.lower_bound - self.epsilon and self.node_queue:
                BNB_status_logger.info(
                    f"Node Objective Value {node.objective_value} is less than lower bound {self.lower_bound}, Node: {node.name}"
                    )
                raise ValueError(f"Node Objective Value {node.objective_value} is less than lower bound {self.lower_bound}, Node: {node.name}")

            if self.gap <= self.epsilon:
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
