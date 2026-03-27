# -*- coding: utf-8 -*-
import os
import random
import math
from time import perf_counter
from dataclasses import dataclass
from typing import List, Tuple

import gurobipy as gp
from gurobipy import GRB

from .logging_setup import BNB_status_logger

@dataclass(frozen=True)
class DefenderCallbackRecord:
    alloc_weight: float
    alloc: List[Tuple[int, int]]
    alloc_constr: gp.Constr
    facility_crew_placement: List[Tuple[int,int]]

@dataclass(frozen=False)
class AttackerCallbackRecord:
    attack_weight: float
    attack: List[int]
    attack_constr: gp.Constr

class ThreeLevelGame:
    def __init__(self, facility_locations:List[Tuple[float,float]], survivor_locations:List[Tuple[float,float]], num_crews:int, num_regions:int, b_p:int, b_a:int, 
                 strategy_propagation:int, structure_propagation:int, 
                 seed:int, anti_symmetry:bool):
        
        self.seed = seed
        self.M = 1e6
        self.epsilon = 1e-3
        self.timeout = False
        self.distance_penalty = 1e3 
        random.seed(self.seed)

        self.strategy_propagation = strategy_propagation
        self.structure_propagation = structure_propagation
        self.anti_symmetry = anti_symmetry

        
        # ----- Indicies -----
        self.num_facilities = len(facility_locations) 
        self.num_survivors = len(survivor_locations)   
        self.num_crews = num_crews           
        self.num_regions = num_regions           
        self.facilities = range(self.num_facilities) # U
        self.survivors  = range(self.num_survivors)  # V
        self.crews      = range(num_crews)      # K
        self.regions      = range(num_regions)      # R

        # ----- Parameters -----
        self.b_p = b_p
        self.b_a = b_a
        self.capacity_of_crew = gp.tupledict()
        for k in range(self.num_crews):
            self.capacity_of_crew[k] = 1 

        # --------------------------------------------------------------
        self.facility_locations = facility_locations
        self.survivor_locations = survivor_locations
        (
            self.survivor_to_region_map, 
            self.region_to_survivor_map
        ) = self.generate_problem_graph()


        # Compute filtered distances and related mappings 
        # by excluding each survivor's farthest facility (num_excluded=1)
        (
            self.distances,
            self.edges,
            self.facility_to_survivor_map,
            self.survivor_to_facilities_map
        ) = self.compute_filtered_distances_per_survivor(self.facility_locations, self.survivor_locations, num_excluded=1)
        
        # ----- Statistics -----
        self.D_calls = 0
        self.A_calls = 0
        self.num_critical_structures_added = 0
        self.num_critical_strategies_added = 0
        self.time_solving_protector = 0
        self.time_solving_attacker  = 0
        self.time_solving_defender  = 0
        self.time_solving_attacker_recourse = 0
        self.time_solving_defender_recourse = 0 

        # ---- Propagate Cuts ----
        self.propagated_structures: list[DefenderCallbackRecord] = []
        self._seen_structures = set()
        self.propagated_strategies: list[AttackerCallbackRecord] = []
        self._seen_strategies = set()
        self.root_heuristic = None

        # ----- Variables -----
        self.y_vars = gp.tupledict() # Defender Allocation Vars (i,j)
        self.z_vars = gp.tupledict() # Defender Location Vars   (i,k)
        self.x_vars = gp.tupledict() # Attacker Attack Vars     (i,j)
        self.w_vars = gp.tupledict() # Protector Patrol Vars    (a,k)
        self.omega_vars = gp.tupledict() # Protector Patrol Vars (k) (anti-symmetry)
        self.zeta_vars = gp.tupledict() # Defender Location Vars (k) (anti-symmetry)
        self.alpha_var = None
        self.pi_var = None

        # ----- Initialize Models ------
        self.env = gp.Env(empty=True)
        self.env.setParam("LogToConsole", 0)
        self.env.setParam("Seed", seed)
        self.env.setParam("Threads", 8)
        self.env.setParam("SoftMemLimit", 16)
        self.env.start()

        self.defender_model = self.init_defender_model()
        self.attacker_model = self.init_attacker_model()
        self.protector_model = self.init_protector_model()


    def generate_problem_graph(self):
        """
        Partition the 10x10 spatial grid into `num_regions` regions
        and assign each survivor to one region based on location.
        
        Output:
          - survivor_to_region_map: survivor ID → region ID
          - region_to_survivor_map: region ID → list of survivor IDs
        """
        
        survivor_to_region_map = gp.tupledict()
        region_to_survivor_map = {i: [] for i in self.regions}

        # Divide grid into horizontal strips (rows)
        rows = math.floor(math.sqrt(self.num_regions))
        cols_per_row = []  # List of column counts per row

        remaining = self.num_regions
        for _ in range(rows):
            cols = math.ceil(remaining / (rows - len(cols_per_row)))
            cols_per_row.append(cols)
            remaining -= cols

        # Generate spatial bounds for each quadrant (xmin, xmax, ymin, ymax)
        cell_id = 0
        y_step = 10 / rows
        region_bounds = []
        
        for row, cols in enumerate(cols_per_row):
            x_step = 10 / cols
            for col in range(cols):
                xmin = col * x_step
                xmax = (col + 1) * x_step
                ymin = row * y_step
                ymax = (row + 1) * y_step
                region_bounds.append((cell_id, xmin, xmax, ymin, ymax))
                cell_id += 1

        # Assign each survivor to a quadrant based on coordinates
        for index, (x, y) in enumerate(self.survivor_locations):
            for cell_id, xmin, xmax, ymin, ymax in region_bounds:
                if xmin <= x < xmax and ymin <= y < ymax:
                    survivor_to_region_map[index] = cell_id
                    region_to_survivor_map[cell_id].append(index)
                    break

        return survivor_to_region_map, region_to_survivor_map

    def compute_filtered_distances_per_survivor(self, facility_locations, survivor_locations, num_excluded=1):
        """Computes distances between facilities and survivor, excluding the longest `num_excluded` edges 
        for each survivor to reduce the size of the bipartite graph.

        Args:
            facility_locations (list): List of facility coordinates (x, y).
            survivor_locations (list): List of survivor coordinates (x, y).
            num_excluded (int): Number of longest edges to exclude per survivor (default: 1).

        Returns:
            tuple: A tuple containing:
                - distances (gurobipy.tupledict): Filtered distances between facilities and survivors.
                - edges (list): List of valid edges after filtering.
                - facility_to_survivors (dict): Mapping of facilities to their assigned survivors.
                - survivors_to_facilities (dict): Mapping of survivors to their assigned facilities.
        """
        distances = gp.tupledict()
        temp_distances = {}

        # Step 1: Compute all distances
        for i, (fx, fy) in enumerate(facility_locations):
            for j, (cx, cy) in enumerate(survivor_locations):
                distance = math.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)*10
                rounded_distance = int(distance)
                if j not in temp_distances:
                    temp_distances[j] = []
                temp_distances[j].append((i, rounded_distance))  # Store (facility, distance) per survivor

        # Step 2: Exclude the longest `num_excluded` edges per survivor
        edges = []
        facility_to_survivors =  {i: [] for i in range(len(facility_locations))}
        survivor_to_facilities = {j: [] for j in range(len(survivor_locations))}
        for j, distances_list in temp_distances.items():
            if distances_list:  # Ensure there are facilities for this survivor
                # Sort distances in descending order and exclude the longest `num_excluded` ones
                distances_list.sort(key=lambda x: x[1], reverse=True)
                excluded_distances = {distances_list[k][1] for k in range(min(num_excluded, len(distances_list)))}  

                # Store distances that are not among the excluded ones
                for i, dist in distances_list:
                    if dist not in excluded_distances:
                        edges.append((i,j))
                        facility_to_survivors[i].append(j)
                        survivor_to_facilities[j].append(i)
                        distances[i, j] = dist

        return distances, edges, facility_to_survivors, survivor_to_facilities

    # -------------------------------------------------------------------------
    # Model Initialization
    # -------------------------------------------------------------------------
    def init_defender_model(self):
        m = gp.Model("Defender", env=self.env)
 
        # ------- Decision Variables -------
        # Allocation Variables
        for (i,j) in self.edges:
            self.y_vars[i,j] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
        
        # Location Variables
        for i in self.facilities:
            for k in self.crews:
                self.z_vars[i,k] = m.addVar(vtype=GRB.BINARY, name=f"z_{i}_{k}")

        for k in self.crews:
            self.zeta_vars[k] = m.addVar(vtype=GRB.BINARY, name=f"zeta_{k}")

        # ------- Constraints of the Defender Only -------
        # Define zeta[k] based on total facility assignments for crew k (Symmetry Stuff)
        for k in self.crews:
            m.addConstr(self.zeta_vars[k] == gp.quicksum(self.z_vars[i,k] for i in self.facilities),name="crew_facility_indicator")

        # Facility crew capacity constraint
        for i in self.facilities:
            m.addConstr(gp.quicksum(self.y_vars[i,j] for j in self.facility_to_survivor_map[i]) <= 
                        gp.quicksum(self.capacity_of_crew[k]*self.z_vars[i,k] for k in self.crews),name="facility_crew_capacity")

        # Ensure each crew is assigned to at most one facility
        for k in self.crews:
            m.addConstr(gp.quicksum(self.z_vars[i,k] for i in self.facilities) <= 1, name="crew_in_atmost_one_facility")

        # Each survivor is served by at most one facility
        for j in self.survivors:
            m.addConstr(gp.quicksum(self.y_vars[i,j] for i in self.survivor_to_facilities_map[j]) <= 1, name="survivor_servedby_one_facility")

        # ------- Obj Function -------
        m.setObjective(self.distance_penalty*gp.quicksum(self.y_vars[i,j] for (i,j) in self.edges) 
                       -gp.quicksum(self.distances[i,j]*self.y_vars[i,j] for (i,j) in self.edges),
                       sense=GRB.MAXIMIZE)

        
        m.update()
        return m

    def init_attacker_model(self):

        m = gp.Model("Attacker", env=self.env)
        m.Params.PreCrush = 0
        m.Params.Presolve = 0
        m.Params.Heuristics = 0
        m.Params.LazyConstraints = 1 

        # ------- Decision Variables -------
        self.alpha_var = m.addVar(vtype=GRB.CONTINUOUS, name="alpha")
        for (i,j) in self.edges:
            self.x_vars[i,j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # ------- Objective -------
        m.setObjective(self.alpha_var, GRB.MINIMIZE)

        # ------- Constraints -------
        # Attacker budget: max number of edges that can be attacked
        m.addConstr(gp.quicksum(self.x_vars[i,j] for (i,j) in self.edges) <= self.b_a)

        # Prevent unbounded model in early iterations (placeholder constraint)
        m.addConstr(self.alpha_var >= -1) 

        m.update()
        return m

    def init_protector_model(self):

        m = gp.Model("Protector", env=self.env)
        m.Params.PreCrush = 0
        m.Params.Presolve = 0
        m.Params.Heuristics = 0
        m.Params.LazyConstraints = 1 # Enable lazy constraints for callbacks

        # ------- Decision Variables -------
        self.pi_var = m.addVar(vtype=GRB.CONTINUOUS, name="pi")

        for a in self.regions:
            for k in self.crews:
                self.w_vars[a,k] = m.addVar(vtype=GRB.BINARY, name=f"w_{a}_{k}" )

        # Anti Symmetry
        for k in self.crews:
            self.omega_vars[k] = m.addVar(vtype=GRB.BINARY, name=f"omega_{k}")

        # ------- Objective -------  
        m.setObjective(self.pi_var, GRB.MAXIMIZE)

        # ------- Constraints -------
        # omega[k] = 1 if crew k is used (covers any region)
        for k in self.crews:
            m.addConstr(self.omega_vars[k] == gp.quicksum(self.w_vars[a,k] for a in self.regions),name="crew_patrol_indicator")

        # Limit number of active patrols by crew count and global patrol cap
        m.addConstr(gp.quicksum(self.w_vars[a,k] for a in self.regions for k in self.crews) <= self.num_crews, name="cap_on_patrols1")
        m.addConstr(gp.quicksum(self.w_vars[a,k] for a in self.regions for k in self.crews) <= self.b_p, name="cap_on_patrols2")

        # Each region patrolled by at most one crew
        m.addConstrs((gp.quicksum(self.w_vars[a,k] for k in self.crews) <= 1 for a in self.regions),name="one_crew_per_region")

        # Each crew can patrol at most one region
        m.addConstrs((gp.quicksum(self.w_vars[a,k] for a in self.regions) <= 1 for k in self.crews), name="crew_patrol_one_region")

        # Anti-symmetry constraints: enforce lexicographic order of crew usage
        if self.anti_symmetry:
            for k in self.crews:
                if k < self.num_crews-1: 
                    m.addConstr(self.omega_vars[k] >= self.omega_vars[k+1], name="anti-symmetry")

        # Upper bound on pi to prevent unbounded objectives in early steps
        allocation, location, defender_obj_val, d_time = self.solve_defender(attacker_strategy=[])
        self.time_solving_defender += d_time
        m.addConstr(self.pi_var <= defender_obj_val)

        m.update()
        return m

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def cut_crit_struct_callback(self, attacker_model:gp.Model, where:int, timer:bool):
        if where != GRB.Callback.MIPSOL:
            return
        
        self.D_calls += 1 
        x_sol = self.attacker_model.cbGetSolution(self.x_vars)
        attack = [(i,j) for (i,j) in self.edges if x_sol[i,j] > 0.5]

        # Solve defender response given the attack strategy
        allocation_sol, location_sol, obj_val, d_time = self.solve_defender(attacker_strategy = attack)
        if timer: self.time_solving_defender += d_time

        # If defender's response is too good, add a lazy constraint to cut it off
        if obj_val > self.attacker_model.cbGet(GRB.Callback.MIPSOL_OBJ):
            self.num_critical_structures_added += 1 
            attacker_model.cbLazy(
                self.alpha_var >= obj_val - self.M*gp.quicksum(self.x_vars[i,j] for (i,j) in allocation_sol)
            )

            # Store lazy cut to promote later...
            # Keeping attack in here just to debug sanity check
            candidate = tuple(allocation_sol)
            if candidate not in self._seen_structures:
                self.policy_local_critical_structures.append(
                    (attack, obj_val, allocation_sol, location_sol)
                )
                self._seen_structures.add(candidate)

    def cut_crit_attack_callback(self, protector_model:gp.Model, where:int, timer:bool, timelimit:float):
        if where != GRB.Callback.MIPSOL:
            return
        
        self.A_calls += 1 
        w_sol = protector_model.cbGetSolution(self.w_vars)
        protection = [(a,k) for a in self.regions for k in self.crews if w_sol[a,k] > 0.5]

        attack, attacker_obj_val, a_time = self.solve_attacker(protector_policy=protection, timer=timer, timelimit=timelimit)
        if timer: self.time_solving_attacker += a_time
        if self.timeout:
            return

        # For the given protection, the resulting attack causes Aid-delivery to be worse (lower)
        # than what the protector thought. This is a critical attack that needs to be added
        # so the protector can protect against it. 
        if attacker_obj_val < protector_model.cbGet(GRB.Callback.MIPSOL_OBJ):
            self.num_critical_strategies_added += 1 

            # Region that must be covered to block critical attack 
            must_cover_regions = set()
            for (_,j) in attack:
                must_cover_regions |= {self.survivor_to_region_map[j]}

            protector_model.cbLazy(
                self.pi_var <= attacker_obj_val + self.M*gp.quicksum(self.w_vars[a,k] for a in must_cover_regions for k in self.crews)
            )

            #Store Critical Strategies for Propagation Later
            candidate = tuple(attack)
            if candidate not in self._seen_strategies:
                self.local_critical_strategies.append((attacker_obj_val, attack))
                self._seen_strategies.add(candidate)

    # -------------------------------------------------------------------------
    # Cut Propagation
    # -------------------------------------------------------------------------

    def add_global_strategy_cuts(self):
        for weight, attack in self.local_critical_strategies:
            attacked_survivors = {j for (i,j) in attack}


            rhs = weight + self.M *gp.quicksum(self.w_vars[self.survivor_to_region_map[j],k] for j in attacked_survivors for k in self.crews)
            
            # For C.Strat popagation, we always mark constraints as constr.Lazy = 0
            constr = self.protector_model.addConstr(self.pi_var <= rhs)

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
        for  (attack, obj_val, allocation_sol, location_sol) in self.policy_local_critical_structures:
            constr = self.attacker_model.addConstr(
                self.alpha_var >= obj_val - self.M*gp.quicksum(self.x_vars[i,j] for (i,j) in allocation_sol)
            )
            constr.Lazy = self.structure_propagation

            # must store them, so that we can enable/disable them based on relevant branching conditions
            self.propagated_structures.append(
                DefenderCallbackRecord(
                    alloc_weight = obj_val,
                    alloc = allocation_sol,
                    alloc_constr = constr,
                    facility_crew_placement = location_sol
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

        self.protector_model.optimize(lambda model, where: self.cut_crit_attack_callback(model, where, timer=True, timelimit=timelimit))
        
        if self.protector_model.Status == GRB.OPTIMAL:
            protection = [(a, k) for a in self.regions for k in self.crews if self.w_vars[a, k].x > 0.5]
            protector_obj_val = self.protector_model.ObjVal
        elif self.protector_model.Status == GRB.INFEASIBLE:
            protection = []
            protector_obj_val = self.M
        elif self.protector_model.Status == GRB.TIME_LIMIT:
            raise RuntimeError("TimeLimit reached and caught in Protector Solve.")
        else:
            raise ValueError(f"Unexpected Protector Model Status: {self.protector_model.Status}")
        
        # Propagate temporary critical strategies to global cuts
        if self.strategy_propagation >= 0:
            self.add_global_strategy_cuts()

        p_time = perf_counter() - p_time
        return protection, protector_obj_val, p_time 

    def solve_attacker(self, protector_policy=[], timer=False, timelimit:float = 60*60*1):
        """
        Solves the Attacker-Defender subgame, 
        where the Defender is INDEPENDENT of the protector_policy
        """
        a_time = perf_counter()
     
        # Reset all attacker var bounds, 
        # since those are not managed by the BNB - no harm
        for (i,j) in self.edges:
            self.x_vars[i,j].ub = 1 
        self.attacker_model.update()  

        # Apply protector policy to attacker
        for (a,k) in protector_policy:
            for j in self.region_to_survivor_map[a]:
                for i in self.survivor_to_facilities_map[j]:
                    self.x_vars[i,j].ub = 0 
        self.attacker_model.update()

        # Apply protector policy to defender
        # NOTE: No! we are solving independent game
        # only bnb enforces those interdependecies 

        self.policy_local_critical_structures = []

        self.attacker_model.Params.TimeLimit = timelimit
        self.attacker_model.optimize(lambda model, where: self.cut_crit_struct_callback(model,where,timer))

        status = self.attacker_model.Status
        if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            raise ValueError(f"Unexpected attacker model status: {status}")
        if status == GRB.TIME_LIMIT:
            self.attacker_model.terminate() # Gurobi ignores python erros in callbacks so we need to HARD STOP IT
            self.timeout = True
            raise RuntimeError(f"TimeLimit reached and caught in the Attacker Model.")


        attack = [(i,j) for (i,j) in self.edges if self.x_vars[i,j].x > 0.5]
        attacker_obj_val = self.attacker_model.ObjVal

        allocation, location, defender_obj_val, d_time = self.solve_defender(attacker_strategy=attack)

        spread = abs(defender_obj_val - attacker_obj_val)
        if spread > self.epsilon:
             raise ValueError(
                f"A-D subgame verification failed. "
                f"a={attacker_obj_val:.9}, d={defender_obj_val:.9} "
                f"diff={abs(attacker_obj_val - defender_obj_val):.9}"
            )

        # We ran a free heuristic at root node, so lets store it and use that.
        if protector_policy == [] and self.root_heuristic is None: 
            self.root_heuristic = ("RootHeuristic", protector_policy, attack, allocation, location, defender_obj_val, perf_counter()-a_time)
        
        # propagate the temporary critical structures to propagated, global(all nodes) cuts
        if self.structure_propagation >= 0: #-1=no propagation, 0=regular constr, 1=lazy, 2=lazy. 
            self.add_global_structure_cuts()
 
        # Reset attacker var bounds. Not managed by BNB 
        for (a,k) in protector_policy:
            for j in self.region_to_survivor_map[a]:
                for i in self.survivor_to_facilities_map[j]:
                    self.x_vars[i,j].ub = 1 
        self.attacker_model.update()

        a_time = perf_counter() - a_time
        return attack, attacker_obj_val, a_time

    def solve_defender(self, attacker_strategy):
        """
        Solves the INDEPENDENT relaxation. 
        The defender is NOT influenced by protector policy
        """
        d_time = perf_counter()

        # Reset defender variable bounds (y,z)
        # NOTE: No! These bounds are managed by the bnb branching decisions.
        # You cannot reset them here. 

        # Disable allocations on attacked edges (Apply Attacker Strategy)
        original_bound = {}
        for (i,j) in attacker_strategy:
            original_bound[i,j] = self.y_vars[i,j].ub
            self.y_vars[i,j].ub = 0 
        self.defender_model.update()

        # Solve the model
        self.defender_model.optimize()

        if self.defender_model.Status == GRB.OPTIMAL:
            # Extract defender decisions
            allocation =[(i,j) for (i,j) in self.edges if self.y_vars[i,j].x > 0.5]
            location = [(i,k) for i in self.facilities for k in self.crews if self.z_vars[i,k].x > 0.5]
            defender_obj_val = self.defender_model.ObjVal
        elif self.defender_model.Status == GRB.INFEASIBLE:
            allocation = []
            location = []
            defender_obj_val = self.M
        else:
            raise ValueError(f"Unexpected Defender Model Status: {self.defender_model.Status}")

     
        # Reset Attacker Strategy on Defender
        for (i,j) in attacker_strategy:
            self.y_vars[i,j].ub = original_bound[i,j] 
        self.defender_model.update()

        d_time = perf_counter() - d_time
        return allocation, location, defender_obj_val, d_time

    def solve_recourse(self, protector_policy=[], attacker_strategy=[]):
        """
        Solve for the defender’s INTERDEPENDENT solution given a fixed protection
        policy and attacker strategy.

        Note: `solve_defender` is NOT analogous, as it ignores the
        protector–defender interaction. That coupling is handled explicitly
        within the BnB branching scheme.
        """

        # Reset every var in the defender 
        for (i,j) in self.edges:
            self.y_vars[i,j].ub = 1 
        self.defender_model.update()

        # Apply Protection Policy
        for (a,k) in protector_policy:
            self.zeta_vars[k].ub = 0 # Anti Symmetry thing 
            for i in self.facilities:
                self.z_vars[i,k].ub = 0
            
        # Apply Attacker Strategy
        for (i,j) in attacker_strategy:
            self.y_vars[i,j].ub = 0

        # Verify the Attacker isn't targetting some protected regions
        attacked_survivors = {j for (i,j) in attacker_strategy}
        for (a,k) in protector_policy:
            overlap = attacked_survivors & set(self.region_to_survivor_map[a])
            if overlap:
                raise ValueError(f"Attack Strategy is attacking a protected survivor in region: {overlap}")

        self.defender_model.update()

        self.defender_model.optimize()

        if self.defender_model.Status == GRB.OPTIMAL:
            allocation =[(i,j) for (i,j) in self.edges if self.y_vars[i,j].x > 0.5]
            location = [(i,k) for i in self.facilities for k in self.crews if self.z_vars[i,k].x > 0.5]
            defender_obj_val = self.defender_model.ObjVal
        elif self.defender_model.Status == GRB.INFEASIBLE:
            allocation = []
            location = []
            defender_obj_val = self.M
        else:
            raise ValueError(f" Recourse Defender Status: {self.defender_model.Status}")
        
        # Taking no chances, Destroy the model after running this. You should't need this except debugging
        del self.defender_model
        del self.y_vars
        del self.z_vars
        BNB_status_logger.warning("DEFENDER MODEL DESTROYED FOR YOUR OWN SAFETY. USE THIS FUNC CAREFULLY")

        return allocation, location, defender_obj_val

    # -------------------------------------------------------------------------
    # Main Solve
    # -------------------------------------------------------------------------

    def solve_three_level_game(self, timelimit:int):
        """
        Solves the relaxation problem. The branching conditions are enforced by the BrandAndBound class.
        After we solve the protector problem, we need to recalculate recourses to get the full triplet of solutions
        """
        protector_policy, pObjVal, p_time = self.solve_protector(timelimit)
        attacker_strategy, aObjVal, a_time = self.solve_attacker(protector_policy, timer=False, timelimit=timelimit-p_time)
        defender_allocation, defender_location, dObjVal, d_time = self.solve_defender(attacker_strategy)

        self.time_solving_protector          += p_time
        self.time_solving_attacker_recourse  += a_time
        self.time_solving_defender_recourse  += d_time

        spread = max(pObjVal, aObjVal, dObjVal) - min(pObjVal, aObjVal, dObjVal)
        if spread > 0.01:
            raise ValueError(
                f"Objective mismatch: p={pObjVal:.9}, a={aObjVal:.9}, d={dObjVal:.9}, "
                f"spread={spread:.9}, eps={self.epsilon:.9}"
            )

        return (protector_policy, attacker_strategy, defender_allocation, defender_location, dObjVal)
