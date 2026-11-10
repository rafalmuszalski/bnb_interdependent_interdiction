"""Problem data: facilities, survivors, regions, crews and the budgets. No Gurobi here."""
import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Instance:
    facility_locations: list[tuple[float, float]]   # V facilities on a 10x10 grid
    survivor_locations: list[tuple[float, float]]   # U survivors on the same grid
    num_crews: int                                  # K
    num_regions: int                                # R
    b_p: int                                        # protector budget: patrols
    b_a: int                                        # attacker budget: edges cut
    seed: int
    edges: list[tuple[int, int]]                    # (facility i, survivor j) pairs the defender may use
    distances: dict[tuple[int, int], int]           # distance of each edge
    facility_to_survivors: dict[int, list[int]]
    survivor_to_facilities: dict[int, list[int]]
    survivor_to_region: dict[int, int]
    region_to_survivors: dict[int, list[int]]

    @property
    def facilities(self):
        return range(len(self.facility_locations))

    @property
    def survivors(self):
        return range(len(self.survivor_locations))

    @property
    def crews(self):
        return range(self.num_crews)

    @property
    def regions(self):
        return range(self.num_regions)


def random_instance(num_facilities: int, num_survivors: int, num_crews: int, num_regions: int,
                    b_p: int, b_a: int, seed: int) -> Instance:
    """Facilities and survivors placed uniformly at random on a 10x10 grid."""
    # A private RNG: identical sequence to the old random.seed(seed) + module-level calls,
    # but doesn't touch global random state.
    rng = random.Random(seed)
    facility_locations = [(round(rng.uniform(0, 10), 2), round(rng.uniform(0, 10), 2))
                          for _ in range(num_facilities)]
    survivor_locations = [(round(rng.uniform(0, 10), 2), round(rng.uniform(0, 10), 2))
                          for _ in range(num_survivors)]

    survivor_to_region, region_to_survivors = assign_regions(survivor_locations, num_regions)
    edges, distances, facility_to_survivors, survivor_to_facilities = filtered_edges(
        facility_locations, survivor_locations, num_excluded=1)

    return Instance(facility_locations=facility_locations, survivor_locations=survivor_locations,
                    num_crews=num_crews, num_regions=num_regions, b_p=b_p, b_a=b_a, seed=seed,
                    edges=edges, distances=distances,
                    facility_to_survivors=facility_to_survivors, survivor_to_facilities=survivor_to_facilities,
                    survivor_to_region=survivor_to_region, region_to_survivors=region_to_survivors)


def assign_regions(survivor_locations: list, num_regions: int):
    """
    Split the 10x10 grid into `num_regions` rectangular regions (floor(sqrt(R)) horizontal
    strips, each cut into columns) and put each survivor in the region containing it.
    Returns (survivor -> region, region -> survivors).
    """
    survivor_to_region = {}
    region_to_survivors = {r: [] for r in range(num_regions)}

    rows = math.floor(math.sqrt(num_regions))
    cols_per_row = []
    remaining = num_regions
    for _ in range(rows):
        cols = math.ceil(remaining / (rows - len(cols_per_row)))
        cols_per_row.append(cols)
        remaining -= cols

    bounds = []  # (region, xmin, xmax, ymin, ymax)
    region = 0
    y_step = 10 / rows
    for row, cols in enumerate(cols_per_row):
        x_step = 10 / cols
        for col in range(cols):
            bounds.append((region, col * x_step, (col + 1) * x_step, row * y_step, (row + 1) * y_step))
            region += 1

    for j, (x, y) in enumerate(survivor_locations):
        for r, xmin, xmax, ymin, ymax in bounds:
            if xmin <= x < xmax and ymin <= y < ymax:
                survivor_to_region[j] = r
                region_to_survivors[r].append(j)
                break

    return survivor_to_region, region_to_survivors


def filtered_edges(facility_locations: list, survivor_locations: list, num_excluded: int = 1):
    """
    Facility-survivor edges with distance int(10 * euclidean). For each survivor, edges at
    its `num_excluded` largest distances are dropped (every edge at such a distance, ties
    included), which shrinks the bipartite graph.
    Returns (edges, distances, facility -> survivors, survivor -> facilities).
    """
    per_survivor = {}  # survivor -> [(facility, distance)]
    for i, (fx, fy) in enumerate(facility_locations):
        for j, (sx, sy) in enumerate(survivor_locations):
            per_survivor.setdefault(j, []).append((i, int(math.sqrt((fx - sx) ** 2 + (fy - sy) ** 2) * 10)))

    edges = []
    distances = {}
    facility_to_survivors = {i: [] for i in range(len(facility_locations))}
    survivor_to_facilities = {j: [] for j in range(len(survivor_locations))}
    for j, options in per_survivor.items():
        options.sort(key=lambda x: x[1], reverse=True)
        excluded = {options[k][1] for k in range(min(num_excluded, len(options)))}
        for i, dist in options:
            if dist not in excluded:
                edges.append((i, j))
                facility_to_survivors[i].append(j)
                survivor_to_facilities[j].append(i)
                distances[i, j] = dist

    return edges, distances, facility_to_survivors, survivor_to_facilities
