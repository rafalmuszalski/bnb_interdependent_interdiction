"""Problem data: facilities, survivors, regions, crews and the budgets. No Gurobi here.

All geometry is in integers, so distances and region membership are exact (no float rounding):
coordinates are hundredths on the 10x10 square (0..GRID), distances are tenths.
"""
import math
import random
from dataclasses import dataclass

SIDE = 10                 # the square is SIDE x SIDE
COORD_SCALE = 100         # coordinates are stored in hundredths
GRID = SIDE * COORD_SCALE  # integer coordinates lie in 0..GRID


@dataclass(frozen=True)
class Instance:
    facility_locations: list[tuple[int, int]]       # V facilities, integer coordinates in 0..GRID
    survivor_locations: list[tuple[int, int]]       # U survivors on the same grid
    num_crews: int                                  # K
    num_regions: int                                # R
    b_p: int                                        # protector budget: patrols
    b_a: int                                        # attacker budget: edges cut
    seed: int
    edges: list[tuple[int, int]]                    # (facility i, survivor j) pairs the defender may use
    distances: dict[tuple[int, int], int]           # distance of each edge, in tenths
    penalty: int                                    # P: defender's reward per survivor served (see survivor_penalty)
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
    facility_locations = [(random_coordinate(rng), random_coordinate(rng)) for _ in range(num_facilities)]
    survivor_locations = [(random_coordinate(rng), random_coordinate(rng)) for _ in range(num_survivors)]

    survivor_to_region, region_to_survivors = assign_regions(survivor_locations, num_regions)
    edges, distances, facility_to_survivors, survivor_to_facilities = filtered_edges(
        facility_locations, survivor_locations, num_excluded=1)
    penalty = survivor_penalty(distances, survivor_to_facilities, num_crews)

    return Instance(facility_locations=facility_locations, survivor_locations=survivor_locations,
                    num_crews=num_crews, num_regions=num_regions, b_p=b_p, b_a=b_a, seed=seed,
                    edges=edges, distances=distances, penalty=penalty,
                    facility_to_survivors=facility_to_survivors, survivor_to_facilities=survivor_to_facilities,
                    survivor_to_region=survivor_to_region, region_to_survivors=region_to_survivors)


def random_coordinate(rng: random.Random) -> int:
    """A uniform point of [0, SIDE] in hundredths. Same points as the old round(uniform(0, 10), 2)."""
    return round(round(rng.uniform(0, SIDE), 2) * COORD_SCALE)


def assign_regions(survivor_locations: list, num_regions: int):
    """
    Split the 10x10 grid into `num_regions` rectangular regions (floor(sqrt(R)) horizontal
    strips, each cut into columns) and put each survivor in the region containing it.
    A region holds its lower edges; points on the square's top/right edge go to the last
    row/column. Returns (survivor -> region, region -> survivors).
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
    first_region = [sum(cols_per_row[:row]) for row in range(rows)]  # regions are numbered row by row

    for j, (x, y) in enumerate(survivor_locations):
        row = min(y * rows // GRID, rows - 1)
        cols = cols_per_row[row]
        col = min(x * cols // GRID, cols - 1)
        r = first_region[row] + col
        survivor_to_region[j] = r
        region_to_survivors[r].append(j)

    return survivor_to_region, region_to_survivors


def distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """floor(10 * euclidean distance), computed exactly: the coordinates are hundredths,
    so 10 * euclidean = sqrt(dx^2 + dy^2) / 10, and floor(sqrt(n) / 10) = isqrt(n) // 10."""
    dx, dy = a[0] - b[0], a[1] - b[1]
    return math.isqrt(dx * dx + dy * dy) // (COORD_SCALE // 10)


def filtered_edges(facility_locations: list, survivor_locations: list, num_excluded: int = 1):
    """
    Facility-survivor edges with distance floor(10 * euclidean). For each survivor, edges at
    its `num_excluded` largest distances are dropped (every edge at such a distance, ties
    included), which shrinks the bipartite graph.
    Returns (edges, distances, facility -> survivors, survivor -> facilities).
    """
    per_survivor = {}  # survivor -> [(facility, distance)]
    for i, f in enumerate(facility_locations):
        for j, s in enumerate(survivor_locations):
            per_survivor.setdefault(j, []).append((i, distance(f, s)))

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


def survivor_penalty(distances: dict, survivor_to_facilities: dict, num_crews: int) -> int:
    """
    P = D_max + 1, the smallest reward per survivor served that makes serving one more
    survivor always beat any saving in distance.
    D_max bounds the total distance of any allocation: each crew serves at most one survivor,
    so at most min(U, K) are served, each over at most its longest remaining edge.
    Then serving n + 1 survivors is worth >= (n + 1) * P - D_max = n * P + 1 > n * P,
    which is the most serving n can be worth.
    """
    longest = sorted((max((distances[i, j] for i in facilities), default=0)
                      for j, facilities in survivor_to_facilities.items()), reverse=True)
    d_max = sum(longest[:num_crews])  # the min(U, K) longest
    return d_max + 1
