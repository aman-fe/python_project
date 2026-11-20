"""
Autonomous Missile Pathfinding in a 2D Airspace - AI with Python Course Project
===============================================================================

PROBLEM FORMULATION
  State          : (row, col) airspace sector occupied by the missile
  Initial state  : LAUNCH site         Goal state : TARGET coordinates
  Actions        : 8-directional flight (N, NE, E, SE, S, SW, W, NW)
  Constraints    : MOUNTAIN and NO-FLY ZONE (NFZ) cells are impassable;
                   diagonal flight cannot cut corners across mountains/NFZs
  Step cost      : distance (1.0 cardinal, 1.41 diagonal) * airspace risk
                   (1x in Clear Air, 5x inside SAM Radar zones)
  Heuristic h(n) : Octile distance to target (admissible: min risk = 1x)

ALGORITHMS
  Uninformed : BFS, DFS, Uniform Cost Search (UCS), Iterative Deepening (IDS)
  Informed   : Greedy Best-First, A*, Theta*

USAGE
  python missile_pathfinding.py               -> launch Radar GUI
  python missile_pathfinding.py --benchmark   -> headless comparison on random maps
"""
import heapq
import math
import random
import sys
import time
from collections import deque

# ----------------------------------------------------------------- environment
CLEAR = 0
MOUNTAIN = 1
NFZ = 2
SAM_RADAR = 3        # High-risk inner core (6x risk)
SAM_OUTER = 4        # Medium-risk outer ring (3x risk)

# Tactical airspace risk multipliers (1.0 = standard cruise fuel burn)
RISK = {
    CLEAR: 1.0,
    SAM_RADAR: 6.0,
    SAM_OUTER: 3.0,
}
IMPASSABLE = {MOUNTAIN, NFZ}

# 8 compass flight directions
MOVES = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
]
SQRT2 = math.sqrt(2.0)


def in_bounds(grid, r, c):
    """Check if row and column lie inside the grid boundaries."""
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])


def neighbors(grid, node):
    """
    Successor function: legal 8-way flight moves with distance * radar risk cost.
    Guarantees diagonal moves never clip impassable mountain or NFZ corners.
    """
    r, c = node
    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        if not in_bounds(grid, nr, nc) or grid[nr][nc] in IMPASSABLE:
            continue
        # Prevent diagonal corner-cutting across mountains or no-fly zones
        if dr != 0 and dc != 0:
            if grid[r + dr][c] in IMPASSABLE or grid[r][c + dc] in IMPASSABLE:
                continue
        dist = SQRT2 if (dr != 0 and dc != 0) else 1.0
        yield (nr, nc), dist * RISK[grid[nr][nc]]


def heuristic(a, b):
    """Admissible and consistent Octile distance heuristic for 8-directional flight."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr + dc) + (SQRT2 - 2.0) * min(dr, dc)


def rebuild(parent, goal):
    """Reconstruct path from parent pointers starting at goal backwards to start."""
    path, n = [], goal
    while n is not None:
        path.append(n)
        n = parent[n]
    return path[::-1]


def line_of_sight_and_cost(grid, p1, p2):
    """
    Ray-marching Line-of-Sight and continuous risk cost integration.
    Verifies that the line segment from p1 to p2 does not intersect or graze
    any MOUNTAIN or NFZ cells (including corner junctions), and integrates the
    exact risk-weighted path cost along the segment.
    Returns (has_los: bool, integrated_cost: float).
    """
    r1, c1 = p1
    r2, c2 = p2
    if p1 == p2:
        return True, 0.0

    dr = r2 - r1
    dc = c2 - c1
    total_dist = math.hypot(dr, dc)

    # Parametric segment from (r1+0.5, c1+0.5) to (r2+0.5, c2+0.5) for t in [0, 1]
    t_values = [0.0, 1.0]

    # Find all crossings with vertical grid boundaries (c = integer)
    min_c, max_c = min(c1, c2), max(c1, c2)
    if dc != 0:
        for c_bound in range(min_c + 1, max_c + 1):
            t = (c_bound - (c1 + 0.5)) / dc
            if 0.0 < t < 1.0:
                t_values.append(t)

    # Find all crossings with horizontal grid boundaries (r = integer)
    min_r, max_r = min(r1, r2), max(r1, r2)
    if dr != 0:
        for r_bound in range(min_r + 1, max_r + 1):
            t = (r_bound - (r1 + 0.5)) / dr
            if 0.0 < t < 1.0:
                t_values.append(t)

    t_values.sort()

    # Deduplicate boundary points that are within floating point epsilon
    merged_t = [t_values[0]]
    for t in t_values[1:]:
        if t - merged_t[-1] > 1e-9:
            merged_t.append(t)

    # Strict corner guard: if segment passes directly through an integer vertex
    for t in merged_t:
        if 0.0 < t < 1.0:
            y = r1 + 0.5 + t * dr
            x = c1 + 0.5 + t * dc
            round_y = round(y)
            round_x = round(x)
            if abs(y - round_y) < 1e-7 and abs(x - round_x) < 1e-7:
                # 4 adjacent cells sharing this corner
                for adj_r in (round_y - 1, round_y):
                    for adj_c in (round_x - 1, round_x):
                        if in_bounds(grid, adj_r, adj_c) and grid[adj_r][adj_c] in IMPASSABLE:
                            return False, float("inf")

    # Integrate airspace risk across each subsegment
    total_cost = 0.0
    for i in range(len(merged_t) - 1):
        ta = merged_t[i]
        tb = merged_t[i + 1]
        t_mid = 0.5 * (ta + tb)
        ymid = r1 + 0.5 + t_mid * dr
        xmid = c1 + 0.5 + t_mid * dc
        cell_r = int(math.floor(ymid))
        cell_c = int(math.floor(xmid))

        if not in_bounds(grid, cell_r, cell_c):
            return False, float("inf")

        val = grid[cell_r][cell_c]
        if val in IMPASSABLE:
            return False, float("inf")

        dt = tb - ta
        sub_dist = dt * total_dist
        total_cost += sub_dist * RISK.get(val, 1.0)

    return True, total_cost


def straight_tie_breaker(start, goal, curr, p):
    """
    Straight-line tie breaker for A* and UCS.
    Penalizes deviations from the direct start-to-goal vector and heading changes
    so equal-cost paths prefer continuing straight without zig-zagging.
    """
    sr, sc = start
    gr, gc = goal
    r, c = curr
    cross = abs((r - sr) * (gc - sc) - (c - sc) * (gr - sr))
    turn = 0.0
    if p is not None:
        pr, pc = p
        p_dr, p_dc = r - pr, c - pc
        g_dr, g_dc = gr - r, gc - c
        dot = (p_dr * g_dr + p_dc * g_dc)
        norm = math.hypot(p_dr, p_dc) * math.hypot(g_dr, g_dc)
        if norm > 1e-6:
            turn = 1.0 - (dot / norm)
    return cross * 0.001 + turn * 0.01


# ------------------------------------------------------------------ algorithms
def bfs(grid, start, goal):
    """Breadth-First Search (uninformed, finds path with fewest hops)."""
    frontier, parent, order, peak = deque([start]), {start: None}, [], 1
    while frontier:
        peak = max(peak, len(frontier))
        n = frontier.popleft()
        order.append(n)
        if n == goal:
            return rebuild(parent, n), order, peak
        for nb, _ in neighbors(grid, n):
            if nb not in parent:
                parent[nb] = n
                frontier.append(nb)
    return None, order, peak


def dfs(grid, start, goal):
    """Depth-First Search (uninformed, aggressive exploration)."""
    stack, parent, seen, order, peak = [(start, None)], {}, set(), [], 1
    while stack:
        peak = max(peak, len(stack))
        n, p = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        parent[n] = p
        order.append(n)
        if n == goal:
            return rebuild(parent, n), order, peak
        for nb, _ in reversed(list(neighbors(grid, n))):
            if nb not in seen:
                stack.append((nb, n))
    return None, order, peak


def best_first(grid, start, goal, mode):
    """
    Priority-based search engine:
      mode 'ucs'    : f(n) = g(n)
      mode 'greedy' : f(n) = h(n)
      mode 'astar'  : f(n) = g(n) + h(n)
    Includes straight-line tie breaking to prevent unnecessary zig-zags.
    """
    def f(g, n):
        h = heuristic(n, goal)
        return g if mode == "ucs" else h if mode == "greedy" else g + h

    tie = 0
    tb = straight_tie_breaker(start, goal, start, None)
    heap = [(round(f(0.0, start), 7), tb, tie, start)]
    g_cost, parent, closed, order, peak = {start: 0.0}, {start: None}, set(), [], 1

    while heap:
        peak = max(peak, len(heap))
        _, _, _, n = heapq.heappop(heap)
        if n in closed:
            continue
        closed.add(n)
        order.append(n)
        if n == goal:
            return rebuild(parent, n), order, peak

        for nb, c in neighbors(grid, n):
            ng = g_cost[n] + c
            if nb not in g_cost or (mode != "greedy" and ng < g_cost[nb] - 1e-9):
                g_cost[nb], parent[nb] = ng, n
                tie += 1
                tb = straight_tie_breaker(start, goal, nb, n)
                heapq.heappush(heap, (round(f(ng, nb), 7), tb, tie, nb))
    return None, order, peak


def ucs(grid, s, g):
    """Uniform Cost Search (finds lowest-cost path without heuristic)."""
    return best_first(grid, s, g, "ucs")


def greedy(grid, s, g):
    """Greedy Best-First Search (guided purely by distance heuristic)."""
    return best_first(grid, s, g, "greedy")


def astar(grid, s, g):
    """A* Search (optimal informed search balancing cost and heuristic)."""
    return best_first(grid, s, g, "astar")


def thetastar(grid, start, goal):
    """
    Theta* Search (Any-Angle A* with Line-of-Sight parent relaxation).
    Relaxes parent pointers directly to grandparent if an unobstructed line-of-sight
    corridor exists, integrating true airspace risk to avoid radar domes.
    """
    tie = 0
    tb = straight_tie_breaker(start, goal, start, None)
    heap = [(round(heuristic(start, goal), 7), tb, tie, start)]
    g_cost, parent, closed, order, peak = {start: 0.0}, {start: None}, set(), [], 1

    while heap:
        peak = max(peak, len(heap))
        _, _, _, curr = heapq.heappop(heap)
        if curr in closed:
            continue
        closed.add(curr)
        order.append(curr)
        if curr == goal:
            return rebuild(parent, curr), order, peak

        p = parent[curr]
        for nb, step_c in neighbors(grid, curr):
            if nb in closed:
                continue

            best_p = curr
            best_g = g_cost[curr] + step_c

            # Path 2 relaxation: direct line from parent[curr] to nb
            if p is not None:
                has_los, los_cost = line_of_sight_and_cost(grid, p, nb)
                if has_los:
                    g2 = g_cost[p] + los_cost
                    if g2 < best_g - 1e-9:
                        best_g = g2
                        best_p = p

            if nb not in g_cost or best_g < g_cost[nb] - 1e-9:
                g_cost[nb], parent[nb] = best_g, best_p
                tie += 1
                f = best_g + heuristic(nb, goal)
                tb = straight_tie_breaker(start, goal, nb, best_p)
                heapq.heappush(heap, (round(f, 7), tb, tie, nb))

    return None, order, peak


def ids(grid, start, goal):
    """Iterative Deepening Search over airspace sectors with best-depth pruning."""
    sys.setrecursionlimit(10000)
    total_expanded, peak = 0, 1
    max_depth = len(grid) * len(grid[0])

    for limit in range(max_depth + 1):
        best_depth, parent, order = {}, {}, []

        def dls(n, depth, p):
            nonlocal peak
            if n in best_depth and best_depth[n] <= depth:
                return False
            best_depth[n] = depth
            parent[n] = p
            order.append(n)
            peak = max(peak, depth + 1)
            if n == goal:
                return True
            if depth == limit:
                return False
            for nb, _ in neighbors(grid, n):
                if dls(nb, depth + 1, n):
                    return True
            return False

        found = dls(start, 0, None)
        total_expanded += len(order)
        if found:
            return rebuild(parent, goal), order, peak, total_expanded
        if best_depth and limit > 0 and not any(d == limit for d in best_depth.values()):
            break
    return None, order, peak, total_expanded


ALGORITHMS = {
    "A*": astar,
    "Theta*": thetastar,
    "UCS": ucs,
    "Greedy": greedy,
    "BFS": bfs,
    "DFS": dfs,
    "IDS": ids,
}


# -------------------------------------------------- metrics & path smoothing
def count_sharp_turns(path):
    """
    Count number of abrupt direction changes (>= 45 degrees) along the path.
    Used as an aerodynamic guidance quality metric.
    """
    if not path or len(path) < 3:
        return 0
    turns = 0
    for i in range(1, len(path) - 1):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        v1 = (r1 - r0, c1 - c0)
        v2 = (r2 - r1, c2 - c1)
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        m1 = math.hypot(v1[0], v1[1])
        m2 = math.hypot(v2[0], v2[1])
        if m1 * m2 > 1e-9:
            cos_a = max(-1.0, min(1.0, dot / (m1 * m2)))
            angle_deg = math.degrees(math.acos(cos_a))
            if angle_deg >= 44.99:
                turns += 1
    return turns


def sam_exposure(grid, path):
    """
    Compute total radar-weighted penalty accumulated along the path.
    Clear air contributes 0.0 penalty; SAM zones contribute (risk - 1.0) * dist.
    """
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for p1, p2 in zip(path, path[1:]):
        r1, c1 = p1
        r2, c2 = p2
        if abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
            dist = SQRT2 if (r1 != r2 and c1 != c2) else 1.0
            r_val = RISK.get(grid[r2][c2], 1.0)
            total += dist * (r_val - 1.0)
        else:
            dr = r2 - r1
            dc = c2 - c1
            tot_dist = math.hypot(dr, dc)
            steps = max(10, int(tot_dist * 5))
            for s in range(steps):
                t = (s + 0.5) / steps
                y = r1 + 0.5 + t * dr
                x = c1 + 0.5 + t * dc
                ir = int(math.floor(y))
                ic = int(math.floor(x))
                if in_bounds(grid, ir, ic):
                    total += (tot_dist / steps) * (RISK.get(grid[ir][ic], 1.0) - 1.0)
    return round(total, 1)


def path_cost(grid, path):
    """
    Compute total flight cost along the path.
    Supports both standard grid waypoints and any-angle non-adjacent segments.
    """
    if not path or len(path) < 2:
        return 0.0 if path else None
    total = 0.0
    for p1, p2 in zip(path, path[1:]):
        r1, c1 = p1
        r2, c2 = p2
        if abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1:
            dist = SQRT2 if (r1 != r2 and c1 != c2) else 1.0
            total += dist * RISK.get(grid[r2][c2], 1.0)
        else:
            _, c = line_of_sight_and_cost(grid, p1, p2)
            total += c
    return round(total, 1)


def is_point_safe(grid, y, x):
    """Verify that a continuous point (y, x) remains safely clear of impassable cells."""
    r = int(math.floor(y))
    c = int(math.floor(x))
    if not in_bounds(grid, r, c) or grid[r][c] in IMPASSABLE:
        return False
    # Verify buffer distance to nearby mountain/NFZ boundaries
    for ir in (r - 1, r, r + 1):
        for ic in (c - 1, c, c + 1):
            if in_bounds(grid, ir, ic) and grid[ir][ic] in IMPASSABLE:
                near_y = max(ir, min(y, ir + 1.0))
                near_x = max(ic, min(x, ic + 1.0))
                if (y - near_y) ** 2 + (x - near_x) ** 2 < 0.04:  # 0.2 cell margin
                    return False
    return True


def is_segment_safe(grid, p1, p2):
    """Verify that every sample on segment between p1 and p2 is collision-free."""
    y1, x1 = p1
    y2, x2 = p2
    d = math.hypot(y2 - y1, x2 - x1)
    steps = max(10, int(d * 15))
    for s in range(steps + 1):
        t = s / steps
        y = y1 + t * (y2 - y1)
        x = x1 + t * (x2 - x1)
        if not is_point_safe(grid, y, x):
            return False
    return True


def string_pulling(grid, path):
    """
    Line-of-Sight string-pulling (shortcutting).
    Removes redundant collinear or visible waypoints while preserving obstacle clearance.
    """
    if not path or len(path) <= 2:
        return list(path) if path else []
    simplified = [path[0]]
    curr_idx = 0
    while curr_idx < len(path) - 1:
        furthest = curr_idx + 1
        for nxt in range(len(path) - 1, curr_idx, -1):
            has_los, _ = line_of_sight_and_cost(grid, path[curr_idx], path[nxt])
            if has_los:
                furthest = nxt
                break
        simplified.append(path[furthest])
        curr_idx = furthest
    return simplified


def smooth_trajectory(grid, path, iterations=3):
    """
    Safe Aerodynamic Path Smoother:
    1. Line-of-Sight String-Pulling to extract essential turning waypoints.
    2. Chaikin corner-cutting with strict collision guards against MOUNTAIN/NFZ cells.
    Returns a sequence of smooth continuous (y, x) coordinates for missile flight.
    """
    if not path:
        return []
    if len(path) == 1:
        return [(float(path[0][0]), float(path[0][1]))]

    pts = [(float(r), float(c)) for r, c in string_pulling(grid, path)]
    if len(pts) <= 2:
        return pts

    for _ in range(iterations):
        if len(pts) < 3:
            break
        new_pts = [pts[0]]
        for i in range(len(pts) - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])

            # Collision guard: ensure corner cut doesn't clip into terrain or NFZ
            if is_segment_safe(grid, new_pts[-1], q) and is_segment_safe(grid, q, r):
                new_pts.extend([q, r])
            else:
                new_pts.append(p0)
                new_pts.append(p1)
        new_pts.append(pts[-1])
        pts = new_pts

    # Filter out redundant micro-steps
    filtered = [pts[0]]
    for p in pts[1:]:
        if math.hypot(p[0] - filtered[-1][0], p[1] - filtered[-1][1]) > 0.05:
            filtered.append(p)
    return filtered


def run(algo, grid, start, goal):
    """Execute a guidance algorithm and compute full flight telemetry."""
    t0 = time.perf_counter()
    res = ALGORITHMS[algo](grid, start, goal)
    elapsed = (time.perf_counter() - t0) * 1000
    path, order, peak = res[0], res[1], res[2]
    expanded = res[3] if len(res) > 3 else len(order)

    return {
        "algo": algo,
        "path": path,
        "smoothed_path": smooth_trajectory(grid, path) if path else None,
        "order": order,
        "found": path is not None,
        "waypoints": len(path) - 1 if path else None,
        "cost": path_cost(grid, path),
        "sam_exposure": sam_exposure(grid, path),
        "sharp_turns": count_sharp_turns(path),
        "expanded": expanded,
        "frontier": peak,
        "ms": elapsed,
    }


# ------------------------------------------------------------- map generation
def random_airspace(rows=20, cols=24, mountains=0.20, nfz=0.05, radar=0.14, seed=None):
    """
    Generate realistic contiguous tactical airspace features:
    - Mountain Ranges: Connected ridgelines and clustered terrain formations.
    - No-Fly Zones (NFZ): Solid rectangular or circular restricted diplomatic corridors.
    - SAM Radar Domes: Circular engagement zones with 6x inner core and 3x outer ring.
    - Launch & Target sites with clear safety bubbles and verified BFS path reachability.
    """
    rng = random.Random(seed)
    start = (rows // 2, 1)
    goal = (rows // 2, cols - 2)

    while True:
        grid = [[CLEAR] * cols for _ in range(rows)]

        # 1. Mountain Ranges: connected ridgelines & terrain clusters
        num_ranges = rng.randint(2, 4)
        for _ in range(num_ranges):
            curr_r = rng.randint(1, rows - 2)
            curr_c = rng.randint(2, cols - 3)
            length = rng.randint(6, 12)
            angle = rng.uniform(0, 2 * math.pi)
            dr = math.sin(angle)
            dc = math.cos(angle)
            for _ in range(length):
                ir, ic = int(round(curr_r)), int(round(curr_c))
                if in_bounds(grid, ir, ic):
                    grid[ir][ic] = MOUNTAIN
                if rng.random() < 0.6:
                    adj_r = ir + rng.choice([-1, 0, 1])
                    adj_c = ic + rng.choice([-1, 0, 1])
                    if in_bounds(grid, adj_r, adj_c):
                        grid[adj_r][adj_c] = MOUNTAIN
                curr_r += dr + rng.uniform(-0.4, 0.4)
                curr_c += dc + rng.uniform(-0.4, 0.4)

        # 2. No-Fly Zones (NFZ): solid rectangular or circular restricted zones
        num_nfz = rng.randint(1, 2)
        for _ in range(num_nfz):
            shape = rng.choice(["rect", "circle"])
            if shape == "rect":
                h = rng.randint(3, 5)
                w = rng.randint(3, 6)
                r0 = rng.randint(1, max(1, rows - h - 1))
                c0 = rng.randint(3, max(3, cols - w - 3))
                for r in range(r0, r0 + h):
                    for c in range(c0, c0 + w):
                        if in_bounds(grid, r, c):
                            grid[r][c] = NFZ
            else:
                radius = rng.uniform(1.8, 3.0)
                r0 = rng.randint(2, rows - 3)
                c0 = rng.randint(4, cols - 5)
                for r in range(int(r0 - radius), int(r0 + radius + 1)):
                    for c in range(int(c0 - radius), int(c0 + radius + 1)):
                        if in_bounds(grid, r, c) and (r - r0) ** 2 + (c - c0) ** 2 <= radius ** 2:
                            grid[r][c] = NFZ

        # 3. SAM Radar Domes: circular engagement zones (inner core 6x, outer ring 3x)
        num_sam = rng.randint(2, 4)
        for _ in range(num_sam):
            r_sam = rng.randint(2, rows - 3)
            c_sam = rng.randint(3, cols - 4)
            outer_rad = rng.uniform(2.8, 4.0)
            inner_rad = rng.uniform(1.2, 1.8)
            for r in range(int(r_sam - outer_rad), int(r_sam + outer_rad + 1)):
                for c in range(int(c_sam - outer_rad), int(c_sam + outer_rad + 1)):
                    if not in_bounds(grid, r, c) or grid[r][c] in IMPASSABLE:
                        continue
                    d2 = (r - r_sam) ** 2 + (c - c_sam) ** 2
                    if d2 <= inner_rad ** 2:
                        grid[r][c] = SAM_RADAR
                    elif d2 <= outer_rad ** 2:
                        grid[r][c] = SAM_OUTER

        # Clear launch & target safety bubbles
        for br, bc in (start, goal):
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = br + dr, bc + dc
                    if in_bounds(grid, nr, nc) and (dr * dr + dc * dc) <= 5:
                        grid[nr][nc] = CLEAR

        # Verify reachability via BFS
        if bfs(grid, start, goal)[0]:
            return grid, start, goal


# ------------------------------------------------------------------ benchmark
def benchmark(trials=25, rows=20, cols=24):
    """
    Headless Monte Carlo benchmark evaluating all 7 guidance algorithms across
    identical random tactical theaters. Computes core AI search efficiency and
    missile guidance quality metrics.
    """
    stats = {
        a: dict(ms=0.0, exp=0, front=0, cost=0.0, wp=0, sam=0.0, turns=0.0, ok=0, opt=0)
        for a in ALGORITHMS
    }

    for t in range(trials):
        g, s, e = random_airspace(rows, cols, seed=t)
        results = {a: run(a, g, s, e) for a in ALGORITHMS}
        grid_opt_cost = min(
            (results[a]["cost"] for a in ("UCS", "A*") if results[a]["found"]),
            default=None
        )

        for a, r in results.items():
            st = stats[a]
            st["ms"] += r["ms"]
            st["exp"] += r["expanded"]
            st["front"] += r["frontier"]
            st["cost"] += r["cost"] if r["cost"] is not None else 0.0
            st["wp"] += r["waypoints"] if r["waypoints"] is not None else 0
            st["sam"] += r["sam_exposure"]
            st["turns"] += r["sharp_turns"]
            st["ok"] += int(r["found"])

            if a in ("UCS", "A*"):
                st["opt"] += int(r["found"] and grid_opt_cost is not None and abs(r["cost"] - grid_opt_cost) < 1e-4)
            elif a == "Theta*":
                st["opt"] += int(r["found"] and grid_opt_cost is not None and r["cost"] <= grid_opt_cost + 1e-4)
            else:
                st["opt"] += int(r["found"] and grid_opt_cost is not None and abs(r["cost"] - grid_opt_cost) < 1e-4)

    print(f"\nMissile Guidance & Pathfinding Benchmark ({trials} trials, {rows}x{cols} airspace)\n")
    print(
        f"{'Algo':8}{'Time(ms)':>10}{'Expanded':>10}{'Frontier':>10}{'Cost':>8}"
        f"{'Waypts':>8}{'SAM Exp':>9}{'Turns':>8}{'Optimal %':>11}"
    )
    print("-" * 84)
    for a, st in stats.items():
        print(
            f"{a:8}{st['ms']/trials:10.2f}{st['exp']/trials:10.1f}{st['front']/trials:10.1f}"
            f"{st['cost']/trials:8.1f}{st['wp']/trials:8.1f}{st['sam']/trials:9.1f}"
            f"{st['turns']/trials:8.1f}{100*st['opt']/trials:10.0f}%"
        )


# ------------------------------------------------------------------------ GUI
def launch_gui():
    """Tactical Radar Command & Control GUI with Dual Path & Compare All dashboard."""
    import tkinter as tk
    from tkinter import ttk

    CELL = 28
    ROWS, COLS = 20, 24
    COLORS = {
        CLEAR: "#0f172a",      # Dark radar slate
        MOUNTAIN: "#475569",   # Terrain obstacle
        NFZ: "#991b1b",        # No-Fly Zone
        SAM_RADAR: "#b45309",  # SAM radar core (cost 6x)
        SAM_OUTER: "#78350f",  # SAM radar ring (cost 3x)
    }

    class App:
        def __init__(self, root):
            self.root = root
            root.title("AI Missile Guidance & Pathfinding")
            root.configure(bg="#090d16")

            self.grid, self.start, self.goal = random_airspace(ROWS, COLS, seed=1)
            self.anim = None
            self.overlay = {}
            self.raw_path = []
            self.smoothed_points = []
            self.flight_path_anim = []
            self.missile_pos = None
            self.missile_heading = (0, 1)

            self.brush = tk.StringVar(value="SAM Radar")
            self.algo = tk.StringVar(value="A*")
            self.speed = tk.IntVar(value=10)
            self.smooth_path_var = tk.BooleanVar(value=True)

            self._build_styles()
            self._build_controls()
            self._build_canvas()
            self.draw()

        def _build_styles(self):
            style = ttk.Style()
            style.theme_use("clam")
            style.configure("TFrame", background="#090d16")
            style.configure("TLabel", background="#090d16", foreground="#e2e8f0", font=("Consolas", 9))
            style.configure("Header.TLabel", background="#090d16", foreground="#38bdf8",
                            font=("Consolas", 10, "bold"))
            style.configure("TButton", font=("Consolas", 9, "bold"), padding=3)
            style.configure("TRadiobutton", background="#090d16", foreground="#cbd5e1", font=("Consolas", 9))
            style.configure("TCheckbutton", background="#090d16", foreground="#38bdf8", font=("Consolas", 9, "bold"))

        def _build_controls(self):
            bar = ttk.Frame(self.root)
            bar.pack(fill="x", padx=8, pady=(6, 3))

            ttk.Label(bar, text="Guidance:").pack(side="left")
            ttk.Combobox(bar, textvariable=self.algo, values=list(ALGORITHMS),
                         width=8, state="readonly").pack(side="left", padx=4)
            ttk.Button(bar, text="Launch", command=self.run_one).pack(side="left", padx=2)
            ttk.Button(bar, text="Compare All", command=self.compare).pack(side="left", padx=4)
            ttk.Button(bar, text="Random Airspace", command=self.randomize).pack(side="left", padx=2)
            ttk.Button(bar, text="Clear Map", command=self.clear).pack(side="left", padx=4)

            ttk.Checkbutton(bar, text="Smooth Flight Path", variable=self.smooth_path_var,
                            command=self.draw).pack(side="left", padx=8)

            ttk.Label(bar, text="Speed:").pack(side="left", padx=(6, 2))
            ttk.Scale(bar, from_=1, to=40, variable=self.speed, length=80).pack(side="left")

            bar2 = ttk.Frame(self.root)
            bar2.pack(fill="x", padx=8, pady=3)
            ttk.Label(bar2, text="Deploy:").pack(side="left")
            for b in ("Mountain", "No-Fly Zone", "SAM Radar", "Erase", "Launch", "Target"):
                ttk.Radiobutton(bar2, text=b, value=b, variable=self.brush).pack(side="left", padx=4)

        def _build_canvas(self):
            self.canvas = tk.Canvas(
                self.root, width=CELL * COLS, height=CELL * ROWS,
                bg="#0f172a", highlightthickness=1, highlightbackground="#334155"
            )
            self.canvas.pack(padx=8, pady=6)
            self.canvas.bind("<Button-1>", self.paint)
            self.canvas.bind("<B1-Motion>", self.paint)

            self.info = tk.StringVar(value="READY // Deploy airspace obstacles and press Launch (or Compare All).")
            ttk.Label(self.root, textvariable=self.info, style="Header.TLabel").pack(pady=(2, 2))

            legend = (
                "LEGEND:  ■ Slate=Mountain   ■ Red=No-Fly Zone   ■ Amber=SAM Core(6x)   "
                "■ Brown=SAM Ring(3x)   ┄ Dashed=Raw Waypoints   ━ Solid=Smoothed Flight"
            )
            ttk.Label(self.root, text=legend, foreground="#94a3b8").pack(pady=(0, 6))

        def draw(self):
            self.canvas.delete("all")
            for r, row in enumerate(self.grid):
                for c, v in enumerate(row):
                    fill = self.overlay.get((r, c), COLORS[v])
                    x0, y0 = c * CELL, r * CELL
                    x1, y1 = x0 + CELL, y0 + CELL
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#1e293b")

                    # Tactical radar icons
                    if (r, c) not in self.overlay:
                        if v == SAM_RADAR:
                            self.canvas.create_oval(x0 + 6, y0 + 6, x1 - 6, y1 - 6, outline="#f59e0b", width=1)
                        elif v == SAM_OUTER:
                            self.canvas.create_oval(x0 + 9, y0 + 9, x1 - 9, y1 - 9, outline="#d97706", width=1)
                        elif v == NFZ:
                            self.canvas.create_line(x0 + 4, y0 + 4, x1 - 4, y1 - 4, fill="#ef4444", width=1)
                            self.canvas.create_line(x1 - 4, y0 + 4, x0 + 4, y1 - 4, fill="#ef4444", width=1)

            # Dual Path Visualization
            if len(self.raw_path) >= 2:
                raw_coords = []
                for r, c in self.raw_path:
                    raw_coords.extend([c * CELL + CELL / 2, r * CELL + CELL / 2])

                if self.smooth_path_var.get() and len(self.smoothed_points) >= 2:
                    # Subtle dashed line for raw search waypoints
                    self.canvas.create_line(
                        *raw_coords, fill="#64748b", width=1.5, dash=(4, 4), capstyle="round"
                    )
                    for r, c in self.raw_path:
                        cx, cy = c * CELL + CELL / 2, r * CELL + CELL / 2
                        self.canvas.create_oval(cx - 2, cy - 2, cx + 2, cy + 2, fill="#94a3b8", outline="")

                    # Bright aerodynamic smoothed trajectory curve
                    sm_coords = []
                    for y, x in self.smoothed_points:
                        sm_coords.extend([x * CELL + CELL / 2, y * CELL + CELL / 2])
                    self.canvas.create_line(
                        *sm_coords, fill="#38bdf8", width=3, capstyle="round", joinstyle="round"
                    )
                else:
                    # Standard raw grid waypoints
                    self.canvas.create_line(
                        *raw_coords, fill="#38bdf8", width=3, capstyle="round", joinstyle="round"
                    )
                    for r, c in self.raw_path:
                        cx, cy = c * CELL + CELL / 2, r * CELL + CELL / 2
                        self.canvas.create_oval(cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5, fill="#e0f2fe", outline="")

            # Launch Site (L) and Target (T)
            for (r, c), txt, col, out_col in (
                (self.start, "L", "#10b981", "#ecfdf5"),
                (self.goal, "T", "#dc2626", "#fef2f2")
            ):
                self.canvas.create_oval(
                    c * CELL + 3, r * CELL + 3, (c + 1) * CELL - 3, (r + 1) * CELL - 3,
                    fill=col, outline=out_col, width=1.5
                )
                self.canvas.create_text(
                    c * CELL + CELL / 2, r * CELL + CELL / 2,
                    text=txt, fill="white", font=("Consolas", 10, "bold")
                )

            # Animated Missile Nose-Cone Sprite
            if self.missile_pos:
                my, mx = self.missile_pos
                cx, cy = mx * CELL + CELL / 2, my * CELL + CELL / 2
                dy, dx = self.missile_heading
                norm = math.hypot(dy, dx) or 1.0
                ux, uy = dx / norm, dy / norm
                px, py = -uy, ux

                nose = (cx + ux * 9, cy + uy * 9)
                left_fin = (cx - ux * 6 + px * 5, cy - uy * 6 + py * 5)
                right_fin = (cx - ux * 6 - px * 5, cy - uy * 6 - py * 5)
                self.canvas.create_polygon(
                    nose, left_fin, right_fin, fill="#facc15", outline="#fef08a", width=1
                )

        def stop(self):
            if self.anim:
                self.root.after_cancel(self.anim)
                self.anim = None
            self.overlay = {}
            self.raw_path = []
            self.smoothed_points = []
            self.flight_path_anim = []
            self.missile_pos = None

        def paint(self, e):
            r, c = e.y // CELL, e.x // CELL
            if not in_bounds(self.grid, r, c):
                return
            self.stop()
            b = self.brush.get()
            if b == "Launch" and (r, c) != self.goal and self.grid[r][c] not in IMPASSABLE:
                self.start = (r, c)
            elif b == "Target" and (r, c) != self.start and self.grid[r][c] not in IMPASSABLE:
                self.goal = (r, c)
            elif (r, c) not in (self.start, self.goal):
                mapping = {
                    "Mountain": MOUNTAIN,
                    "No-Fly Zone": NFZ,
                    "SAM Radar": SAM_RADAR,
                    "Erase": CLEAR,
                }
                if b in mapping:
                    self.grid[r][c] = mapping[b]
            self.draw()

        def randomize(self):
            self.stop()
            self.grid, self.start, self.goal = random_airspace(ROWS, COLS)
            self.draw()

        def clear(self):
            self.stop()
            self.grid = [[CLEAR] * COLS for _ in range(ROWS)]
            self.start, self.goal = (ROWS // 2, 1), (ROWS // 2, COLS - 2)
            self.draw()

        def run_one(self):
            self.stop()
            res = run(self.algo.get(), self.grid, self.start, self.goal)
            if not res["found"]:
                self.info.set(f"[{res['algo']}] NO FLIGHT CORRIDOR // Scanned {res['expanded']} sectors")
                self.animate(res)
                return

            self.info.set(
                f"[{res['algo']}] LOCKED | Cost: {res['cost']} | SAM Exp: {res['sam_exposure']} | "
                f"Turns: {res['sharp_turns']} | Scanned: {res['expanded']} | Time: {res['ms']:.2f}ms"
            )
            self.animate(res)

        def animate(self, res):
            order = res["order"]
            raw_path = res["path"] or []
            sm_path = res["smoothed_path"] or []
            self.raw_path = raw_path
            self.smoothed_points = sm_path

            # Determine flight animation trajectory points
            if self.smooth_path_var.get() and len(sm_path) >= 2:
                flight_points = sm_path
            else:
                flight_points = [(float(r), float(c)) for r, c in raw_path]

            i = 0

            def scan_airspace():
                nonlocal i
                for _ in range(self.speed.get()):
                    if i < len(order):
                        self.overlay[order[i]] = "#1e3a5f"
                        i += 1
                self.draw()
                if i < len(order):
                    self.anim = self.root.after(20, scan_airspace)
                else:
                    fly_missile(0)

            def fly_missile(k):
                if k < len(flight_points):
                    y, x = flight_points[k]
                    self.missile_pos = (y, x)
                    if k > 0:
                        py, px = flight_points[k - 1]
                        if math.hypot(y - py, x - px) > 1e-4:
                            self.missile_heading = (y - py, x - px)
                    self.draw()
                    self.anim = self.root.after(30, lambda: fly_missile(k + 1))
                else:
                    self.anim = None

            scan_airspace()

        def compare(self):
            """Launch Compare All dashboard with telemetry Treeview and interactive bar chart."""
            self.stop()
            self.draw()
            results = [run(a, self.grid, self.start, self.goal) for a in ALGORITHMS]
            best_cost = min((r["cost"] for r in results if r["found"]), default=None)

            win = tk.Toplevel(self.root)
            win.title("Guidance Telemetry & Performance Benchmark")
            win.configure(bg="#090d16")

            # Table of metrics
            cols = (
                "Algorithm", "Locked", "Waypoints", "Cost",
                "SAM Exp", "Sharp Turns", "Expanded", "Max Frontier", "Time (ms)", "Optimal?"
            )
            tv = ttk.Treeview(win, columns=cols, show="headings", height=len(results))
            for c in cols:
                tv.heading(c, text=c)
                tv.column(c, width=88, anchor="center")

            for r in results:
                is_opt = (
                    "yes" if (r["found"] and best_cost is not None and r["cost"] <= best_cost + 1e-4) else "no"
                )
                tv.insert("", "end", values=(
                    r["algo"],
                    "yes" if r["found"] else "no",
                    r["waypoints"] if r["found"] else "-",
                    r["cost"] if r["found"] else "-",
                    r["sam_exposure"] if r["found"] else "-",
                    r["sharp_turns"] if r["found"] else "-",
                    r["expanded"],
                    r["frontier"],
                    f"{r['ms']:.2f}",
                    is_opt,
                ))
            tv.pack(padx=10, pady=(10, 6))

            # Metric selector bar for chart
            metric_frame = ttk.Frame(win)
            metric_frame.pack(fill="x", padx=10, pady=(0, 4))
            ttk.Label(metric_frame, text="COMPARE METRIC:").pack(side="left", padx=(0, 6))

            metric_choice = tk.StringVar(value="Sectors Expanded (Efficiency)")
            chart_canvas = tk.Canvas(win, width=700, height=210, bg="#0f172a", highlightthickness=0)
            chart_canvas.pack(padx=10, pady=(0, 10))

            def draw_chart():
                chart_canvas.delete("all")
                selected = metric_choice.get()

                if selected == "Sectors Expanded (Efficiency)":
                    title = "Search Space Sectors Expanded (Lower = Faster Onboard Guidance)"
                    vals = [r["expanded"] for r in results]
                    val_strs = [str(v) for v in vals]
                    best_val = min(vals) if vals else 1
                elif selected == "Total Path Cost (Survivability)":
                    title = "Total Mission Flight Cost (Lower = Higher Survivability & Less Threat)"
                    vals = [r["cost"] if r["cost"] is not None else 0.0 for r in results]
                    val_strs = [f"{v:.1f}" if r["found"] else "N/A" for v, r in zip(vals, results)]
                    valid_costs = [v for v, r in zip(vals, results) if r["found"]]
                    best_val = min(valid_costs) if valid_costs else 1
                else:  # Execution Time (ms)
                    title = "Guidance Law Execution Time (ms) (Lower = Faster Decision Loop)"
                    vals = [r["ms"] for r in results]
                    val_strs = [f"{v:.2f}ms" for v in vals]
                    best_val = min(vals) if vals else 1

                chart_canvas.create_text(
                    350, 16, text=title, fill="#e2e8f0", font=("Consolas", 10, "bold")
                )

                max_val = max(vals) if vals and max(vals) > 0 else 1.0

                for k, r in enumerate(results):
                    v = vals[k]
                    x = 35 + k * 94
                    h = 125 * (v / max_val) if max_val > 0 else 0
                    is_best = (v == best_val and r["found"])

                    bar_col = "#38bdf8" if is_best else "#64748b"
                    chart_canvas.create_rectangle(x, 168 - h, x + 56, 168, fill=bar_col, outline="")
                    chart_canvas.create_text(
                        x + 28, 184, text=r["algo"], fill="#cbd5e1", font=("Consolas", 9, "bold")
                    )
                    chart_canvas.create_text(
                        x + 28, 156 - h, text=val_strs[k], fill="#f8fafc", font=("Consolas", 8)
                    )

            for opt in (
                "Sectors Expanded (Efficiency)",
                "Total Path Cost (Survivability)",
                "Execution Time (ms)",
            ):
                ttk.Radiobutton(
                    metric_frame, text=opt, value=opt,
                    variable=metric_choice, command=draw_chart
                ).pack(side="left", padx=6)

            draw_chart()

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        benchmark()
    else:
        try:
            launch_gui()
        except Exception as e:
            if "no display name and no $DISPLAY environment variable" in str(e) or "couldn't connect to display" in str(e):
                print("Notice: No graphical display detected. Running benchmark mode:")
                benchmark(trials=10)
            else:
                raise
