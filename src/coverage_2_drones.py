import itertools
from collections import deque

########################################################
# 1) Basic helpers: Manhattan distance, BFS path, TSP
########################################################

def manhattan_distance(cell_a, cell_b):
    return abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1])

def shortest_path(start_cell, end_cell):
    """
    Return a shortest path (list of cells) from start_cell to end_cell
    within a 3x3 grid. BFS is enough for a small grid.
    """
    def neighbors(x, y):
        for nx, ny in [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]:
            if 0 <= nx < 3 and 0 <= ny < 3:
                yield (nx, ny)
    
    queue = deque()
    queue.append((start_cell, [start_cell]))
    visited = set([start_cell])
    while queue:
        current, path = queue.popleft()
        if current == end_cell:
            return path
        for nb in neighbors(*current):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return [start_cell, end_cell]  # fallback

def build_distance_matrix(cells):
    n = len(cells)
    dist_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = manhattan_distance(cells[i], cells[j])
    return dist_matrix

def solve_tsp(cells, dist_matrix):
    """
    Brute force TSP for 9 cells in a 3x3, returning to start.
    Returns route (list of indices) plus total distance.
    """
    n = len(cells)
    start_index = 0
    best_route = None
    min_dist = float('inf')
    
    indices_to_permute = list(range(1, n))
    for perm in itertools.permutations(indices_to_permute):
        route = [start_index] + list(perm) + [start_index]
        route_dist = 0
        for i in range(len(route)-1):
            route_dist += dist_matrix[route[i]][route[i+1]]
        if route_dist < min_dist:
            min_dist = route_dist
            best_route = route
    
    return best_route, min_dist

########################################################
# 2) Single-Drone Sub-Route Scheduling
########################################################

def build_all_segmentations(num_edges, max_segment=6):
    """
    All ways to split 'num_edges' into segments each <= max_segment.
    E.g., if num_edges=5, possible splits: [5], [3,2], [2,3], [2,2,1], etc.
    """
    results = []
    def backtrack(remaining, current):
        if remaining == 0:
            results.append(current[:])
            return
        for seg in range(1, max_segment+1):
            if seg <= remaining:
                current.append(seg)
                backtrack(remaining - seg, current)
                current.pop()
    backtrack(num_edges, [])
    return results

def build_feasible_schedules_for_subroute(
    route_cells,
    start_edge_index,
    end_edge_index,
    max_time_slots=12,
    max_battery=6,
    recharge_time=1
):
    """
    Builds all feasible schedules for a single drone to cover 
    the TSP edges [start_edge_index .. end_edge_index - 1].
    
    Each schedule is a list of (x, y, battery).
    Drone starts at base station = route_cells[0], full battery.
    If sub-route is empty (start_edge_index==end_edge_index), 
      we return a trivial schedule that sits at base the whole time.
    """
    base_station = route_cells[0]  # should be (0,0)
    num_edges = end_edge_index - start_edge_index
    if num_edges <= 0:
        # Drone has no edges; it effectively does nothing but stay at base.
        single_state = [(base_station[0], base_station[1], max_battery)]
        return [single_state]

    # We want TSP cells from start_edge_index..end_edge_index
    subroute_cells = route_cells[start_edge_index : end_edge_index+1]

    segmentations = build_all_segmentations(num_edges, max_segment=max_battery)
    all_schedules = []

    def follow_segmentation(
        seg_idx,
        current_time,
        current_pos,
        battery,
        schedule,
        route_index,
        current_seg
    ):
        # If we've used all segments, sub-route is complete
        if seg_idx == len(current_seg):
            all_schedules.append(schedule[:])
            return

        segment_length = current_seg[seg_idx]

        # If not enough battery for the entire segment, we attempt a recharge detour
        if battery < segment_length:
            # path to base
            path_to_base = shortest_path(current_pos, base_station)
            path_back = shortest_path(base_station, current_pos)
            trip_time = (len(path_to_base)-1) + recharge_time + (len(path_back)-1)
            
            if current_time + trip_time > max_time_slots - 1:
                return
            
            states_added = 0
            tmp_time = current_time
            tmp_battery = battery

            # 1) Move to base
            for i in range(len(path_to_base)-1):
                if tmp_time >= max_time_slots-1:
                    break
                tmp_time += 1
                tmp_battery -= 1
                if tmp_battery < 0:  # NEW: battery < 0 => prune immediately
                    break
                new_cell = path_to_base[i+1]
                schedule.append((new_cell[0], new_cell[1], tmp_battery))
                states_added += 1
            
            # check if we ran out of battery
            if tmp_battery < 0:
                # backtrack
                for _ in range(states_added):
                    schedule.pop()
                return
            
            # 2) recharge
            if tmp_time < max_time_slots:
                tmp_time += recharge_time
                if tmp_time < max_time_slots:
                    tmp_battery = max_battery
                    schedule.append((base_station[0], base_station[1], tmp_battery))
                    states_added += 1
                else:
                    # backtrack
                    for _ in range(states_added):
                        schedule.pop()
                    return
            else:
                # no time to recharge fully
                for _ in range(states_added):
                    schedule.pop()
                return
            
            # 3) return from base
            for i in range(len(path_back)-1):
                if tmp_time >= max_time_slots-1:
                    break
                tmp_time += 1
                tmp_battery -= 1
                if tmp_battery < 0:  # NEW: battery < 0 => prune
                    break
                new_cell = path_back[i+1]
                schedule.append((new_cell[0], new_cell[1], tmp_battery))
                states_added += 1
            
            if tmp_battery < 0:
                # backtrack
                for _ in range(states_added):
                    schedule.pop()
                return
            
            # If we still have time, continue from that position
            if tmp_time < max_time_slots:
                last_pos = schedule[-1][0:2]
                follow_segmentation(
                    seg_idx,
                    tmp_time,
                    last_pos,
                    tmp_battery,
                    schedule,
                    route_index,
                    current_seg
                )
            
            # backtrack
            for _ in range(states_added):
                schedule.pop()
            return

        else:
            # We can do this segment with current battery
            tmp_time = current_time
            tmp_battery = battery
            tmp_schedule_len = len(schedule)
            tmp_pos = current_pos
            
            for edge_i in range(segment_length):
                if tmp_time >= max_time_slots-1:
                    break
                next_cell = subroute_cells[route_index + edge_i + 1]
                step_path = shortest_path(tmp_pos, next_cell)
                for step_idx in range(1, len(step_path)):
                    if tmp_time >= max_time_slots:
                        break
                    tmp_time += 1
                    tmp_battery -= 1
                    if tmp_battery < 0:  # NEW: battery < 0 => prune
                        break
                    scell = step_path[step_idx]
                    schedule.append((scell[0], scell[1], tmp_battery))
                # If battery < 0, we bail out entirely
                if tmp_battery < 0:
                    break
                tmp_pos = next_cell
            
            if tmp_battery >= 0 and tmp_time < max_time_slots:
                follow_segmentation(
                    seg_idx+1,
                    tmp_time,
                    tmp_pos,
                    tmp_battery,
                    schedule,
                    route_index + segment_length,
                    current_seg
                )
            
            # backtrack
            while len(schedule) > tmp_schedule_len:
                schedule.pop()

    # Build schedules by enumerating segmentations
    init_state = (base_station[0], base_station[1], max_battery)
    for seg in segmentations:
        schedule = [init_state]
        follow_segmentation(
            seg_idx=0,
            current_time=0,
            current_pos=base_station,
            battery=max_battery,
            schedule=schedule,
            route_index=0,
            current_seg=seg
        )

    return all_schedules


########################################################
# 3) Multi-Drone Coverage
########################################################

def get_cells_visited_in_slot(state_from, state_to):
    (x1, y1, _) = state_from
    (x2, y2, _) = state_to
    path = shortest_path((x1,y1), (x2,y2))
    return set(path)

def compute_union_coverage(drone_schedules, cells, max_time_slots=12):
    """
    Each element in 'drone_schedules' is a single-drone schedule 
    (list of states). We'll unify coverage across all drones, 
    then compute average coverage fraction.
    
    In time slot t, each drone d moves from schedule_d[t] to schedule_d[t+1], 
    marking intermediate cells visited. We union those sets across all drones.
    
    The actual # of time slots used is the max length of any drone schedule. 
    If that exceeds 'max_time_slots', we consider it infeasible for final selection 
    (or treat coverage = 0, or skip).
    """
    # maximum length across all drones
    max_len = max(len(sched) for sched in drone_schedules)
    if max_len > max_time_slots:
        # This schedule doesn't fit in 12 slots
        return 0.0, []
    
    visited_per_slot = []
    n_cells = len(cells)
    
    # We'll go up to max_time_slots - 1 transitions
    for t in range(max_time_slots - 1):
        # union of visited cells from all drones
        union_visited = set()
        for sched in drone_schedules:
            if t < len(sched) - 1:
                st_from = sched[t]
                st_to   = sched[t+1]
                visited = get_cells_visited_in_slot(st_from, st_to)
                union_visited = union_visited.union(visited)
            elif t < len(sched):
                # If there's no move (schedule ended or is at last state),
                # we can assume the drone stays put in the last cell
                st = sched[-1]
                union_visited.add((st[0], st[1]))
        visited_per_slot.append(len(union_visited)/n_cells)
    
    # final slot: each drone just sits
    union_visited = set()
    for sched in drone_schedules:
        if len(sched) > 0:
            last_st = sched[-1]
            union_visited.add((last_st[0], last_st[1]))
    visited_per_slot.append(len(union_visited)/n_cells)
    
    total_coverage = sum(visited_per_slot) / len(visited_per_slot)
    return total_coverage, visited_per_slot

########################################################
# 4) MAIN: 2-Drone Approach
########################################################

def main_two_drones():
    # 1) Setup cells, TSP
    cells = [
        (0,0), (0,1), (0,2),
        (1,0), (1,1), (1,2),
        (2,0), (2,1), (2,2)
    ]
    dist_matrix = build_distance_matrix(cells)
    tsp_route, tsp_dist = solve_tsp(cells, dist_matrix)
    print("TSP best route (indices):", tsp_route)
    print("TSP distance:", tsp_dist)
    print("Cell order:")
    for idx in tsp_route:
        print(cells[idx], end=" -> ")
    print("\n")

    # The TSP route has (len(tsp_route)-1) edges
    num_edges = len(tsp_route) - 1
    route_cells = [cells[idx] for idx in tsp_route]

    # 2) For each possible split of these 9 edges into two contiguous blocks,
    #    e.g. DroneA gets edges [0..k], DroneB gets edges [k+1..9].
    best_coverage = -1.0
    best_combo = None

    for split_point in range(num_edges+1):
        # Drone A covers edges [0..split_point)
        # Drone B covers edges [split_point.. num_edges)
        # example: if split_point=5, DroneA covers edges [0..4], DroneB covers [5..8]
        # "end_edge_index" is exclusive in build_feasible_schedules_for_subroute,
        # so we pass (split_point) not (split_point-1).
        
        # Build feasible schedules for Drone A
        A_schedules = build_feasible_schedules_for_subroute(
            route_cells,
            start_edge_index=0,
            end_edge_index=split_point,
            max_time_slots=12,
            max_battery=6,
            recharge_time=1
        )
        
        # Build feasible schedules for Drone B
        B_schedules = build_feasible_schedules_for_subroute(
            route_cells,
            start_edge_index=split_point,
            end_edge_index=num_edges,
            max_time_slots=12,
            max_battery=6,
            recharge_time=1
        )
        
        # 3) Combine each feasible A_schedules with each feasible B_schedule
        for Asched in A_schedules:
            for Bsched in B_schedules:
                coverage, _ = compute_union_coverage([Asched, Bsched], cells, max_time_slots=12)
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_combo = (Asched, Bsched, split_point)

    # 4) Print best result
    print(f"\n\nBEST 2-Drone Coverage = {best_coverage}")
    if best_combo is None:
        print("No feasible combination found.")
        return
    
    Asched, Bsched, sp = best_combo
    print(f"Split Point = {sp} (Drone A edges 0..{sp-1}, Drone B edges {sp}..{num_edges-1})")
    print("\nDrone A schedule:")
    for t, st in enumerate(Asched):
        x,y,b = st
        print(f"  t={t}: Cell=({x},{y}), Battery={b}")
    print("\nDrone B schedule:")
    for t, st in enumerate(Bsched):
        x,y,b = st
        print(f"  t={t}: Cell=({x},{y}), Battery={b}")

    # Optional: coverage by timeslot
    coverage, cov_by_slot = compute_union_coverage([Asched, Bsched], cells, 12)
    print("\nCoverage by slot:")
    for t, cfrac in enumerate(cov_by_slot):
        print(f"  Slot {t}: {cfrac:.2f}")

if __name__=="__main__":
    main_two_drones()
