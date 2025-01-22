import itertools

###############################
# 1) TSP SOLUTION (unchanged)
###############################
def manhattan_distance(cell_a, cell_b):
    return abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1])

def build_distance_matrix(cells):
    n = len(cells)
    dist_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = manhattan_distance(cells[i], cells[j])
    return dist_matrix

def solve_tsp(cells, dist_matrix):
    """
    Brute-force TSP to find minimal route visiting all cells exactly once
    and returning to start. For a 3x3=9 cell problem, brute force is still feasible.
    """
    n = len(cells)
    start_index = 0  # base station at cells[0]
    best_route = None
    min_dist = float('inf')
    
    indices_to_permute = list(range(1, n))  # skip 0 for permutations
    for perm in itertools.permutations(indices_to_permute):
        route = [start_index] + list(perm) + [start_index]
        route_dist = 0
        for i in range(len(route) - 1):
            route_dist += dist_matrix[route[i]][route[i+1]]
        if route_dist < min_dist:
            min_dist = route_dist
            best_route = route
    
    return best_route, min_dist

###############################
# 2) BUILD SCHEDULES ALONG THE TSP PATH
###############################
def shortest_path(base, start_cell, end_cell):
    """
    Return a list of cells (including start, end) for a shortest path
    between 'start_cell' and 'end_cell' by Manhattan distance.
    For a small 3x3 grid, a BFS is straightforward.
    """
    from collections import deque
    
    def neighbors(x, y):
        for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
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
    
    # Should not happen in a 3x3 if within bounds
    return [start_cell, end_cell]

def build_all_segmentations(num_edges, max_segment=6):
    """
    Suppose the TSP route has `num_edges`. We want all ways to split
    those edges into segments, each of length <= max_segment (battery limit=6).
    E.g. if num_edges=9, possible splits: [6,3], [5,4], [3,3,3], etc.
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

def build_schedules_from_tsp(
    cells, tsp_route, 
    dist_matrix, 
    max_time_slots=12, 
    max_battery=6, 
    recharge_time=1
):
    """
    Build schedules by:
      1. Breaking the TSP route (with num_edges) into segments 
         that each fit 'max_battery' moves.
      2. For each segment, the drone flies that many edges along TSP in order.
      3. If there's another segment after that, the drone may:
         - continue if it has battery
         - or detour back to base, recharge, then return
      4. We prune if the total schedule length > max_time_slots.

    Returns all feasible schedules that complete the TSP route within 12 slots.
    """
    base_station = cells[0]  # (0,0)
    # TSP route has len(tsp_route)-1 edges
    num_edges = len(tsp_route) - 1
    # Precompute TSP cells for each index
    route_cells = [cells[idx] for idx in tsp_route]
    
    segmentations = build_all_segmentations(num_edges, max_battery)
    all_schedules = []
    
    def follow_segmentation(seg_idx, current_time, current_pos, battery, schedule, route_index):
        """
        seg_idx: which segment index we are executing
        current_time: current time slot
        current_pos: current (x,y)
        battery: current battery
        schedule: list[(x, y, battery)]
        route_index: which edge of TSP route we are about to traverse
        """
        # If we've used up all segments, TSP is complete
        if seg_idx == len(current_seg):
            all_schedules.append(schedule[:])
            return
        
        segment_length = current_seg[seg_idx]
        
        # If not enough battery to fly 'segment_length' edges, we must detour & recharge
        if battery < segment_length:
            # Path to base
            path_to_base = shortest_path(base_station, current_pos, base_station)
            # Path back from base
            path_back = shortest_path(base_station, base_station, current_pos)
            
            trip_time = (len(path_to_base)-1) + recharge_time + (len(path_back)-1)
            
            # Check if adding that detour still leaves us time to finish
            if current_time + trip_time > max_time_slots - 1:
                # Not feasible, prune
                return
            
            # 1) Go to base
            states_added = 0
            tmp_time = current_time
            tmp_battery = battery
            # Move along path_to_base
            for i in range(len(path_to_base)-1):
                if tmp_time >= max_time_slots-1:
                    break
                tmp_time += 1
                tmp_battery -= 1
                new_cell = path_to_base[i+1]
                schedule.append((new_cell[0], new_cell[1], tmp_battery))
                states_added += 1
            
            # 2) Recharge (1 time slot, no movement)
            if tmp_time < max_time_slots:
                tmp_time += recharge_time
                tmp_battery = max_battery
                # add 1 schedule state for the recharge slot
                schedule.append((base_station[0], base_station[1], tmp_battery))
                states_added += 1
            else:
                # not enough time, backtrack
                for _ in range(states_added):
                    schedule.pop()
                return
            
            # 3) Return to current_pos from base
            for i in range(len(path_back)-1):
                if tmp_time >= max_time_slots-1:
                    break
                tmp_time += 1
                tmp_battery -= 1
                new_cell = path_back[i+1]
                schedule.append((new_cell[0], new_cell[1], tmp_battery))
                states_added += 1
            
            # If we have time left after returning to current_pos:
            if tmp_time < max_time_slots:
                # Now we are back at current_pos w/ battery fully charged
                # But we have to re-assign current_pos properly:
                # The final position from path_back is indeed 'current_pos'
                last_cell = schedule[-1][0:2]  
                follow_segmentation(
                    seg_idx, 
                    tmp_time, 
                    last_cell, 
                    tmp_battery, 
                    schedule, 
                    route_index
                )
            
            # Backtrack
            for _ in range(states_added):
                schedule.pop()
            return
        else:
            # We can do this segment directly
            tmp_time = current_time
            tmp_battery = battery
            tmp_schedule_len = len(schedule)
            tmp_pos = current_pos
            
            for edge_i in range(segment_length):
                if tmp_time >= max_time_slots-1:
                    break
                # Next cell along TSP route
                next_cell = route_cells[route_index + edge_i + 1]
                
                # Move step by step from tmp_pos to next_cell
                step_path = shortest_path(base_station, tmp_pos, next_cell)
                
                # The BFS returns a path that includes tmp_pos as [0],
                # so we iterate from index 1 to end to "move" each step:
                for step_idx in range(1, len(step_path)):
                    if tmp_time >= max_time_slots:
                        break
                    tmp_time += 1
                    tmp_battery -= 1
                    step_cell = step_path[step_idx]
                    schedule.append((step_cell[0], step_cell[1], tmp_battery))
                tmp_pos = next_cell
            
            # If we haven't run out of time:
            if tmp_time < max_time_slots:
                follow_segmentation(
                    seg_idx+1, 
                    tmp_time, 
                    tmp_pos, 
                    tmp_battery, 
                    schedule, 
                    route_index+segment_length
                )
            
            # Backtrack: remove anything we appended
            while len(schedule) > tmp_schedule_len:
                schedule.pop()

    # Build schedules by enumerating possible segmentations
    for current_seg in segmentations:
        init_state = (base_station[0], base_station[1], max_battery)
        schedule = [init_state]
        follow_segmentation(
            seg_idx=0,
            current_time=0,
            current_pos=base_station,
            battery=max_battery,
            schedule=schedule,
            route_index=0
        )

    return all_schedules

###############################
# 3) REFINED COVERAGE CALCULATION
###############################
def get_cells_visited_in_slot(state_from, state_to):
    """
    Return set of cells visited when drone goes from 'state_from' to 'state_to'.
    That includes start, end, and any intermediate cells if dist > 1.
    """
    (x1, y1, _) = state_from
    (x2, y2, _) = state_to
    dist = manhattan_distance((x1,y1), (x2,y2))
    if dist == 0:
        return {(x1, y1)}
    
    path = shortest_path(None, (x1, y1), (x2, y2))
    return set(path)

def compute_coverage(schedule, cells, max_time_slots=12):
    """
    For each time slot t:
      - If schedule[t] -> schedule[t+1] exists, 
        mark all intermediate cells as visited.
      - coverage_in_slot = (visited_count) / total_cells (9)

    Return average coverage over the 12 slots, plus slot-by-slot coverage.
    """
    visited_per_slot = []
    n_cells = len(cells)
    
    for t in range(max_time_slots - 1):
        if t >= len(schedule) - 1:
            # If schedule is shorter, assume last position remains
            visited_per_slot.append(1 / n_cells)
            continue
        
        state_from = schedule[t]
        state_to   = schedule[t+1]
        visited_set = get_cells_visited_in_slot(state_from, state_to)
        visited_per_slot.append(len(visited_set)/n_cells)
    
    # Final slot
    if len(schedule) > 0:
        visited_per_slot.append(1.0/n_cells)
    else:
        visited_per_slot.append(0.0)

    total_coverage = sum(visited_per_slot) / len(visited_per_slot)
    return total_coverage, visited_per_slot

###############################
# 4) PICK THE BEST SCHEDULE
###############################
def select_best_schedule(all_schedules, cells, max_time_slots=12):
    best_schedule = None
    best_coverage = -1.0
    for schedule in all_schedules:
        coverage, _ = compute_coverage(schedule, cells, max_time_slots)
        if coverage > best_coverage:
            best_coverage = coverage
            best_schedule = schedule
    return best_schedule, best_coverage

###############################
# MAIN
###############################
def main():
    cells = [
        (0,0), (0,1), (0,2),
        (1,0), (1,1), (1,2),
        (2,0), (2,1), (2,2)
    ]
    
    # 1) TSP
    dist_matrix = build_distance_matrix(cells)
    tsp_route, tsp_dist = solve_tsp(cells, dist_matrix)

    print("TSP best route (by cell indices):", tsp_route)
    print("TSP minimum total distance:", tsp_dist)
    print("Ordered cells in TSP route:")
    for idx in tsp_route:
        print(cells[idx], end=" -> ")
    print("\n")

    # 2) Build schedules from the TSP route
    schedules = build_schedules_from_tsp(
        cells, tsp_route, dist_matrix,
        max_time_slots=12,
        max_battery=6,
        recharge_time=1
    )
    print(f"Number of feasible TSP-based schedules generated: {len(schedules)}")

    # 3) Coverage + Selection
    best_schedule, best_coverage = select_best_schedule(
        schedules, cells, max_time_slots=12
    )

    print("Best schedule coverage:", best_coverage)
    if not best_schedule:
        print("No feasible schedule found.")
        return

    print("Best schedule detail:")
    for t, state in enumerate(best_schedule):
        x, y, b = state
        print(f"  Time {t}: Cell=({x},{y}), Battery={b}")

    _, coverage_by_slot = compute_coverage(best_schedule, cells, max_time_slots=12)
    print("\nCoverage by time slot:")
    for t, cfrac in enumerate(coverage_by_slot):
        print(f"  Time {t}: coverage fraction = {cfrac:.2f}")

if __name__ == "__main__":
    main()
