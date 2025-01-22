import random
from collections import deque

###############################
# Parameters
###############################
GRID_SIZE = 3
TIME_SLOTS = 12
HORIZON = 2          # We'll look ahead 2 steps each time we plan
MAX_BATTERY = 6
RECHARGE_TIME = 1    # Not fully demonstrated in short horizon, but included for completeness
REVERSION_PROB = 0.1 # Probability that a cell reverts to uncertain each step


###############################
# Environment / Grid Setup
###############################
# We'll represent each cell by (x, y), 0 <= x < GRID_SIZE, 0 <= y < GRID_SIZE
# "uncertainty[x][y]" is a float in [0,1], 1 => completely uncertain, 0 => certain.

uncertainty = [[1.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

def in_bounds(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def neighbors(x, y):
    """Return valid adjacent cells (including stay-put if you want)."""
    deltas = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
    for dx, dy in deltas:
        nx, ny = x+dx, y+dy
        if in_bounds(nx, ny):
            yield (nx, ny)


###############################
# BFS shortest path on a 3x3 grid
###############################
def shortest_path(start_cell, end_cell):
    """
    Return a shortest path (list of cells) from start_cell to end_cell
    within a 3x3 grid. BFS is enough for a small grid.
    """
    from collections import deque
    
    def valid_neighbors(cx, cy):
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                yield (nx, ny)
    
    if start_cell == end_cell:
        return [start_cell]
    
    queue = deque()
    queue.append((start_cell, [start_cell]))
    visited = set([start_cell])
    while queue:
        (curx, cury), path = queue.popleft()
        if (curx, cury) == end_cell:
            return path
        for nb in valid_neighbors(curx, cury):
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return [start_cell, end_cell]  # fallback, shouldn't happen in 3x3 if in-bounds


###############################
# Drone State
###############################
# We'll store each drone state as a dict:
#   {
#       "pos": (x, y),
#       "battery": int,
#       "recharging": bool,
#   }
# Both drones start at base station (0,0) at full battery.

droneA = {"pos": (0,0), "battery": MAX_BATTERY, "recharging": False}
droneB = {"pos": (0,0), "battery": MAX_BATTERY, "recharging": False}
base_station = (0,0)


###############################
# Rolling-Horizon Planner
###############################
def plan_next_moves(drones, horizon=3):
    """
    Generate a short-horizon plan (up to 'horizon' steps) for both drones,
    picking the best combination of moves that reduces uncertainty.
    """
    droneA_state, droneB_state = drones
    feasible_plans_A = generate_feasible_plans_for_drone(droneA_state, horizon)
    feasible_plans_B = generate_feasible_plans_for_drone(droneB_state, horizon)
    
    best_plan = None
    best_coverage_gain = -1
    
    for planA in feasible_plans_A:
        for planB in feasible_plans_B:
            coverage_gain = simulate_plan_coverage_gain(droneA_state, droneB_state, planA, planB)
            if coverage_gain > best_coverage_gain:
                best_coverage_gain = coverage_gain
                best_plan = (planA, planB)
    
    return best_plan

def generate_feasible_plans_for_drone(drone_state, horizon):
    """
    For a single drone, enumerate all possible 'horizon'-length sequences of moves
    that respect battery constraints (but ignoring recharging for brevity).
    We allow movement to adjacent cells if battery>0, or staying put if battery=0.
    
    Returns a list of feasible move sequences, each is a list[(x, y), ...].
    """
    plans = []
    
    def backtrack(steps_so_far, pos, battery):
        if len(steps_so_far) == horizon:
            plans.append(steps_so_far[:])
            return
        
        if battery > 0:
            # can move or stay
            for (nx, ny) in neighbors(*pos):
                steps_so_far.append((nx, ny))
                new_battery = battery - 1 if (nx, ny) != pos else battery
                backtrack(steps_so_far, (nx, ny), new_battery)
                steps_so_far.pop()
        else:
            # battery == 0 => must stay
            steps_so_far.append(pos)
            backtrack(steps_so_far, pos, battery)
            steps_so_far.pop()
    
    backtrack([], drone_state["pos"], drone_state["battery"])
    return plans

def simulate_plan_coverage_gain(droneA_state, droneB_state, planA, planB):
    """
    Simulate how much coverage (uncertainty reduction) we get if Drone A follows planA
    and Drone B follows planB over the horizon. 
    We do not permanently modify 'uncertainty', just measure potential coverage gain.
    """
    global uncertainty
    # Make a local copy of the uncertainty map
    original_uncert = [row[:] for row in uncertainty]
    
    total_reduction = 0.0
    
    # Copy states so we don't overwrite the real drone states
    dApos = droneA_state["pos"]
    dA_battery = droneA_state["battery"]
    dBpos = droneB_state["pos"]
    dB_battery = droneB_state["battery"]
    
    # We'll assume planA and planB have the same length = horizon 
    # (since we enumerated that way)
    for t in range(len(planA)):
        pathA = shortest_path(dApos, planA[t])
        pathB = shortest_path(dBpos, planB[t])
        
        visited_cells = set(pathA).union(set(pathB))
        
        # reduce uncertainty
        reduction_here = 0.0
        for (cx, cy) in visited_cells:
            if original_uncert[cx][cy] > 0:
                reduction_here += original_uncert[cx][cy]
                original_uncert[cx][cy] = 0.0
        
        total_reduction += reduction_here
        
        # Update battery
        stepsA = max(len(pathA)-1, 0)
        stepsB = max(len(pathB)-1, 0)
        dA_battery -= stepsA
        dB_battery -= stepsB
        
        if dA_battery < 0 or dB_battery < 0:
            # invalid plan
            return -9999
        
        dApos = planA[t]
        dBpos = planB[t]
    
    return total_reduction


###############################
# Reversion Logic
###############################
def apply_reversion():
    """
    Each time step, certain cells may revert to uncertainty 
    with probability REVERSION_PROB.
    """
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if uncertainty[x][y] == 0.0:
                if random.random() < REVERSION_PROB:
                    # revert to uncertain
                    uncertainty[x][y] = 1.0


###############################
# Execute 1-step of plan
###############################
def step_execute_plan(droneA_state, droneB_state, planA_step, planB_step):
    """
    Execute one step from the chosen plan for each drone:
      - Move them from current pos to next pos
      - Decrease battery
      - Update global 'uncertainty'
      - Instantly recharge if at base (toy logic)
    """
    pathA = shortest_path(droneA_state["pos"], planA_step)
    pathB = shortest_path(droneB_state["pos"], planB_step)
    
    visited = set(pathA).union(set(pathB))
    for (vx, vy) in visited:
        uncertainty[vx][vy] = 0.0
    
    stepsA = max(len(pathA)-1, 0)
    stepsB = max(len(pathB)-1, 0)
    droneA_state["battery"] -= stepsA
    droneB_state["battery"] -= stepsB
    
    if droneA_state["battery"] < 0 or droneB_state["battery"] < 0:
        print("Warning: negative battery - plan was invalid!")
    
    droneA_state["pos"] = planA_step
    droneB_state["pos"] = planB_step
    
    # Instant recharge if at base
    if droneA_state["pos"] == base_station:
        droneA_state["battery"] = MAX_BATTERY
    if droneB_state["pos"] == base_station:
        droneB_state["battery"] = MAX_BATTERY


###############################
# Main Rolling-Horizon Loop
###############################
def main():
    global uncertainty
    
    # Seed for reproducible behavior
    random.seed(0)
    
    # Reset the uncertainty map
    uncertainty = [[1.0]*GRID_SIZE for _ in range(GRID_SIZE)]
    
    # Initialize drones
    A = {"pos": (0,0), "battery": MAX_BATTERY, "recharging": False}
    B = {"pos": (0,0), "battery": MAX_BATTERY, "recharging": False}
    
    print("Initial Uncertainty:")
    for row in uncertainty:
        print(row)
    print("")
    
    time_slot = 0
    while time_slot < TIME_SLOTS:
        # 1) Plan for the next 'HORIZON' steps
        plan = plan_next_moves([A, B], horizon=HORIZON)
        if plan is None:
            print("No feasible plan found (both drones might be out of battery).")
            break
        
        planA, planB = plan  # each is a list of positions for that horizon
        
        # 2) Execute the first step of that plan
        step_execute_plan(A, B, planA[0], planB[0])
        
        # 3) Possibly apply reversion
        apply_reversion()
        
        # 4) Print status
        print(f"After time slot {time_slot}:")
        print(f"  Drone A: pos={A['pos']}, battery={A['battery']}")
        print(f"  Drone B: pos={B['pos']}, battery={B['battery']}")
        print("  Uncertainty grid:")
        for row in uncertainty:
            print(["{:.1f}".format(v) for v in row])
        print("")
        
        time_slot += 1
    
    # Final coverage result
    total_uncert = sum(sum(row) for row in uncertainty)
    print("Final total uncertainty:", total_uncert)
    coverage_fraction = 1.0 - (total_uncert / (GRID_SIZE*GRID_SIZE))
    print(f"Approx coverage fraction = {coverage_fraction:.2f}")


if __name__ == "__main__":
    main()
