#!/usr/bin/env python3
"""
main_rolling_horizon.py
Rolling-horizon MIP + dynamic coverage reversion.
Writes state data to 'states.json' so a separate script can visualize.

Requirements:
  pip install ortools
"""

import json
from ortools.sat.python import cp_model

def cell_index(x, y, grid_size=3):
    return x*grid_size + y

def in_bounds(x, y, grid_size=3):
    return 0 <= x < grid_size and 0 <= y < grid_size

def build_adjacency(grid_size=3):
    adjacency = [[] for _ in range(grid_size*grid_size)]
    for x in range(grid_size):
        for y in range(grid_size):
            c = cell_index(x,y,grid_size)
            for dx, dy in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]:
                nx, ny = x+dx, y+dy
                if in_bounds(nx, ny, grid_size):
                    adjacency[c].append(cell_index(nx, ny, grid_size))
    return adjacency

def move_cost(i, j, adjacency):
    if i == j:
        return 0
    return 1 if j in adjacency[i] else 999999

def main():
    # Problem parameters
    GRID_SIZE = 5
    NUM_DRONES = 3
    TOTAL_TIME = 60
    HORIZON = 5
    MAX_BATTERY = 30
    BASE_CELL = 0
    REVERSION_THRESHOLD = 2
    
    adjacency = build_adjacency(GRID_SIZE)
    num_cells = GRID_SIZE*GRID_SIZE

    # Drone states
    drones = [{"pos": BASE_CELL, "battery": MAX_BATTERY} for _ in range(NUM_DRONES)]

    # Dynamic coverage
    since_visited = [0]*num_cells
    uncertainty   = [1]*num_cells  # start all at 1

    # We'll store states in a list, then write to JSON at the end (or incrementally).
    all_states = []

    for real_t in range(TOTAL_TIME):
        # Build short-horizon MIP
        model = cp_model.CpModel()
        
        x = {}
        battery_var = {}
        coverage_var = {}

        for d in range(NUM_DRONES):
            for t in range(HORIZON):
                battery_var[(d,t)] = model.NewIntVar(0, MAX_BATTERY, f"battery_d{d}_t{t}")

        for d in range(NUM_DRONES):
            for i in range(num_cells):
                for t in range(HORIZON):
                    x[(d,i,t)] = model.NewBoolVar(f"x_d{d}_c{i}_t{t}")

        for i in range(num_cells):
            for t in range(HORIZON):
                coverage_var[(i,t)] = model.NewBoolVar(f"cov_c{i}_t{t}")

        # Init conditions at t=0
        for d in range(NUM_DRONES):
            init_pos = drones[d]["pos"]
            init_bat = drones[d]["battery"]
            model.Add(x[(d,init_pos,0)] == 1)
            for i in range(num_cells):
                if i != init_pos:
                    model.Add(x[(d,i,0)] == 0)
            model.Add(battery_var[(d,0)] == init_bat)

        # Exactly 1 cell per drone/time
        for d in range(NUM_DRONES):
            for t in range(HORIZON):
                model.Add(sum(x[(d,i,t)] for i in range(num_cells)) == 1)

        # Adjacency + Battery
        for d in range(NUM_DRONES):
            for t in range(HORIZON-1):
                bt  = battery_var[(d,t)]
                btn = battery_var[(d,t+1)]
                for i in range(num_cells):
                    for j in range(num_cells):
                        if j not in adjacency[i]:
                            model.Add(x[(d,i,t)] + x[(d,j,t+1)] <= 1)
                        else:
                            cost_ij = move_cost(i,j, adjacency)
                            move_bool = model.NewBoolVar(f"move_{d}_{i}->{j}_{t}")
                            model.Add(move_bool == 1).OnlyEnforceIf([x[(d,i,t)], x[(d,j,t+1)]])
                            model.Add(move_bool == 0).OnlyEnforceIf(x[(d,i,t)].Not())
                            model.Add(move_bool == 0).OnlyEnforceIf(x[(d,j,t+1)].Not())

                            M = MAX_BATTERY + 10
                            if j == BASE_CELL:
                                # recharge
                                model.Add(btn >= MAX_BATTERY - M*(1-move_bool))
                                model.Add(btn <= MAX_BATTERY + M*(1-move_bool))
                            else:
                                model.Add(btn >= bt - cost_ij - M*(1-move_bool))
                                model.Add(btn <= bt - cost_ij + M*(1-move_bool))

        # Battery range
        for d in range(NUM_DRONES):
            for t in range(HORIZON):
                model.Add(battery_var[(d,t)] >= 0)
                model.Add(battery_var[(d,t)] <= MAX_BATTERY)

        # Coverage constraints
        for i in range(num_cells):
            for t in range(HORIZON):
                for d in range(NUM_DRONES):
                    model.Add(coverage_var[(i,t)] >= x[(d,i,t)])
                model.Add(coverage_var[(i,t)] <= sum(x[(d,i,t)] for d in range(NUM_DRONES)))

        # Objective: coverage - travel
        coverage_terms = []
        travel_terms   = []
        lam = 1

        for t in range(HORIZON):
            for i in range(num_cells):
                coverage_terms.append(uncertainty[i]*coverage_var[(i,t)])
        for d in range(NUM_DRONES):
            for t in range(HORIZON-1):
                for i in range(num_cells):
                    for j in adjacency[i]:
                        c_ij = move_cost(i,j, adjacency)
                        if c_ij>0:
                            mb = model.NewBoolVar(f"cost_{d}_{i}->{j}_{t}")
                            model.Add(mb == 1).OnlyEnforceIf([x[(d,i,t)], x[(d,j,t+1)]])
                            model.Add(mb == 0).OnlyEnforceIf(x[(d,i,t)].Not())
                            model.Add(mb == 0).OnlyEnforceIf(x[(d,j,t+1)].Not())
                            travel_terms.append(c_ij*mb)

        model.Maximize(sum(coverage_terms) - lam*sum(travel_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # No feasible -> do nothing
            pass
        else:
            # Execute local time=0->1
            visited_cells = []
            for d in range(NUM_DRONES):
                # find new position
                old_pos = drones[d]["pos"]
                new_pos = None
                for i in range(num_cells):
                    if solver.Value(x[(d,i,1)])==1:
                        new_pos = i
                new_bat = solver.Value(battery_var[(d,1)])
                drones[d]["pos"] = new_pos
                drones[d]["battery"] = new_bat
                visited_cells.append(new_pos)

            # coverage at local t=1
            cov_cells = [i for i in range(num_cells) if solver.Value(coverage_var[(i,1)])==1]
            # dynamic coverage update
            for i in range(num_cells):
                if i in cov_cells:
                    since_visited[i] = 0
                    uncertainty[i]   = 0
                else:
                    since_visited[i]+=1
                    if since_visited[i]>=REVERSION_THRESHOLD:
                        uncertainty[i] = 1

        # record the current state
        # note we store after we move
        # store all info needed for later visualization
        state_dict = {
            "time": real_t,
            "drones": [],
            "uncertainty": uncertainty[:]
        }
        for d in range(NUM_DRONES):
            state_dict["drones"].append({
                "pos": drones[d]["pos"],
                "battery": drones[d]["battery"]
            })
        all_states.append(state_dict)

    # Write states to JSON
    with open("states.json","w") as f:
        json.dump(all_states, f, indent=2)
    print("Done. Wrote states to 'states.json'.")

if __name__=="__main__":
    main()
