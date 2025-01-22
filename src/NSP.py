from ortools.sat.python import cp_model

# Initialize model
model = cp_model.CpModel()

# Define parameters
drones = ["Drone1", "Drone2", "Drone3", "Drone4", "Drone5", "Drone6", "Drone7"]
zones = ["ZoneA", "ZoneB", "ZoneC"]
time_slots = ["T1", "T2", "T3", "T4", "T5"]
max_time_slots_per_drone = 3
battery_life = 2  # Max consecutive time slots before recharging
priority_zones = {"ZoneA": 2, "ZoneB": 1}  # ZoneA requires monitoring by 2 drones per time slot

# Create variables
monitoring_vars = {}
for drone in range(len(drones)):
    for time_slot in range(len(time_slots)):
        for zone in range(len(zones)):
            monitoring_vars[(drone, time_slot, zone)] = model.NewBoolVar(f'drone_{drone}_time_{time_slot}_zone_{zone}')

# Constraints
# 1. Each zone must be monitored in each time slot
for time_slot in range(len(time_slots)):
    for zone in range(len(zones)):
        required_drones = priority_zones.get(zones[zone], 1)  # Default to 1 if no priority set
        model.Add(sum(monitoring_vars[(drone, time_slot, zone)] for drone in range(len(drones))) >= required_drones)

# 2. Each drone works at most max_time_slots_per_drone
for drone in range(len(drones)):
    model.Add(sum(monitoring_vars[(drone, time_slot, zone)] for time_slot in range(len(time_slots)) for zone in range(len(zones))) <= max_time_slots_per_drone)

# 3. Each drone monitors only one zone per time slot
for drone in range(len(drones)):
    for time_slot in range(len(time_slots)):
        model.Add(sum(monitoring_vars[(drone, time_slot, zone)] for zone in range(len(zones))) <= 1)

# 4. Battery life constraint: Recharge after consecutive time slots
for drone in range(len(drones)):
    for time_slot in range(len(time_slots) - battery_life):
        model.Add(
            sum(monitoring_vars[(drone, time_slot + t, zone)] for t in range(battery_life) for zone in range(len(zones)))
            <= battery_life
        )

# 5. Fair distribution of monitoring (optional soft constraint)
total_slots = len(time_slots) * len(zones)
fair_slots_per_drone = total_slots // len(drones)
for drone in range(len(drones)):
    model.Add(sum(monitoring_vars[(drone, time_slot, zone)] for time_slot in range(len(time_slots)) for zone in range(len(zones))) >= fair_slots_per_drone)

# Solve the model
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Display results
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Drone Monitoring Schedule:")
    for time_slot in range(len(time_slots)):
        print(f"\n{time_slots[time_slot]}:")
        for zone in range(len(zones)):
            assigned = False
            for drone in range(len(drones)):
                if solver.Value(monitoring_vars[(drone, time_slot, zone)]) == 1:
                    print(f"  {zones[zone]}: {drones[drone]}")
                    assigned = True
            if not assigned:
                print(f"  {zones[zone]}: No drone assigned!")
else:
    print("No feasible solution found.")
