from ortools.sat.python import cp_model

# Define parameters
drones = ["Drone1", "Drone2", "Drone3", "Drone4", "Drone5", "Drone6", "Drone7"]
zones = ["ZoneA", "ZoneB", "ZoneC"]
time_slots = ["T1", "T2", "T3", "T4", "T5"]
max_time_slots_per_drone = 3
battery_life = 2
priority_zones = {"ZoneA": 2, "ZoneB": 1}  # ZoneA requires more frequent monitoring

# Values (benefits) for monitoring zones (arbitrary weights for demonstration)
zone_values = {"ZoneA": 10, "ZoneB": 8, "ZoneC": 6}

# Create variables and data structures
model = cp_model.CpModel()
values = []
weights = []
variables = []

for drone in range(len(drones)):
    for time_slot in range(len(time_slots)):
        for zone in range(len(zones)):
            # Assign a value based on the zone's priority
            values.append(zone_values[zones[zone]])
            weights.append(1)  # Each task uses 1 time slot
            # Create a decision variable for each assignment
            variables.append(model.NewBoolVar(f"drone_{drone}_ts_{time_slot}_zone_{zone}"))

# Total capacity (maximum time slots across all drones)
capacity = max_time_slots_per_drone * len(drones)

# Objective: Maximize the total value of zone monitoring
model.Maximize(
    sum(variables[i] * values[i] for i in range(len(values)))
)

# Constraint: Total time slots used should not exceed capacity
model.Add(
    sum(variables[i] * weights[i] for i in range(len(weights))) <= capacity
)

# Constraint: A drone can monitor only one zone per time slot
for drone in range(len(drones)):
    for time_slot in range(len(time_slots)):
        model.Add(
            sum(
                variables[i]
                for i in range(len(values))
                if i // (len(time_slots) * len(zones)) == drone and (i // len(zones)) % len(time_slots) == time_slot
            ) <= 1
        )

# Priority constraints: Zones like ZoneA need more frequent monitoring
for time_slot in range(len(time_slots)):
    for zone, priority in priority_zones.items():
        zone_index = zones.index(zone)
        model.Add(
            sum(
                variables[i]
                for i in range(len(values))
                if (i // len(zones)) % len(time_slots) == time_slot and i % len(zones) == zone_index
            ) >= priority
        )

# Solve the model
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Output results
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print(f"Total value of monitoring: {solver.ObjectiveValue()}")
    total_weight = sum(
        weights[i] for i in range(len(weights)) if solver.Value(variables[i]) == 1
    )
    print(f"Total weight (time slots used): {total_weight}")
    print("\nSelected items (drone assignments):")
    for i in range(len(variables)):
        if solver.Value(variables[i]) == 1:  # Corrected condition
            drone_index = i // (len(time_slots) * len(zones))
            time_slot_index = (i // len(zones)) % len(time_slots)
            zone_index = i % len(zones)
            print(f"  Drone {drones[drone_index]} monitors {zones[zone_index]} during {time_slots[time_slot_index]}")
else:
    print("No feasible solution found.")
