from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Define parameters
zones = ["ZoneA", "ZoneB", "ZoneC", "ZoneD", "ZoneE"]
zone_distances = [
    [0, 2, 9, 10, 7],
    [2, 0, 6, 4, 3],
    [9, 6, 0, 8, 5],
    [10, 4, 8, 0, 6],
    [7, 3, 5, 6, 0]
]
num_zones = len(zones)
depot = 0  # Starting zone (ZoneA)

# Create the routing index manager
manager = pywrapcp.RoutingIndexManager(num_zones, 1, depot)

# Create the routing model
routing = pywrapcp.RoutingModel(manager)

# Define cost function (distance between zones)
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return zone_distances[from_node][to_node]

# Register the distance callback
transit_callback_index = routing.RegisterTransitCallback(distance_callback)

# Set the cost function for the routing model
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Define search parameters
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
search_parameters.log_search = True  # Enable solver logging

# Solve the problem
solution = routing.SolveWithParameters(search_parameters)

# Output results
if solution:
    print("Optimal route:")
    route = []
    index = routing.Start(0)
    route_distance = 0
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route.append(zones[node])
        next_index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(index, next_index, 0)
        index = next_index
    route.append(zones[depot])  # Return to starting point
    print(" -> ".join(route))
    print(f"Total distance: {route_distance}")
else:
    print("No solution found.")
