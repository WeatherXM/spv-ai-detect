import os
import json
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class SequenceOptimizer:
    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config["paths"]["cache_dir"])
        self.sequence_path = self.cache_dir / "sequence.json"
        self.override_path = self.cache_dir / "sequence_override.json"

    def optimize_sequence(self, filenames, dist_matrix, force_recompute=False):
        N = len(filenames)
        if N <= 1:
            return filenames

        # Check for user sequence override first!
        if self.override_path.exists() and not force_recompute:
            try:
                with open(self.override_path, "r") as f:
                    override_data = json.load(f)
                override_seq = override_data.get("sequence", [])
                valid_override = [fname for fname in override_seq if fname in filenames]
                if len(valid_override) > 0:
                    logger.info(f"Loaded user-approved sequence override ({len(valid_override)} photos).")
                    return valid_override
            except Exception as e:
                logger.warning(f"Could not load sequence override ({e}). Falling back to standard sequence...")

        if self.sequence_path.exists() and not force_recompute:
            try:
                with open(self.sequence_path, "r") as f:
                    cached_data = json.load(f)
                if cached_data.get("count") == N and set(cached_data.get("sequence", [])) == set(filenames):
                    logger.info(f"Loaded cached sequence order ({N} images).")
                    return cached_data["sequence"]
            except Exception as e:
                logger.warning(f"Could not load sequence cache ({e}). Recomputing...")

        logger.info(f"Optimizing sequence for {N} images using Traveling Salesperson Problem (TSP) solver...")

        route_indices = self._solve_tsp_ortools(dist_matrix)
        if route_indices is None or len(route_indices) != N:
            logger.warning("OR-Tools solver returned incomplete route. Falling back to 2-Opt Local Search...")
            route_indices = self._solve_tsp_2opt(dist_matrix)

        ordered_filenames = [filenames[idx] for idx in route_indices]

        total_cost = 0.0
        for i in range(N - 1):
            total_cost += dist_matrix[route_indices[i], route_indices[i + 1]]

        logger.info(f"Sequence optimization complete! Total path transition cost: {total_cost:.4f}")

        cache_data = {
            "count": N,
            "total_cost": float(total_cost),
            "sequence": ordered_filenames
        }
        with open(self.sequence_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        return ordered_filenames

    def _solve_tsp_ortools(self, dist_matrix):
        try:
            from ortools.constraint_solver import routing_enums_pb2
            from ortools.constraint_solver import pywrapcp

            N = dist_matrix.shape[0]
            scale_factor = 100000
            int_dist_matrix = (dist_matrix * scale_factor).astype(np.int64)

            manager = pywrapcp.RoutingIndexManager(N, 1, 0)
            routing = pywrapcp.RoutingModel(manager)

            def distance_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return int_dist_matrix[from_node, to_node]

            transit_callback_index = routing.RegisterTransitCallback(distance_callback)
            routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

            search_parameters = pywrapcp.DefaultRoutingSearchParameters()
            search_parameters.first_solution_strategy = (
                routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
            )
            search_parameters.local_search_metaheuristic = (
                routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
            )
            search_parameters.time_limit.seconds = 15

            solution = routing.SolveWithParameters(search_parameters)

            if solution:
                route = []
                index = routing.Start(0)
                while not routing.IsEnd(index):
                    node = manager.IndexToNode(index)
                    route.append(node)
                    index = solution.Value(routing.NextVar(index))

                if len(route) == N:
                    return route
        except Exception as e:
            logger.warning(f"OR-Tools solver exception: {e}")

        return None

    def _solve_tsp_2opt(self, dist_matrix):
        N = dist_matrix.shape[0]
        visited = [False] * N
        route = [0]
        visited[0] = True

        for _ in range(N - 1):
            curr = route[-1]
            distances = dist_matrix[curr].copy()
            distances[visited] = np.inf
            next_node = int(np.argmin(distances))
            route.append(next_node)
            visited[next_node] = True

        improved = True
        max_iter = 100
        iteration = 0

        def route_distance(r):
            return sum(dist_matrix[r[i], r[i+1]] for i in range(len(r)-1))

        best_distance = route_distance(route)

        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            for i in range(1, N - 1):
                for j in range(i + 1, N):
                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    new_dist = route_distance(new_route)
                    if new_dist < best_distance:
                        route = new_route
                        best_distance = new_dist
                        improved = True
                        break
                if improved:
                    break

        return route
