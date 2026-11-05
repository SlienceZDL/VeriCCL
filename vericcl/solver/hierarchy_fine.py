"""Joint discrete route/channel/order refinement of one equivalent demand."""
from dataclasses import replace
from vericcl.errors import ConstructionInfeasibleError
from vericcl.solver.milp import solve_milp
from vericcl.solver.routing import RoutePattern, RoutingModelStats
from vericcl.topology.model import LinkKey


def solve_fine_pattern(template, problem, channels, objective, budget, *, result=None):
    unit = template.representative
    problem = replace(problem, node=unit.node, demands=unit.demands,
                      infeasible_demand_ids=tuple(d.demand_id for d in unit.demands if not d.candidate_paths))
    if result is None:
        result = solve_milp(problem, channels, objective, budget, None)
    schedule = result.node_schedules.get(unit.node.node_id)
    if schedule is None:
        raise ConstructionInfeasibleError('fine representative has no valid incumbent: ' + result.metrics.termination_reason)
    flows = schedule.metadata['selected_flows']
    paths, edges = [], set()
    for demand in unit.demands:
        outgoing = {src: dst for src, dst in flows[demand.demand_id]}
        rank, path, visited = demand.root_rank, [], set()
        while rank != demand.required_leaf_rank:
            if rank in visited or rank not in outgoing:
                raise ConstructionInfeasibleError('fine representative flow is not a source-to-target path')
            visited.add(rank)
            link = rank, outgoing[rank]
            path.append(link)
            edges.add(link)
            rank = outgoing[rank]
        paths.append((demand.demand_id, tuple(path)))
    epochs = {value: index for index, value in enumerate(sorted({t.st_time for t in schedule.transfers}))}
    priorities = {(t.src_rank, t.dst_rank): epochs[t.st_time] for t in schedule.transfers}
    return RoutePattern(template.template_id, channels, objective, tuple(sorted(edges)), tuple(paths), result.metrics,
                        RoutingModelStats(**schedule.metadata['native_model_stats']),
                        tuple((src, dst, priorities[src, dst]) for src, dst in sorted(edges)))
