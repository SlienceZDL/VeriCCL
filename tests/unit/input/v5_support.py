import copy
import json


def request_documents(operator="allreduce", ranks=4):
    return {
        "topology": {"name": "test-only", "ranks": ranks,
                     "nodes": [{"id": 0, "ranks": list(range(ranks))}],
                     "directed_links": [{"src": a, "dst": b, "alpha_us": 1.0,
                                         "beta_us_per_byte": 0.01}
                                        for a in range(ranks) for b in range(ranks) if a != b]},
        "sketch": {
            "collective": {"operator": operator},
            "hyperparameters": {"total_size_bytes": 4096, "slice_size_bytes": 1024},
            "solver": {"per_model_timeout_s": 2, "cpu_budget": 2},
            "budgets": {"max_initial_synthesis_s": 10, "max_offline_validation_per_round_s": 5,
                        "max_offline_validation_total_s": 20, "max_repair_per_round_s": 5,
                        "max_repair_total_s": 20, "max_tuning_rounds": 2, "cleanup_reserve_s": 0.5},
            "tolerances": {"absolute": 1e-5, "relative": 1e-5},
            "calibration": {"enabled": False},
        },
        "atom": {"synthesis_method": "hierarchy"},
        "runtime": {"rank_map": [{"rank": r, "host": "worker-a", "device": r} for r in range(ranks)],
                    "environment": {"NCCL_BUFFSIZE": "16777216"},
                    "scratch_bytes_per_rank": 16777216},
    }


def write_request(tmp_path, documents=None):
    documents = copy.deepcopy(documents or request_documents())
    result = {}
    for name, document in documents.items():
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(document))
        result[name] = path
    return result
