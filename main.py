# -*- coding: utf-8 -*-

import pandas as pd
import json
import csv
import hashlib
from pathlib import Path
import statistics
import numpy as np
from typing import List, Set, Dict, Any, Tuple, Callable, Optional
from datatype import TrSPInstance, Agent, Point
from func import compute_JR_approximation, compute_core_approximation
from func import Initialize_Instance
from model import GC_TrSP, ECA, Lambda_Hybrid
import argparse

# file paths
AGENT_ROUTES = "../data/agents_routes.csv"
POINTS_CSV = "../data/unique_points_with_id.csv"
AUTO_MATRIX = "../data/matrix_auto_sec.npz"
PED_MATRIX = "../data/matrix_pedestrian_sec.npz"

def sample_function(
    n: int, # number of agents
    m: int, # number of candidate stops
    K: int, # number of stop selection
    transit_scaling: int, # transit cost scaling
    candidate_sample: int, # 0: sample within agent location; 1: sample in the pool
    ):
    rng = np.random.default_rng()

    # creating the pool
    agents = pd.read_csv(AGENT_ROUTES)
    if "pickup_point_id" not in agents.columns or "dropoff_point_id" not in agents.columns:
        raise ValueError("AGENTS_CSV must contain pickup_point_id and dropoff_point_id")
    
    if len(agents) < n:
        raise ValueError(f"Not enough agents in pool: {len(agents)} < {n}")

    sample_idx = rng.choice(len(agents), size=n, replace=False)
    agents_s = agents.iloc[sample_idx].reset_index(drop=True)

    if candidate_sample == 0:
        cand_pool = np.unique(
            np.concatenate([agents_s["pickup_point_id"].to_numpy(),
                            agents_s["dropoff_point_id"].to_numpy()])
        )
    elif candidate_sample == 1:
        # read from the whole pool
        points = pd.read_csv(POINTS_CSV)
        cand_pool = points["point_id"].to_numpy(dtype=np.int32)
    else:
        raise ValueError("Unknown SAMPLE MODE")
    
    if len(cand_pool) < m:
        cand_pool = np.unique(
            np.concatenate(
                [
                    agents["pickup_point_id"].to_numpy(),
                    agents["dropoff_point_id"].to_numpy(),
                ]
            )
        ).astype(np.int32)

    candidates_old_idx = rng.choice(cand_pool, size=m, replace=False).astype(np.int32)

    all_points_old_idx = np.unique(
        np.concatenate([
            agents_s["pickup_point_id"].to_numpy(),
            agents_s["dropoff_point_id"].to_numpy(),
            candidates_old_idx
        ])
    ).astype(np.int32)

    all_points_old_sorted = np.sort(all_points_old_idx)

    old2new = {int(old): i for i, old in enumerate(all_points_old_sorted)} # index mapping
    m = len(all_points_old_sorted)

    # get new index for agent routes and candidates 
    agent_routes_new_idx = pd.DataFrame({
        "agent_id": np.arange(len(agents_s), dtype=np.int32),
        "a": agents_s["pickup_point_id"].map(old2new).astype(np.int32),
        "b": agents_s["dropoff_point_id"].map(old2new).astype(np.int32),
    })

    candidates_new_idx = pd.DataFrame({
        "candidate_new_id": np.arange(len(candidates_old_idx), dtype=np.int32),
        "candidate_point": [old2new[int(x)] for x in candidates_old_idx]
    })

    auto_matrix = np.load(AUTO_MATRIX)["time_sec"]
    ped_matrix = np.load(PED_MATRIX)["time_sec"]

    idx = all_points_old_sorted
    ped_sample_matrix = (ped_matrix[np.ix_(idx, idx)] / 3600).astype(np.float32)
    auto_sample_matrix = auto_matrix[np.ix_(idx, idx)].astype(np.float32)

    agents_routes = list(zip(agent_routes_new_idx["a"].tolist(), agent_routes_new_idx["b"].tolist()))
    candidates = candidates_new_idx["candidate_point"].tolist()
    auto_sample_matrix = (transit_scaling * auto_sample_matrix / 3600).astype(np.float32)

    Inst = Initialize_Instance(agents_routes, candidates, K, ped_sample_matrix, auto_sample_matrix)
    return Inst

def Func_approximation(Instance: TrSPInstance, check_type: str):
    GC_res = GC_TrSP(Instance)
    ECA_res = ECA(Instance)
    Hybrid_res = Lambda_Hybrid(Instance, 0.5)
    if check_type == "JR":
        GC_approx = compute_JR_approximation(Instance, GC_res)
        ECA_approx = compute_JR_approximation(Instance, ECA_res)
        Hybrid_approx = compute_JR_approximation(Instance, Hybrid_res)
        return GC_approx, ECA_approx, Hybrid_approx
    elif check_type == "core":
        GC_approx = compute_core_approximation(Instance, GC_res)
        ECA_approx = compute_core_approximation(Instance, ECA_res)
        Hybrid_approx = compute_core_approximation(Instance, Hybrid_res)
        return GC_approx, ECA_approx, Hybrid_approx
    else:
        raise ValueError("No such type!")       

def _stable_experiment_id(params: Dict[str, Any]) -> str:
    """
    generate_exp_id
    """
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0, "min": float(values[0]), "max": float(values[0])}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }

def write_results(
    output_dir: str,
    params: Dict[str, Any],
    gc_vals: List[float],
    eca_vals: List[float],
    hybrid_vals: List[float],
) -> Path:
    """
    output_dir/
      exp_<id>/
        params.json
        results.csv
        summary.csv
    """
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    exp_id = _stable_experiment_id(params)
    exp_dir = base / f"exp_{exp_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2, sort_keys=True)

    n = max(len(gc_vals), len(eca_vals), len(hybrid_vals))
    results_path = exp_dir / "results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "GC", "ECA", "Hybrid"])
        for i in range(n):
            writer.writerow([
                i,
                gc_vals[i] if i < len(gc_vals) else "",
                eca_vals[i] if i < len(eca_vals) else "",
                hybrid_vals[i] if i < len(hybrid_vals) else "",
            ])

    summary_path = exp_dir / "summary.csv"
    gc_s = _summarize(gc_vals)
    eca_s = _summarize(eca_vals)
    hyb_s = _summarize(hybrid_vals)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["algo", "mean", "std", "min", "max", "rounds"])
        writer.writerow(["GC", gc_s["mean"], gc_s["std"], gc_s["min"], gc_s["max"], len(gc_vals)])
        writer.writerow(["ECA", eca_s["mean"], eca_s["std"], eca_s["min"], eca_s["max"], len(eca_vals)])
        writer.writerow(["Hybrid", hyb_s["mean"], hyb_s["std"], hyb_s["min"], hyb_s["max"], len(hybrid_vals)])

    return exp_dir


def JR_run(
    rounds: int,
    output_dir: str,
    k: int,
    transit_scaling: int,
    n_agents: int = 400,
    m_candidates: int = 800,
    candidate_mode: int = 0,
):
    params = {
        "n_agents": n_agents,
        "m_candidates": m_candidates,
        "k": k,
        "transit_scaling": transit_scaling,
        "candidate_mode": candidate_mode,
        "rounds": rounds,
        "check_type": "JR",
    }

    gc_vals: List[float] = []
    eca_vals: List[float] = []
    hybrid_vals: List[float] = []

    for _ in range(rounds):
        inst = sample_function(
            params["n_agents"],
            params["m_candidates"],
            params["k"],
            params["transit_scaling"],
            params["candidate_mode"],
        )
        gc, eca, hybrid = Func_approximation(inst, params["check_type"])
        gc_vals.append(gc)
        eca_vals.append(eca)
        hybrid_vals.append(hybrid)

    exp_dir = write_results(output_dir, params, gc_vals, eca_vals, hybrid_vals)
    print(f"Saved to {exp_dir}")


def core_run(
    rounds: int,
    output_dir: str,
    k: int,
    transit_scaling: int,
    n_agents: int = 40,
    m_candidates: int = 80,
    candidate_mode: int = 0,
):
    params = {
        "n_agents": n_agents,
        "m_candidates": m_candidates,
        "k": k,
        "transit_scaling": transit_scaling,
        "candidate_mode": candidate_mode,
        "rounds": rounds,
        "check_type": "core",
    }

    gc_vals: List[float] = []
    eca_vals: List[float] = []
    hybrid_vals: List[float] = []

    for _ in range(rounds):
        inst = sample_function(
            params["n_agents"],
            params["m_candidates"],
            params["k"],
            params["transit_scaling"],
            params["candidate_mode"],
        )
        gc, eca, hybrid = Func_approximation(inst, params["check_type"])
        gc_vals.append(gc)
        eca_vals.append(eca)
        hybrid_vals.append(hybrid)

    exp_dir = write_results(output_dir, params, gc_vals, eca_vals, hybrid_vals)
    print(f"Saved to {exp_dir}")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("rounds", type=int)
    p.add_argument("n_agents", type=int, nargs="?", default=400)
    p.add_argument("m_candidates", type=int, nargs="?", default=800)
    p.add_argument("k", type=int)
    p.add_argument("candidate_mode", type=int, nargs="?", default=0)
    p.add_argument("check_type", choices=["JR", "core"], help="Currently only JR is wired here")
    p.add_argument("transit_scaling", type=int)
    p.add_argument("output_dir", type=str)
    return p


def main():
    args = build_parser().parse_args()

    if args.check_type == "JR":
        JR_run(
            rounds=args.rounds,
            output_dir=args.output_dir,
            k=args.k,
            transit_scaling=args.transit_scaling,
            n_agents=args.n_agents,
            m_candidates=args.m_candidates,
            candidate_mode=args.candidate_mode,
        )
    elif args.check_type == "core":
        core_run(
            rounds=args.rounds,
            output_dir=args.output_dir,
            k=args.k,
            transit_scaling=args.transit_scaling,
            n_agents=args.n_agents,
            m_candidates=args.m_candidates,
            candidate_mode=args.candidate_mode,
        )

if __name__ == "__main__":
    main()
