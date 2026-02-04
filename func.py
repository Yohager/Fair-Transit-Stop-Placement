import itertools
import math
import gurobipy as gp
from gurobipy import GRB
gp.setParam("LogToConsole", 0)
import numpy as np
from collections import Counter, defaultdict
from typing import List, Set, Dict, Tuple, Callable
from datatype import Agent, TrSPInstance, Point

def Initialize_Instance(
        agents_routes: List[tuple[int, int]],
        candidates: List[int],
        k: int,
        M_walk: np.ndarray,
        M_transit: np.ndarray,
) -> TrSPInstance:
    agents = [Agent(int(a), int(b)) for a, b in agents_routes]
    M_walk = np.asarray(M_walk, dtype=np.float32)
    M_transit = np.asarray(M_transit, dtype=np.float32)
    inst = TrSPInstance(
        agents=agents,
        candidates=candidates,
        k=k,
        walk_matrix=M_walk,
        transit_matrix=M_transit
    )
    return inst 

def bool_to_bitint(mask_bool: np.ndarray) -> int:
    """
    convert boolean array into python int bitset
    """
    packed = np.packbits(mask_bool.astype(np.uint8), bitorder="little")
    return int.from_bytes(packed.tobytes(), byteorder="little", signed=False)

def active_indices_from_bitint(mask: int) -> np.ndarray:
    idxs = []
    while mask:
        lsb = mask & -mask
        idxs.append(lsb.bit_length() - 1)
        mask ^= lsb 
    return np.array(idxs, dtype=np.int32)

def indices_from_bitint(mask: int, n: int) -> np.ndarray:
    """Return indices i in [0,n) where mask has bit i=1."""
    if mask == 0:
        return np.empty((0,), dtype=np.int32)
    nbytes = (n + 7) // 8
    raw = mask.to_bytes(nbytes, byteorder="little", signed=False)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="little")[:n]
    return np.flatnonzero(bits).astype(np.int32)

def update_dict(dict_1: Dict, dict_2: Dict) -> Dict:
    for k, v in dict_2.items():
        dict_1[k] -= v
    return dict_1

def generate_improved_pairs(Instance: TrSPInstance, solution: Set[Point], gamma: float) -> List[List[Tuple[int, int]]]:
    agents = Instance.agents 
    candidates = list(map(int, Instance.candidates))
    n = len(agents)

    M_walk = np.asarray(Instance.walk_matrix, dtype=np.float32)
    M_transit = np.asarray(Instance.transit_matrix, dtype=np.float32)

    A = np.array([int(ag.a) for ag in agents], dtype=np.int32)
    B = np.array([int(ag.b) for ag in agents], dtype=np.int32)

    direct = M_walk[A, B]
    # print(f"checking {direct}")

    improved_pairs = [[] for _ in range(n)]
    sol_pairs = list(itertools.combinations(sorted(map(int, solution)), 2))
    sol_cost = direct.copy()
    for (y1, y2) in sol_pairs:
        c1 = M_walk[A, y1] + M_transit[y1, y2] + M_walk[y2, B]
        c2 = M_walk[A, y2] + M_transit[y2, y1] + M_walk[y1, B]
        sol_cost = np.minimum(sol_cost, np.minimum(c1, c2)) 
    for (y1, y2) in itertools.combinations(candidates, 2):
        c1 = M_walk[A, y1] + M_transit[y1, y2] + M_walk[y2, B]
        c2 = M_walk[A, y2] + M_transit[y2, y1] + M_walk[y1, B]
        tmp = np.minimum(direct, np.minimum(c1, c2))

        improved = (gamma * tmp) < sol_cost  # boolean length n
        idx = np.where(improved)[0]
        if idx.size:
            pair = (int(y1), int(y2))
            for i in idx:
                improved_pairs[int(i)].append(pair)

    # print(f"checking improved_pairs", improved_pairs)
    return improved_pairs

def check_JR(Instance: TrSPInstance, solution: Set[Point], gamma: float, beta=1) -> bool:
    improved_pairs = generate_improved_pairs(Instance, solution, gamma)
    freq = defaultdict(int)
    for agent_pref_pair in improved_pairs:
        for pair in agent_pref_pair:
            freq[pair] += 1
    for _, v in freq.items():
        if v >= beta * math.ceil(2 * len(Instance.agents) / Instance.k):
            return False 
    return True

def compute_JR_approximation(Instance: TrSPInstance, solution: Set[Point]) -> float:
    l, r = 1, 1000
    while r - l > 1e-6:
        m = (r + l) / 2
        if not check_JR(Instance, solution, m):
            l = m
        else:
            r = m
    return l

def check_core(Instance: TrSPInstance, solution: Set[Point], gamma: float, beta: float=1) -> bool:
    agents = Instance.agents
    C = list(map(int, Instance.candidates))
    n = len(agents)
    m = len(C)
    k = int(Instance.k)

    idx_mapping = {p: i for i, p in enumerate(C)}
    improving_pairs = generate_improved_pairs(Instance, solution, gamma)

    improving_pairs_new_idx = []
    for i in range(n):
        pair_i = []
        for a, b in improving_pairs[i]:
            new_a, new_b = idx_mapping.get(a), idx_mapping.get(b)
            pair_i.append((new_a, new_b))
        improving_pairs_new_idx.append(pair_i)
    

    model = gp.Model()
    model.Params.LogToConsole = 0

    x = model.addVars(n, vtype=GRB.BINARY, name="x")        # agent var
    y = model.addVars(m, vtype=GRB.BINARY, name="y")        # stop var
    z = model.addVars(m, m, vtype=GRB.BINARY, name="z")     # stop pair var

    model.addConstrs((z[i, j] <= y[i] for i in range(m) for j in range(m)), name="constraint_2_1")
    model.addConstrs((z[i, j] <= y[j] for i in range(m) for j in range(m)), name="constraint_2_2")
    model.addConstrs(
        (
            x[i] <= gp.quicksum(z[t0, t1] for (t0, t1) in improving_pairs_new_idx[i])
            for i in range(n)
        ),
        name="constraint_1"
    )
    model.addConstr(
        gp.quicksum(x[i] for i in range(n)) >= beta * gp.quicksum(y[j] for j in range(m)) * (n / k),
        name="constraint_3"
    )

    model.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    model.optimize()

    if model.ObjVal == 0:
        return True
    else:
        return False 

def compute_core_approximation(Instance: TrSPInstance, solution: Set[Point], beta=1) -> float:
    l, r = 1, 100
    while r - l > 1e-6:
        m = (r + l) / 2
        # print("here")
        if not check_core(Instance, solution, m, beta):
            l = m
        else:
            r = m
    return l 
