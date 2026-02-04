# -*- coding: utf-8 -*-

import itertools
import math
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Set, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
from datatype import TrSPInstance, Agent, Point
from func import bool_to_bitint, active_indices_from_bitint, indices_from_bitint

def GC_TrSP(Instance: TrSPInstance) -> Set[Point]:
    Agents, C, k = Instance.agents, list(map(int, Instance.candidates)), int(Instance.k)
    M = np.asarray(Instance.walk_matrix, dtype=np.float32)
    n = 2 * len(Agents)
    if n == 0 or len(C) == 0:
        return set()

    threshold = math.ceil(n / k)

    theta = defaultdict(int)
    for agent in Agents:
        theta[int(agent.a)] += 1
        theta[int(agent.b)] += 1
    
    E = np.array(list(theta.keys()), dtype=np.int32)
    active = np.array([theta[int(p)] for p in E], dtype=np.int32)
    m = len(E)

    C_sorted = sorted(C)

    Selected = set()
    Unselected = set(C_sorted)

    D = {c: M[c, E] for c in C_sorted}

    sort_idx = {c: np.argsort(D[c], kind="mergesort") for c in C_sorted}

    def deactivate_by_candidate(c: int, r: float) -> None:
        mask = (D[c] <= r)
        active[mask] = 0
    
    def deactivate_by_selected(r: float) -> None:
        for s in Selected:
            deactivate_by_candidate(s, r)
    
    def cover_weight(c: int, r: float) -> int:
        mask = (D[c] <= r)
        return int(active[mask].sum())
    
    def r_ast(c: int) -> float:
        idx = sort_idx[c]
        w = active[idx]
        total = int(w.sum())
        if total < threshold:
            return float("inf")

        pref = np.cumsum(w, dtype=np.int64)
        j = int(np.searchsorted(pref, threshold, side="left"))
        return float(D[c][idx[j]])
    
    while active.sum() > 0:
        best_r = float("inf")
        best_c = None
        for c in C_sorted:
            if c not in Unselected:
                continue
            rc = r_ast(c)
            if rc < best_r:
                best_r = rc 
                best_c = c 
        if best_c is None or not np.isfinite(best_r):
            break 

        r = float(best_r)
        deactivate_by_selected(r)
        if active.sum() == 0:
            break 

        while True:
            chosen = None 
            for c in C_sorted:
                if c not in Unselected:
                    continue
                if cover_weight(c, r) >= threshold:
                    chosen = c
                    break 
            if chosen is None:
                break 
            # print(f"checking selection: {chosen, r}")
            Selected.add(chosen)
            Unselected.remove(chosen)
            deactivate_by_candidate(chosen, r)

            if active.sum() == 0:
                break 
    return Selected

def ECA(Instance: TrSPInstance) -> Set[Point]:
    Agents = Instance.agents
    C = list(Instance.candidates)
    k = Instance.k 
    M_walk = Instance.walk_matrix
    M_transit = Instance.transit_matrix 
    n = len(Agents)

    threshold = math.ceil(2 * n / k)
    threshold_2 = math.ceil(n / k)
    A = np.array([agent.a for agent in Agents], dtype=np.int32)
    B = np.array([agent.b for agent in Agents], dtype=np.int32)

    pairs = list(itertools.combinations(C, 2))
    P = len(pairs)
    pair_to_idx = {}
    for p, (y1, y2) in enumerate(pairs):
        key = (y1, y2) if y1 < y2 else (y2, y1)
        pair_to_idx[key] = p

    ActiveMask = (1 << n) - 1
    Selected_Stops = set()
    Selected_Pair_Idx = []

    direct = M_walk[A, B]
    costs_all = [None] * P 
    for p, (y1, y2) in enumerate(pairs):
        c1 = M_walk[A, y1] + M_transit[y1, y2] + M_walk[y2, B]
        c2 = M_walk[A, y2] + M_transit[y2, y1] + M_walk[y1, B]
        costs_all[p] = np.minimum(direct, np.minimum(c1, c2)).astype(np.float32)

    def deactivate_by_existing_Y(r: float) -> None:
        nonlocal ActiveMask
        if ActiveMask == 0:
            return 
        served_mask = 0 
        served_mask |= bool_to_bitint(direct <= r)
        if len(Selected_Stops) >= 2:
            S = sorted(Selected_Stops)
            for y1, y2 in itertools.combinations(S, 2):
                key = (y1, y2) if y1 < y2 else (y2, y1)
                p_idx = pair_to_idx.get(key, None)
                if p_idx is None:
                    continue
                served_mask |= bool_to_bitint(costs_all[p_idx] <= r)
        
        ActiveMask &= ~served_mask

    while ActiveMask:
        idx_act = active_indices_from_bitint(ActiveMask)
        if idx_act.size == 0:
            break

        if idx_act.size < threshold:
            deactivate_by_existing_Y(float('inf'))
            break 

        selected_pair_set = set(Selected_Pair_Idx)

        best_r = None 
        best_r_pair_idx = None 
        for p in range(P):
            if p in selected_pair_set:
                continue
            ca = costs_all[p][idx_act]
            r_ast = float(np.partition(ca, threshold - 1)[threshold - 1])
            if best_r is None or r_ast < best_r or (r_ast == best_r and p < best_r_pair_idx):
                best_r = r_ast 
                best_r_pair_idx = p
        
        if best_r is None:
            deactivate_by_existing_Y(float("inf"))
            break 

        r = float(best_r)

        deactivate_by_existing_Y(r)
        if ActiveMask == 0:
            break 

        cover= [0] * P 
        for p in range(P):
            if p in selected_pair_set:
                continue
            cover[p] = bool_to_bitint(costs_all[p] <= r)

        while True:
            best_p = -1
            for p in range(P):
                if p in selected_pair_set:
                    continue
                y1, y2 = pairs[p]
                if (y1 in Selected_Stops) or (y2 in Selected_Stops):
                    continue
                if (cover[p] & ActiveMask).bit_count() >= threshold:
                    best_p = p
                    break

            if best_p != -1:
                y1, y2 = pairs[best_p]
                Selected_Stops.add(y1); Selected_Stops.add(y2)
                Selected_Pair_Idx.append(best_p); selected_pair_set.add(best_p)
                ActiveMask &= ~cover[best_p]
                if ActiveMask == 0:
                    break
                continue

            best_su_p = -1
            for p in range(P):
                if p in selected_pair_set:
                    continue
                y1, y2 = pairs[p]

                y1_sel = (y1 in Selected_Stops)
                y2_sel = (y2 in Selected_Stops)
                if y1_sel == y2_sel:
                    continue  # skip UU and SS

                if (cover[p] & ActiveMask).bit_count() >= threshold_2:
                    best_su_p = p
                    break 

            if best_su_p == -1:
                break

            y1, y2 = pairs[best_su_p]
            if y1 in Selected_Stops:
                new_stop = y2
            else:
                new_stop = y1

            Selected_Stops.add(new_stop)
            Selected_Pair_Idx.append(best_su_p); selected_pair_set.add(best_su_p)
            ActiveMask &= ~cover[best_su_p]
            if ActiveMask == 0:
                break

    return Selected_Stops

def Lambda_Hybrid(Instance: TrSPInstance, lambada: float) -> Set[Point]:
    Agents = Instance.agents
    C = list(map(int, Instance.candidates))
    k = int(Instance.k)

    M_walk = np.asarray(Instance.walk_matrix, dtype=np.float32)
    M_transit = np.asarray(Instance.transit_matrix, dtype=np.float32)

    n_agents = len(Agents)
    if n_agents == 0 or len(C) == 0:
        return set()

    thresh_agents = int(math.ceil((2 * n_agents) / k))
    thresh_endpts = int(math.ceil((2 * n_agents) / k))

    A = np.asarray([int(agent.a) for agent in Agents], dtype=np.int32)
    B = np.asarray([int(agent.b) for agent in Agents], dtype=np.int32)

    direct = M_walk[A, B].astype(np.float32)

    pairs = list(itertools.combinations(C, 2))
    P = len(pairs)

    pair_to_idx = {}
    for p, (y1, y2) in enumerate(pairs):
        key = (y1, y2) if y1 < y2 else (y2, y1)
        pair_to_idx[key] = p

    costs_all = [None] * P
    for p, (y1, y2) in enumerate(pairs):
        c1 = M_walk[A, y1] + M_transit[y1, y2] + M_walk[y2, B]
        c2 = M_walk[A, y2] + M_transit[y2, y1] + M_walk[y1, B]
        costs_all[p] = np.minimum(direct, np.minimum(c1, c2)).astype(np.float32)

    endpoint_counts = {}
    for i in range(n_agents):
        endpoint_counts[int(A[i])] = endpoint_counts.get(int(A[i]), 0) + 1
        endpoint_counts[int(B[i])] = endpoint_counts.get(int(B[i]), 0) + 1

    endpoint_ids = np.asarray(list(endpoint_counts.keys()), dtype=np.int32)
    m_end = int(endpoint_ids.size)

    idx_of_endpoint = {int(e): j for j, e in enumerate(endpoint_ids)}
    weights = np.asarray([endpoint_counts[int(e)] for e in endpoint_ids], dtype=np.int32)

    posA = np.asarray([idx_of_endpoint[int(a)] for a in A], dtype=np.int32)
    posB = np.asarray([idx_of_endpoint[int(b)] for b in B], dtype=np.int32)

    Y = set()

    inf = float("inf")
    dist_to_Y = np.full(m_end, inf, dtype=np.float32)

    ActiveAgentsMask = (1 << n_agents).__sub__(1)

    def update_dist_with_stop(y):
        nonlocal dist_to_Y
        dy = M_walk[int(y), endpoint_ids]
        dist_to_Y = np.minimum(dist_to_Y, dy.astype(np.float32))

    def eligible_agent_indices():
        if ActiveAgentsMask == 0:
            return np.zeros(0, dtype=np.int32)
        idx = active_indices_from_bitint(ActiveAgentsMask)
        if idx.size == 0:
            return idx
        ok = (weights[posA[idx]] > 0) & (weights[posB[idx]] > 0)
        return idx[ok]

    def current_cost_for_agents(agent_idx):
        if agent_idx.size == 0:
            return np.zeros(0, dtype=np.float32)
        cost = direct[agent_idx].copy()
        if len(Y) >= 2:
            S = sorted(Y)
            for y1, y2 in itertools.combinations(S, 2):
                key = (y1, y2) if y1 < y2 else (y2, y1)
                p_idx = pair_to_idx.get(key, None)
                if p_idx is None:
                    continue
                cost = np.minimum(cost, costs_all[p_idx][agent_idx])
        return cost

    def deactivate_by_existing_Y(r):
        nonlocal ActiveAgentsMask, weights

        changed = True
        while changed:
            changed = False

            idx_el = eligible_agent_indices()
            if idx_el.size > 0:
                cc = current_cost_for_agents(idx_el)
                served = cc <= float(r)
                if bool(served.any()):
                    served_idx = idx_el[served]
                    for i in served_idx.tolist():
                        ActiveAgentsMask &= ~(1 << int(i))
                        pa = int(posA[i])
                        pb = int(posB[i])
                        if weights[pa] > 0:
                            weights[pa] = int(weights[pa]).__sub__(1)
                        if weights[pb] > 0:
                            weights[pb] = int(weights[pb]).__sub__(1)
                    changed = True

            if (float(lambada) > 0.0) and (len(Y) > 0):
                rho = float(lambada) * float(r)
                mask = (weights > 0) & (dist_to_Y <= rho)
                if bool(mask.any()):
                    weights[mask] = 0
                    changed = True

    def r_event_cost_min():
        idx_el = eligible_agent_indices()
        if idx_el.size == 0:
            return inf
        cc = current_cost_for_agents(idx_el)
        if cc.size == 0:
            return inf
        return float(cc.min())

    def r_event_dist_min():
        if (float(lambada) <= 0.0) or (len(Y) == 0):
            return inf
        mask = weights > 0
        if not bool(mask.any()):
            return inf
        return float((dist_to_Y[mask].min()) / float(lambada))

    def r_event_best_pair():
        idx_el = eligible_agent_indices()
        if idx_el.size < thresh_agents:
            return inf
        kth = int(thresh_agents).__sub__(1)

        best = inf
        for p in range(P):
            y1, y2 = pairs[p]
            if (y1 in Y) and (y2 in Y):
                continue
            ca = costs_all[p][idx_el]
            r_ast = float(np.partition(ca, kth)[kth])
            if r_ast < best:
                best = r_ast
        return best

    def r_event_best_stop():
        if float(lambada) <= 0.0:
            return inf
        mask_end = weights > 0
        if not bool(mask_end.any()):
            return inf

        E_ids = endpoint_ids[mask_end]
        E_w = weights[mask_end].astype(np.int64)
        need = int(thresh_endpts)

        best = inf
        for c in C:
            if c in Y:
                continue
            d = M_walk[int(c), E_ids].astype(np.float32)
            order = np.argsort(d, kind="mergesort")
            d_sorted = d[order]
            w_sorted = E_w[order]
            pref = np.cumsum(w_sorted, dtype=np.int64)
            j = int(np.searchsorted(pref, need, side="left"))
            if j >= d_sorted.size:
                continue
            r_ast = float(d_sorted[j] / float(lambada))
            if r_ast < best:
                best = r_ast
        return best

    def select_pairs_at_r(r):
        nonlocal ActiveAgentsMask, weights, Y

        while True:
            idx_el = eligible_agent_indices()
            if idx_el.size < thresh_agents:
                return

            chosen = None
            cover_mask_bits = None

            for p in range(P):
                y1, y2 = pairs[p]
                if (y1 in Y) and (y2 in Y):
                    continue
                ok = costs_all[p][idx_el] <= float(r)
                if int(ok.sum()) >= thresh_agents:
                    chosen = p
                    cover_mask_bits = ok
                    break

            if chosen is None:
                return

            y1, y2 = pairs[chosen]
            if y1 not in Y:
                Y.add(y1)
                update_dist_with_stop(y1)
            if y2 not in Y:
                Y.add(y2)
                update_dist_with_stop(y2)

            served_idx = idx_el[cover_mask_bits]
            for i in served_idx.tolist():
                ActiveAgentsMask &= ~(1 << int(i))
                pa = int(posA[i])
                pb = int(posB[i])
                if weights[pa] > 0:
                    weights[pa] = int(weights[pa]).__sub__(1)
                if weights[pb] > 0:
                    weights[pb] = int(weights[pb]).__sub__(1)

    def select_stops_at_r(r):
        nonlocal weights, Y

        if float(lambada) <= 0.0:
            return

        rho = float(lambada) * float(r)
        while True:
            if not bool((weights > 0).any()):
                return

            chosen = None
            cover_mask_end = None

            for c in C:
                if c in Y:
                    continue
                d = M_walk[int(c), endpoint_ids].astype(np.float32)
                mask = (weights > 0) & (d <= rho)
                if int(weights[mask].sum()) >= thresh_endpts:
                    chosen = c
                    cover_mask_end = mask
                    break

            if chosen is None:
                return

            Y.add(int(chosen))
            update_dist_with_stop(int(chosen))
            weights[cover_mask_end] = 0

    r = 0.0
    while bool((weights > 0).any()):
        rc = r_event_cost_min()
        rd = r_event_dist_min()
        rp = r_event_best_pair()
        rs = r_event_best_stop()

        r_next = min(rc, rd, rp, rs)
        if not np.isfinite(r_next):
            break
        if r_next < r:
            r_next = r
        r = float(r_next)

        deactivate_by_existing_Y(r)
        if not bool((weights > 0).any()):
            break

        select_pairs_at_r(r)
        if not bool((weights > 0).any()):
            break

        select_stops_at_r(r)
    return Y      
    

