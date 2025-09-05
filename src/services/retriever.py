import numpy as np


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))


def mmr(query_vec, cand_vecs, lambda_mult=0.5, k=5):
    selected, selected_idx = [], []
    sim_to_query = [cosine(query_vec, v) for _, v in cand_vecs]
    while len(selected) < min(k, len(cand_vecs)):
        max_score, max_j = -1e9, -1
        for j, (idx, v) in enumerate(cand_vecs):
            if j in selected_idx:
                continue
            if not selected:
                score = sim_to_query[j]
            else:
                sim_to_selected = max(cosine(v, cand_vecs[s][1]) for s in selected_idx)
                score = lambda_mult * sim_to_query[j] - (1 - lambda_mult) * sim_to_selected
            if score > max_score:
                max_score, max_j = score, j
        selected_idx.append(max_j)
        selected.append(cand_vecs[max_j])
    return [cand_vecs[j][0] for j in selected_idx]
