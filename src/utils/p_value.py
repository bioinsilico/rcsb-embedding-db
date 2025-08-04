import numpy as np


def compute_p_value(score, embedding, background):
    e_i = np.linalg.norm(embedding)
    scores = []
    for e_j in background:
        scores.append(np.dot(e_i, e_j))

    return 1 + len([s for s in scores if s >= score]) / 1 + len(background)
