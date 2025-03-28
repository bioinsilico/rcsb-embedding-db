import argparse
import os
import random
import time

import numpy as np
from scipy import stats

from utils.embedding_provider import EmbeddingProvider, MilvusCollection


def confidence_interval(data, confidence=0.95):
    """
    Calculates the confidence interval for a list of values.

    Args:
      data: A list or numpy array of numerical data.
      confidence: The desired confidence level (e.g., 0.95 for 95% confidence).

    Returns:
      A tuple containing the lower and upper bounds of the confidence interval.
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    se = stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m - h, m + h


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rcsb_host', type=int, required=True)
    parser.add_argument('--afdb_host', type=int, required=True)
    parser.add_argument('--collection_name', type=int, required=True)
    parser.add_argument('--embedding_path', type=int, required=True)
    parser.add_argument('--n_queries', type=int, required=True)
    parser.add_argument('--n_results', type=int, required=True)
    args = parser.parse_args()

    rcsb_host = args.rcsb_host
    afdb_host = args.afdb_host
    collection_name = args.collection_name
    embedding_path = args.embedding_path
    n_queries = args.n_queries
    n_results = args.n_results

    embedding_provider = EmbeddingProvider()
    embedding_provider.connect(
        rcsb_host,
        afdb_host
    )

    embedding_files = list(os.listdir(embedding_path))
    times = []
    for _ in range(10):
        random_id = random.sample(embedding_files, 1)[0]
        random_id = ".".join(random_id.split(".")[0:2])
        rcsb_embedding = embedding_provider.get_by_id(
            MilvusCollection.instance_collection,
            random_id
        )
        start_time = time.time()
        if rcsb_embedding[0] is None:
            print(f"Ignoring {random_id}")
            continue
        search_result = embedding_provider.get_by_embedding(
            collection_name,
            rcsb_embedding[0],
            query_length=None,
            n_results=n_results,
            param={
                "metric_type": "IP",
                "params": {}
            }
        )
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function execution time: {execution_time:.6f} seconds")
        times.append(execution_time)

    (b_int, t_int) = confidence_interval(times)
    print(0.5*(b_int+t_int), 0.5*(t_int-b_int))
