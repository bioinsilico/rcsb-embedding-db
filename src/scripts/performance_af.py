import argparse
import random
import time

import numpy as np

from scripts.performance_test import confidence_interval
from utils.embedding_provider import EmbeddingProvider, MilvusCollection


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rcsb_host', type=str, required=True)
    parser.add_argument('--afdb_host', type=str, required=True)
    parser.add_argument('--up_ids_list_file', type=str, required=True)
    parser.add_argument('--n_queries', type=int, required=True)
    parser.add_argument('--n_results', type=int, required=True)
    args = parser.parse_args()

    rcsb_host = args.rcsb_host
    afdb_host = args.afdb_host
    up_ids_list_file = args.up_ids_list_file
    n_queries = args.n_queries
    n_results = args.n_results

    embedding_provider = EmbeddingProvider()
    embedding_provider.connect(
        rcsb_host,
        afdb_host
    )

    embedding_files = [f"AF-{r.strip()}-F1" for r in open(up_ids_list_file)]
    times = []
    for _ in range(n_queries):
        random_queries = []
        acc = random.sample(embedding_files, 1)[0]
        rcsb_embedding = embedding_provider.get_by_id(
            MilvusCollection.af_collection,
            acc
        )
        start_time = time.time()
        if rcsb_embedding[0] is None:
            print(f"Ignoring {acc}")
            continue
        search_result = embedding_provider.get_by_embedding(
            MilvusCollection.af_collection,
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