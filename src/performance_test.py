import argparse
import os
import random
import time

from embedding_provider import EmbeddingProvider

embedding_path = "/mnt/vdb1/embedding-33855646"
collection_name = "instance_embeddings"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_queries', type=int, required=True)
    parser.add_argument('--n_results', type=int, required=True)
    args = parser.parse_args()

    n_queries = args.n_queries
    n_results = args.n_results

    embedding_provider = EmbeddingProvider(collection_name)

    embedding_files = list(os.listdir(embedding_path))
    random_queries = []
    for f in random.sample(embedding_files, n_queries):
        random_id = ".".join(f.split(".")[0:2])
        rcsb_embedding = embedding_provider.get_by_id(random_id)
        random_queries.append(rcsb_embedding)

    start_time = time.time()
    for rcsb_embedding in random_queries:
        search_result = embedding_provider.get_by_embedding(
            rcsb_embedding,
            True,
            n_results
        )
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time:.6f} seconds")
