import argparse
import random
import struct
import time
import numpy as np

from utils.embedding_provider import EmbeddingProvider


def bytes_to_float16_list(byte_sequence):
    """
    Converts a sequence of bytes to a list of float16 values.

    Args:
        byte_sequence: A bytes object representing the sequence of bytes.

    Returns:
        A list of float16 values.
    """
    float16_list = []
    for i in range(0, len(byte_sequence), 2):
        try:
            float16_value = struct.unpack("<e", byte_sequence[i:i + 2])[0]
            float16_list.append(float16_value)
        except struct.error:
            # Handle cases where there are insufficient bytes
            break
    return np.array(float16_list).astype(np.float16)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--collection_name', type=int, required=True)
    parser.add_argument('--up_ids_list_file', type=int, required=True)
    parser.add_argument('--n_queries', type=int, required=True)
    parser.add_argument('--n_results', type=int, required=True)
    args = parser.parse_args()

    collection_name = args.collection_name
    up_ids_list_file = args.up_ids_list_file
    n_queries = args.n_queries
    n_results = args.n_results

    embedding_provider = EmbeddingProvider(collection_name)

    embedding_files = [f"AF_AF{r.strip()}F1" for r in open(up_ids_list_file)]
    random_queries = []
    for acc in random.sample(embedding_files, n_queries):
        rcsb_embedding = embedding_provider.get_by_id(acc)
        random_queries.append(bytes_to_float16_list(rcsb_embedding[0]))

    start_time = time.time()
    search_result = embedding_provider.get_by_multi_embedding(
        random_queries,
        True,
        n_results,
        param={
            "metric_type": "IP",
            "params": {}
        }
    )
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time:.6f} seconds")
