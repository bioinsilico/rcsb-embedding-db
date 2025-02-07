import os
import random

from embedding_provider import EmbeddingProvider

embedding_path = "/mnt/vdb1/embedding-33855646"
collection_name = "instance_embeddings"

if __name__ == '__main__':
    embedding_provider = EmbeddingProvider(collection_name)
    n_results = 100
    random_id = ".".join(random.choice(os.listdir(embedding_path)).split(".")[0:2])
    rcsb_embedding = embedding_provider.get_by_id(random_id)
    search_result = embedding_provider.get_by_embedding(
        rcsb_embedding,
        True,
        n_results
    )
    pass
