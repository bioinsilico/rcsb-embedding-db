import os

import pandas as pd
from tqdm import tqdm

from embedding_af_loader import EmbeddingLoader
import concurrent.futures


af_embedding_folder = "/mnt/vdc1/computed-models/embeddings"
collection_name = 'af_embeddings'
dim = 1280
embedding_loader = EmbeddingLoader(
    collection_name,
    dim
)


def insert_file(file):
    embedding_loader.insert_df(pd.read_pickle(file))
    return f"Loaded {file}"


def main():

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(insert_file, f'{af_embedding_folder}/{df}') for df in os.listdir(af_embedding_folder)]
        with tqdm(total=len(futures), desc="Loading embeddings", unit="file") as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)

    embedding_loader.flush()
    embedding_loader.index_collection()
    embedding_loader.load_collection()


if __name__ == '__main__':
    main()
