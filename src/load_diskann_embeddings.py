import os

import pandas as pd
from tqdm import tqdm

from embedding_diskann_loader import EmbeddingLoader
import concurrent.futures


af_embedding_folder = "/mnt/vdc1/computed-models/embeddings"
dim = 1280
embedding_loader = EmbeddingLoader()


def insert_file(file):
    embedding_loader.read_df(pd.read_pickle(file))
    return f"Loaded {file}"


def main():

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(insert_file, f'{af_embedding_folder}/{df}') for df in os.listdir(af_embedding_folder)]
        with tqdm(total=len(futures), desc="Loading embeddings", unit="file") as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)


if __name__ == '__main__':
    main()
