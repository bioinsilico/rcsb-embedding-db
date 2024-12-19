import os

import pandas as pd
from tqdm import tqdm

from embedding_diskann_loader import EmbeddingLoader


af_embedding_folder = "/mnt/vdc1/computed-models/embeddings"
diskann_tmp_folder = "/mnt/raid0/DiskANN/tmp"
dim = 1280


def main():
    total_len = 0
    df_files = os.listdir(af_embedding_folder)[0:10]
    with tqdm(total=len(df_files), desc="Loading embeddings", unit="file") as pbar:
        for df in df_files:
            total_len += len(pd.read_pickle(f'{af_embedding_folder}/{df}'))
            pbar.update(1)
    pbar.close()
    print(f"Saving {total_len} embeddings")

    embedding_loader = EmbeddingLoader(diskann_tmp_folder, total_len, dim)
    with tqdm(total=len(df_files), desc="Loading embeddings", unit="file") as pbar:
        for df in df_files:
            embedding_loader.add_to_bin(pd.read_pickle(f'{af_embedding_folder}/{df}'))
            pbar.update(1)


if __name__ == '__main__':
    main()
