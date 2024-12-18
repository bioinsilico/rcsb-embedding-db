import os

import pandas as pd
from tqdm import tqdm

from embedding_diskann_loader import EmbeddingLoader
import concurrent.futures


af_embedding_folder = "/mnt/vdc1/computed-models/embeddings"
diskann_tmp_folder = "/mnt/raid0/DiskANN/tmp"
dim = 1280
embedding_loader = EmbeddingLoader(diskann_tmp_folder)


def insert_file(file, prefix):
    embedding_loader.save_to_tsv(pd.read_pickle(file), prefix)
    return f"Loaded {file}"


def main():

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(insert_file, f'{af_embedding_folder}/{df}', df.split(".")[0]) for df in os.listdir(af_embedding_folder)]
        with tqdm(total=len(futures), desc="Loading embeddings", unit="file") as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)


if __name__ == '__main__':
    main()
