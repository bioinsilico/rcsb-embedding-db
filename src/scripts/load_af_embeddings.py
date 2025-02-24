import argparse
import os

import pandas as pd
from tqdm import tqdm

from utils.embedding_af_loader import EmbeddingLoader
import concurrent.futures

dim = 1536


def main(af_embedding_folder):

    embedding_loader = EmbeddingLoader(
        'af_embeddings',
        dim
    )

    def __insert_file(file):
        embedding_loader.insert_df(pd.read_pickle(file))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(__insert_file, f'{af_embedding_folder}/{df}') for df in os.listdir(af_embedding_folder)]
        with tqdm(total=len(futures), desc="Loading embeddings", unit="file") as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)

    embedding_loader.flush()
    embedding_loader.index_collection()
    embedding_loader.load_collection()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Embedding Search.")
    parser.add_argument('--af_embedding_folder', type=str, help="Embeddings folder", required=True)
    args = parser.parse_args()
    main(args.af_embedding_folder)
