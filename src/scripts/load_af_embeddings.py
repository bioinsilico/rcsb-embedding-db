import argparse
import os
from random import shuffle

import pandas as pd
from tqdm import tqdm

from utils.embedding_af_loader import EmbeddingLoader
import concurrent.futures

dim = 1536


def main(af_embedding_folder, index_only, index_collection, load_collection):

    embedding_loader = EmbeddingLoader(
        'af_embeddings',
        dim
    )

    if index_only:
        embedding_loader.index_collection()
        if load_collection:
            embedding_loader.load_collection()
        return

    def __insert_file(file):
        embedding_loader.insert_df(pd.read_pickle(file))

    embedding_loader.create_embedding_collection()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        af_files = [df for df in os.listdir(af_embedding_folder)]
        shuffle(af_files)
        futures = [executor.submit(__insert_file, f'{af_embedding_folder}/{df}') for df in af_files]
        with tqdm(total=len(futures), desc="Loading embeddings", unit="file") as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)

    embedding_loader.flush()
    embedding_loader.compact()
    if index_collection:
        embedding_loader.index_collection()
    if load_collection:
        embedding_loader.load_collection()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Embedding Search.")
    parser.add_argument('--af_embedding_folder', type=str, help="Embeddings folder", required=True)
    parser.add_argument('--index_only', action='store_true')
    parser.add_argument('--index_collection', action='store_true')
    parser.add_argument('--load_collection', action='store_true')
    args = parser.parse_args()
    main(args.af_embedding_folder, args.index_only, args.index_collection, args.load_collection)
