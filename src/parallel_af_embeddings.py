import os

import pandas as pd
from tqdm import tqdm

import multiprocessing as mp
import numpy as np

from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection, list_collections, utility
)


AF_EMBEDDING_FOLDER = "/mnt/vdc1/computed-models/embeddings"
COLLECTION_NAME = 'af_embeddings'
DIM = 1280
HOST = 'localhost'
PORT = '19530'
ID_FIELD = 'id'
EMBEDDING_FIELD = 'embedding'
BATCH_SIZE = 2000


def connect():
    connections.connect(
        host=HOST,
        port=PORT
    )


def create_collection():
    id_field = FieldSchema(
        name=ID_FIELD,
        dtype=DataType.VARCHAR,
        is_primary=True,
        max_length=100  # Adjust max_length based on your identifier length
    )
    embedding_field = FieldSchema(
        name=EMBEDDING_FIELD,
        dtype=DataType.FLOAT_VECTOR,
        dim=DIM
    )
    collection_schema = CollectionSchema(
        fields=[id_field, embedding_field],
        description="Collection storing embeddings with cosine distance."
    )
    if COLLECTION_NAME in list(list_collections()):
        utility.drop_collection(COLLECTION_NAME)
    Collection(name=COLLECTION_NAME, schema=collection_schema)


def flush():
    Collection(name=COLLECTION_NAME).flush()


def index_collection():
    # Create an index on the embedding field with cosine distance
    index_params = {
        "metric_type": "IP",
        "index_type": "DISKANN",  # You can choose other index types as needed
        "params": {}
    }
    Collection(name=COLLECTION_NAME).create_index(
        field_name=EMBEDDING_FIELD,
        index_params=index_params
    )
    print("Index created with cosine distance metric.")


def insert_file(file):
    df = pd.read_pickle(file)
    if not {ID_FIELD, EMBEDDING_FIELD}.issubset(df.columns):
        raise ValueError(f"DataFrame must contain '{ID_FIELD}' and '{EMBEDDING_FIELD}' columns.")

    batch_size = BATCH_SIZE
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size  # Calculate the number of batches needed
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]

        ids = batch_df[ID_FIELD].tolist()
        embeddings = [(embedding/np.linalg.norm(embedding)).tolist() for embedding in batch_df[EMBEDDING_FIELD]]

        entities = [
            ids,         # List of identifiers
            embeddings   # List of embeddings
        ]
        Collection(name=COLLECTION_NAME).insert(entities)
    return f"Loaded {file}"


def main():
    connect()
    create_collection()

    num_processes = mp.cpu_count()
    embedding_files = [f'{AF_EMBEDDING_FOLDER}/{df}' for df in os.listdir(AF_EMBEDDING_FOLDER)]
    with tqdm(total=len(embedding_files), desc="Loading embeddings", unit="file") as pbar:
        with mp.Pool(processes=num_processes) as pool:
            for _ in pool.imap_unordered(insert_file,  embedding_files):
                pbar.update(1)
    flush()
    index_collection()


if __name__ == '__main__':
    main()
