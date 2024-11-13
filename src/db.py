import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import pandas as pd


client = QdrantClient(host="localhost", port=6333)


def init_db_collection(chain_path, csm_path, assembly_path):
    create_collection("chain_collection")
    create_collection("csm_collection")
    create_collection("assembly_collection")
    load_path_to_collection(chain_path, "csm_collection", "chain_collection")
    load_path_to_collection(csm_path, "csm_collection")
    load_path_to_collection(assembly_path, "assembly_collection")


def create_collection(collection_name):
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1280,
            distance=Distance.COSINE
        ),
    )


def display_progress(current, total):
    percent = (current / total) * 100
    print(f"\rProgress: {percent:.2f}% ({current}/{total} files)", end='')
    if current == total:
        print()


def load_path_to_collection(embedding_path, *db_collection_list):
    print(f"DB loading data from path {embedding_path}")
    files = os.listdir(embedding_path)
    buffer = {
        "ids": [],
        "embeddings": []
    }
    buffer_size = 40000
    for idx, r in enumerate(files):
        instance_id = file_name(r)
        v = list(pd.read_csv(f"{embedding_path}/{r}").iloc[:, 0].values)
        buffer["ids"].append(instance_id)
        buffer["embeddings"].append(v)
        if idx % buffer_size == 0:
            for db_collection in db_collection_list:
                client.upsert(
                    collection_name=db_collection,
                    points=[
                        PointStruct(
                            id=idx,
                            vector=vector,
                            payload={"color": "red", "rand_number": idx % 10}
                        )
                        for idx, vector in zip(buffer["ids"], buffer["embeddings"])
                    ]
                )
            buffer["embeddings"] = []
            buffer["ids"] = []
        display_progress(idx+1, len(files))
    if len(buffer["embeddings"]) > 0:
        for db_collection in db_collection_list:
            client.upsert(
                collection_name=db_collection,
                points=[
                    PointStruct(
                        id=idx,
                        vector=vector.tolist(),
                        payload={"color": "red", "rand_number": idx % 10}
                    )
                    for idx, vector in zip(buffer["ids"], buffer["embeddings"])
                ]
            )
    print(f"DB load path {embedding_path} done")


def file_name(file):
    return os.path.splitext(file)[0]