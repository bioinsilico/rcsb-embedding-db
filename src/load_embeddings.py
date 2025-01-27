from embedding_loader import EmbeddingLoader


dim = 1536
tag = "33855646"


def main():

    embedding_loader = EmbeddingLoader(
        "instance_embeddings",
        dim
    )
    embedding_loader.insert_folder(f"/mnt/vdb1/embedding-{tag}", False)
    embedding_loader.insert_folder(f"/mnt/vdb1/csm-{tag}", True)
    embedding_loader.flush()
    embedding_loader.index_collection()

    embedding_loader = EmbeddingLoader(
        "assembly_embeddings",
        dim
    )
    embedding_loader.insert_folder(f"/mnt/vdb1/assembly-{tag}", False)
    embedding_loader.flush()
    embedding_loader.index_collection()


if __name__ == '__main__':
    main()
