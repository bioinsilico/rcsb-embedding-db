import argparse

from utils.embedding_loader import EmbeddingLoader


dim = 1536


def main(args):

    embedding_root = args.embedding_root
    embedding_tag = args.embedding_tag
    instance_length_file = args.instance_length_file
    assembly_length_file = args.assembly_length_file

    embedding_loader = EmbeddingLoader(
        "instance_embeddings",
        dim
    )

    embedding_loader.load_lengths(instance_length_file)
    embedding_loader.insert_folder(f"{embedding_root}/embedding-{embedding_tag}", False)
    embedding_loader.insert_folder(f"{embedding_root}/csm-{embedding_tag}", True)
    embedding_loader.flush()
    embedding_loader.index_collection()

    embedding_loader = EmbeddingLoader(
        "assembly_embeddings",
        dim
    )

    embedding_loader.load_lengths(assembly_length_file)
    embedding_loader.insert_folder(f"{embedding_root}/assembly-{embedding_tag}", False)
    embedding_loader.flush()
    embedding_loader.index_collection()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Embedding Search.")
    parser.add_argument('--embedding_root', type=str, help="Embeddings folder", required=True)
    parser.add_argument('--embedding_tag', type=str, help="Embeddings folder", required=True)
    parser.add_argument('--instance_length_file', type=str, help="PDB instance lengths file", required=True)
    parser.add_argument('--assembly_length_file', type=str, help="PDB assembly lengths file", required=True)
    args = parser.parse_args()
    main(args)
