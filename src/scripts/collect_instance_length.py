import argparse
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.length_collector import get_instance_length


def pdb_file_path(pdb_path, pdb):
    if "_" in pdb:
        return os.path.join(
            pdb_path,
            "csm",
            pdb[0:2],
            pdb[-6:-4],
            pdb[-4:-2],
            f"{pdb}.bcif.gz"
        )
    return os.path.join(pdb_path, "pdb", pdb[1:3], f"{pdb}.bcif.gz")


def process_file(args_tuple):
    filename, asym_id_list, pdb = args_tuple
    if not os.path.isfile(filename):
        print(f"File not found: {filename}")
        return None
    # Compute the instance length for the file
    length_list = get_instance_length(filename, asym_id_list)
    return [(f"{pdb}.{asym_id}", length) for asym_id, length in length_list]


def parse_instances(instance_list):
    chain_map = {}
    for rcsb_id in open(instance_list):
        pdb = rcsb_id.split(".")[0].lower()
        asym_id = rcsb_id.split(".")[1].strip()
        if pdb in chain_map:
            chain_map[pdb].append(asym_id)
        else:
            chain_map[pdb] = [asym_id]
    return chain_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test length collectors")
    parser.add_argument('--pdb_path', type=str, help="Embeddings folder", required=True)
    parser.add_argument('--instance_list', type=str, help="List of chain file", required=True)
    parser.add_argument('--instance_length_output_file', type=str, help="Instance length output file", required=True)
    args = parser.parse_args()

    pdb_path = args.pdb_path
    instance_list = args.instance_list
    instance_length_output_file = args.instance_length_output_file

    # Prepare the list of file information
    chain_map = parse_instances(instance_list)
    folder_files = [(
        pdb_file_path(pdb_path, pdb.lower()),
        asym_id_list,
        pdb.lower()
    ) for (pdb, asym_id_list) in chain_map.items()]

    num_cpus = os.cpu_count()
    print(f"Using {num_cpus} CPU cores for processing.")

    BATCH_SIZE = 10000

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = {executor.submit(process_file, item): item for item in folder_files}
        with tqdm(total=len(folder_files), desc="Loading instance length", unit="file") as pbar:
            with open(instance_length_output_file, "w") as f:
                batch_results = []
                for future in as_completed(futures):
                    result = future.result()
                    pbar.update(1)
                    if result is None:
                        continue
                    batch_results.extend([f"{rcsb_id},{length}\n" for (rcsb_id, length) in result])
                    if len(batch_results) >= BATCH_SIZE:
                        f.writelines(batch_results)
                        f.flush()
                        batch_results = []

                if batch_results:
                    f.writelines(batch_results)
                    f.flush()
