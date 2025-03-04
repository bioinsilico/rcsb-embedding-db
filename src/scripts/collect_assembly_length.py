import argparse
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from scripts.collect_instance_length import pdb_file_path
from utils.length_collector import get_instance_length, get_assembly_length


def process_file(args_tuple):
    filename, rcsb_id, assembly_id = args_tuple
    if not os.path.isfile(filename):
        print(f"File not found: {filename}")
        return None
    # Compute the instance length for the file
    return rcsb_id, get_assembly_length(filename, assembly_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test length collectors")
    parser.add_argument('--pdb_path', type=str, help="Embeddings folder", required=True)
    parser.add_argument('--assembly_list', type=str, help="List of chain file", required=True)
    parser.add_argument('--assembly_length_output_file', type=str, help="Instance length output file", required=True)
    args = parser.parse_args()

    pdb_path = args.pdb_path
    assembly_list = args.assembly_list
    assembly_length_output_file = args.assembly_length_output_file

    num_cpus = os.cpu_count()
    print(f"Using {num_cpus} CPU cores for processing.")

    folder_files = [(pdb_file_path(pdb_path, rcsb_id.split("-")[0]), rcsb_id.strip(), rcsb_id.split("-")[1].strip()) for rcsb_id in open(assembly_list)]
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = {executor.submit(process_file, item): item for item in folder_files}
        with tqdm(total=len(folder_files), desc="Loading instance length", unit="file") as pbar:
            with open(assembly_length_output_file, 'w') as f:
                for future in as_completed(futures):
                    rcsb_id, assembly_length = future.result()
                    f.write(f"{rcsb_id},{assembly_length}\n")
                    pbar.update(1)
